"""智能体 3 —— 转录组预处理（Transcriptomics Agent）。

职责
----
把 L1000 原始值变成 `Y_expr`：

1. **对照基线校正**：每个基因减去同数据集对照孔（negcon / control）的均值，
   得到 **Δ 表达（扰动响应签名）**。
   这正是 L1000 level-5 签名的定义——level-3 只是归一化表达值，
   直接拿来当"扰动效应"回归会混入基线表达差异。
2. **稳健标准化**：以对照孔的离散度作为尺度做 z 变换，并裁剪极端值
   （本数据 level-3 存在 -76 这种离群，不裁剪会主导损失函数）。
3. **重复一致性过滤**：同一扰动各重复之间的中位 Spearman 相关过低说明
   信号不稳（噪声扰动），剔除。等价于 grit / e-distance 弱扰动过滤的简化版。
4. **扰动级聚合**。

输出
----
`Y_expr.parquet`（扰动级）、`Y_expr_replicate.parquet`（孔级，供批次分析用）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .common import canonical_meta, get_logger, safe_spearman

log = get_logger("agent3.expr")


def _robust_scale(mat: np.ndarray) -> np.ndarray:
    """逐列稳健离散度（MAD×1.4826），全零时回退到总体标准差。"""
    med = np.nanmedian(mat, axis=0)
    mad = np.nanmedian(np.abs(mat - med), axis=0) * 1.4826
    sd = np.nanstd(mat, axis=0)
    scale = np.where(mad > 1e-6, mad, sd)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return scale


def delta_signature(X: pd.DataFrame, is_control: np.ndarray,
                    clip_z: float) -> tuple[pd.DataFrame, dict]:
    """返回对照基线校正后的 Δ 表达矩阵（未标准化）与基线统计。"""
    ctrl = X.loc[is_control]
    if ctrl.empty:
        log.warning("该数据集没有对照孔，退回用全体中位数作为基线")
        base = X.median(axis=0)
        scale = _robust_scale(X.to_numpy(float))
    else:
        base = ctrl.mean(axis=0)
        scale = _robust_scale(ctrl.to_numpy(float))

    delta = (X - base) / scale
    delta = delta.clip(-clip_z, clip_z)
    stats = {
        "n_control_wells": int(is_control.sum()),
        "baseline_source": "control" if not ctrl.empty else "median_all",
        "n_extreme_clipped": int((np.abs(delta) >= clip_z).to_numpy().sum()),
    }
    return delta, stats


def replicate_consistency(delta: pd.DataFrame, meta: pd.DataFrame) -> pd.Series:
    """每个扰动的重复间中位 Spearman 相关（向量化实现）。

    等价写法：Spearman = 秩上的 Pearson。因此先把每个重复的表达向量转换成秩，
    再一次性算出整个重复×重复的相关矩阵，比逐对调用 `spearmanr` 快约两个数量级
    （主数据集有 1500+ 扰动 × 约 18 个重复，逐对调用需要 8 分钟以上）。
    """
    df = delta.copy()
    pid_col = meta["perturbation_id"].to_numpy()
    keep = (meta["pert_type"] != "control").to_numpy()
    df = df.loc[keep].reset_index(drop=True)
    pid_col = pid_col[keep]

    out: dict[str, float] = {}
    for pid, sub in df.groupby(pd.Series(pid_col)):
        vals = sub.to_numpy(float)
        n = vals.shape[0]
        if n < 2:
            out[pid] = np.nan
            continue
        # 逐行转秩（并列取平均秩），再算秩相关矩阵
        ranks = np.apply_along_axis(rankdata, 1, vals)
        with np.errstate(divide="ignore", invalid="ignore"):
            cm = np.corrcoef(ranks)
        iu = np.triu_indices(n, k=1)
        rs = cm[iu]
        rs = rs[np.isfinite(rs)]
        out[pid] = float(np.median(rs)) if rs.size else np.nan
    return pd.Series(out, name="replicate_spearman")


def build(l1k_raw: pd.DataFrame, dataset: str, cfg: dict,
          genetic_key: str = "gene"
          ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    e_cfg = cfg["expr"]
    meta = canonical_meta(l1k_raw, dataset, kind="l1k", genetic_key=genetic_key)

    gene_cols = [c for c in l1k_raw.columns if not c.startswith("Metadata_")]
    X = l1k_raw[gene_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    log.info("[%s] L1000 原始: %d 孔 x %d landmark 基因",
             dataset, X.shape[0], X.shape[1])

    is_ctrl = (meta["pert_type"] == "control").to_numpy()

    delta, base_stats = delta_signature(X, is_ctrl, float(e_cfg["clip_z"]))

    # ---- 重复一致性 ------------------------------------------------------
    cons = replicate_consistency(delta, meta)
    min_rho = float(e_cfg["min_replicate_spearman"])
    n_rep = meta[meta.pert_type != "control"].groupby("perturbation_id").size()

    keep = set(cons[(cons >= min_rho) & (n_rep >= 2)].index)

    # ---- 聚合 ------------------------------------------------------------
    d = delta.copy()
    d["perturbation_id"] = meta["perturbation_id"].values
    d["pert_type"] = meta["pert_type"].values
    treat = d[d["pert_type"] != "control"]
    Y_pert = treat.groupby("perturbation_id", sort=True)[gene_cols].mean()
    Y_pert = Y_pert.loc[Y_pert.index.intersection(keep)]

    # 扰动级再做一次标准化，让各基因量纲一致
    scale = Y_pert.std(axis=0, ddof=0).replace(0, np.nan)
    Y_pert = ((Y_pert - Y_pert.mean(axis=0)) / scale).fillna(0.0)

    # 孔级表（保留，供批次诊断与检索评估）
    Y_rep = delta.copy()
    Y_rep.index = meta["replicate_id"].values
    Y_rep.attrs = {}

    stats = {
        "dataset": dataset,
        "n_wells_raw": int(X.shape[0]),
        "n_genes": int(len(gene_cols)),
        "n_perturbations_raw": int(n_rep.shape[0]),
        "n_perturbations_final": int(Y_pert.shape[0]),
        "n_perturbations_dropped_noise": int(
            len(set(cons.index) - keep)),
        "dropped_single_replicate": int((n_rep < 2).sum()),
        "median_replicate_spearman": float(np.nanmedian(cons.values))
        if cons.notna().any() else None,
        **base_stats,
    }
    log.info("[%s] Y_expr: %d 扰动 x %d 基因（一致性阈值 ρ≥%.2f）",
             dataset, Y_pert.shape[0], Y_pert.shape[1], min_rho)

    meta_rep = meta.set_index("replicate_id", drop=False)
    meta_pert = pd.DataFrame({
        "perturbation_id": Y_pert.index,
        "replicate_spearman": cons.reindex(Y_pert.index).values,
        "n_replicates": n_rep.reindex(Y_pert.index).astype(int).values,
    })
    return Y_pert, Y_rep, {"stats": stats, "meta_replicate": meta_rep,
                           "meta_perturbation": meta_pert}
