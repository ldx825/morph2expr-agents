"""智能体 1 —— 图像/特征质控（QC Agent）。

职责
----
输入：Cell Painting 原始 replicate 级特征表
输出：清洗后的特征矩阵 + 质控报告

四步质控
--------
1. **黑名单过滤**：剔除孔板坐标、细胞质心、对象编号等非生物学特征
   （这些特征是"实验记录本"，会直接泄漏批次信息）
2. **低方差过滤**：丢弃几乎不变的通道/特征
3. **高相关过滤**：|r| 超过阈值的冗余特征只保留一个
4. **孔级离群剔除**：对前若干主成分做稳健 z 分数检验，剔除染色失败/脱片孔

说明（诚实声明）
----------------
本项目使用 Cell Painting Gallery 已发布的 CellProfiler 特征，
**不重新做细胞分割与特征计算**。因此本智能体的"质控"作用在
特征层面与孔层面，而非原始图像层面。这一点在报告的"局限"一节必须写明。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .common import canonical_meta, feature_columns, get_logger

log = get_logger("agent1.qc")


@dataclass
class QCResult:
    features: pd.DataFrame            # 行=replicate，列=保留特征
    meta: pd.DataFrame                # 行=replicate，统一 schema
    report: dict = field(default_factory=dict)
    kept_features: list[str] = field(default_factory=list)
    dropped: dict[str, list[str]] = field(default_factory=dict)


def _blocklist(feature_names: list[str], patterns: list[str]) -> list[str]:
    rx = [re.compile(p) for p in patterns]
    return [f for f in feature_names if any(r.match(f) for r in rx)]


def _variance_filter(df: pd.DataFrame, low_quantile: float) -> tuple[list[str], list[str]]:
    var = df.var(axis=0, skipna=True).fillna(0.0)
    finite = var[var > 0]
    if finite.empty:
        return [], list(df.columns)
    cutoff = finite.quantile(low_quantile)
    keep = [c for c in df.columns if var[c] > cutoff]
    drop = [c for c in df.columns if c not in set(keep)]
    return keep, drop


def _correlation_filter(df: pd.DataFrame, threshold: float,
                        seed: int = 0) -> tuple[list[str], list[str]]:
    """贪心去冗余：按方差从大到小保留，与已保留特征相关过高的丢弃。"""
    if df.shape[1] <= 1:
        return list(df.columns), []
    var = df.var(axis=0).fillna(0.0)
    order = list(var.sort_values(ascending=False).index)
    corr = df.corr(method="pearson").abs().fillna(0.0)

    keep: list[str] = []
    drop: list[str] = []
    for col in order:
        if not keep:
            keep.append(col)
            continue
        if (corr.loc[keep, col] > threshold).any():
            drop.append(col)
        else:
            keep.append(col)
    # 保持原始列顺序
    keep = [c for c in df.columns if c in set(keep)]
    return keep, drop


def _outlier_filter(df: pd.DataFrame, n_pc: int, z_cut: float,
                    seed: int = 0) -> tuple[np.ndarray, dict]:
    """用前若干主成分的稳健 z 分数（基于中位数/MAD）标记离群孔。"""
    from sklearn.decomposition import PCA

    X = df.to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if X.shape[1] < 2 or X.shape[0] < 10:
        return np.ones(X.shape[0], dtype=bool), {"n_outlier": 0, "n_total": X.shape[0]}

    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    n_pc = int(min(n_pc, Xs.shape[1], Xs.shape[0] - 1))
    pcs = PCA(n_components=n_pc, random_state=seed).fit_transform(Xs)

    med = np.median(pcs, axis=0)
    mad = np.median(np.abs(pcs - med), axis=0) * 1.4826
    mad = np.where(mad < 1e-9, 1e-9, mad)
    z = np.abs(pcs - med) / mad
    keep_mask = (z <= z_cut).all(axis=1)
    return keep_mask, {
        "n_outlier": int((~keep_mask).sum()),
        "n_total": int(len(keep_mask)),
        "pcs_used": n_pc,
        "z_cut": z_cut,
    }


def run(cp_raw: pd.DataFrame, dataset: str, cfg: dict,
        genetic_key: str = "gene") -> QCResult:
    qc_cfg = cfg["qc"]
    seed = int(cfg["seed"])

    # ---- 1. 元数据 / 特征分离 -------------------------------------------
    meta = canonical_meta(cp_raw, dataset, kind="cp", genetic_key=genetic_key)
    if len(meta) != len(cp_raw):
        raise RuntimeError("元数据行数与原始表不一致")

    feats = feature_columns(cp_raw, [c for c in cp_raw.columns if c.startswith("Metadata_")])
    X = cp_raw[feats].apply(pd.to_numeric, errors="coerce")
    log.info("[%s] 原始: %d 孔 x %d 特征", dataset, X.shape[0], X.shape[1])

    # ---- 0. 缺失值插补 ---------------------------------------------------
    # 非数值/空字符串会被 to_numeric 转成 NaN。用逐特征中位数插补，
    # 避免后续 PCA / UMAP / 线性模型因 NaN 报错。
    n_nan = int(X.isna().sum().sum())
    if n_nan:
        X = X.fillna(X.median(numeric_only=True))
        X = X.fillna(0.0)
        log.info("[%s] 插补缺失值 %d 个（占 %.3f%%）", dataset, n_nan,
                 100.0 * n_nan / max(X.size, 1))

    dropped: dict[str, list[str]] = {}

    # ---- 2. 黑名单 -------------------------------------------------------
    bad = _blocklist(feats, qc_cfg["blocklist_regex"])
    dropped["blocklist"] = bad
    feats = [f for f in feats if f not in set(bad)]
    X = X[feats]

    # ---- 3. 低方差 -------------------------------------------------------
    keep_var, drop_var = _variance_filter(X, float(qc_cfg["variance_low_quantile"]))
    dropped["low_variance"] = drop_var
    feats = keep_var
    X = X[feats]

    # ---- 4. 高相关 -------------------------------------------------------
    keep_corr, drop_corr = _correlation_filter(X, float(qc_cfg["corr_threshold"]), seed)
    dropped["high_correlation"] = drop_corr
    feats = keep_corr
    X = X[feats]

    # ---- 5. 孔级离群 -----------------------------------------------------
    mask, out_stats = _outlier_filter(
        X, int(qc_cfg["outlier_pc_count"]), float(qc_cfg["outlier_z"]), seed)
    X = X.loc[mask].reset_index(drop=True)
    meta = meta.loc[mask].reset_index(drop=True)
    dropped["outlier_replicates"] = [meta_well for meta_well in []]

    report = {
        "dataset": dataset,
        "n_replicates_raw": int(len(cp_raw)),
        "n_replicates_final": int(X.shape[0]),
        "n_features_raw": int(len(feature_columns(cp_raw, [c for c in cp_raw.columns if c.startswith("Metadata_")]))),
        "n_features_final": int(X.shape[1]),
        "dropped_counts": {k: len(v) for k, v in dropped.items()},
        "outlier_stats": out_stats,
        "n_perturbations": int(meta.loc[meta.pert_type != "control", "perturbation_id"].nunique()),
        "n_controls": int((meta.pert_type == "control").sum()),
        "n_missing_imputed": n_nan,
        "cell_lines": sorted(meta.cell_line.dropna().unique().tolist()),
        "batches": int(meta.batch.nunique()),
    }
    log.info("[%s] 质控后: %d 孔 x %d 特征 | 扰动 %d 个 | 对照 %d 孔",
             dataset, X.shape[0], X.shape[1], report["n_perturbations"],
             report["n_controls"])
    return QCResult(features=X, meta=meta, report=report,
                    kept_features=list(X.columns), dropped=dropped)
