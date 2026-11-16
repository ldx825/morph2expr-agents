"""智能体 2 —— 形态特征提取（Morphology Agent）。

职责
----
把清洗后的孔级形态特征变成可建模的 `X_morph`：

1. **批内 z 标准化**：同一孔板内做 z 分数，抵消染色强度/曝光/批次差异。
   这是 Cell Painting 领域公认的标准做法（不做的话模型会先学"这是哪块板"）。
2. **孔级 → 扰动级聚合**：同一扰动的多个重复取均值。
   任务书明确要求"须先聚合到孔级/扰动级"，用单细胞或单孔直接比是典型误区。
3. **对齐到共有扰动集合**：只保留同时具备形态与转录组两模态的扰动。

输出
----
`X_morph`：行 = perturbation_id，列 = 形态特征
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import get_logger

log = get_logger("agent2.morph")


def normalize_within_batch(X: pd.DataFrame, batch: pd.Series,
                           clip_z: float = 10.0) -> pd.DataFrame:
    """按批次（孔板）做 z 标准化并裁剪极端值。"""
    out = X.copy().astype(np.float64)
    for b, idx in batch.groupby(batch).groups.items():
        sub = out.loc[list(idx)]
        mu = sub.mean(axis=0)
        sd = sub.std(axis=0, ddof=0).replace(0, np.nan)
        z = (sub - mu) / sd
        z = z.clip(-clip_z, clip_z)
        out.loc[list(idx)] = z.fillna(0.0)
    return out


def aggregate_to_perturbation(X: pd.DataFrame, meta: pd.DataFrame,
                              how: str = "mean") -> pd.DataFrame:
    """孔级 → 扰动级聚合。"""
    df = X.copy()
    df["perturbation_id"] = meta["perturbation_id"].values
    df["pert_type"] = meta["pert_type"].values
    df = df[df["pert_type"] != "control"]
    if df.empty:
        return pd.DataFrame(columns=X.columns)
    grp = df.groupby("perturbation_id", sort=True)
    agg = grp.median(numeric_only=True) if how == "median" else grp.mean(numeric_only=True)
    return agg


def build(cp_features: pd.DataFrame, cp_meta: pd.DataFrame, cfg: dict,
          min_replicates: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    m_cfg = cfg["morph"]
    min_rep = int(min_replicates if min_replicates is not None
                  else cfg["qc"]["min_replicates"])

    X = cp_features
    if m_cfg.get("normalize_within_batch", True):
        X = normalize_within_batch(X, cp_meta["batch"], float(m_cfg["clip_z"]))

    n_rep = (cp_meta[cp_meta.pert_type != "control"]
             .groupby("perturbation_id").size().rename("n_replicates"))

    X_pert = aggregate_to_perturbation(X, cp_meta, str(m_cfg["aggregate"]))

    keep = n_rep[n_rep >= min_rep].index
    X_pert = X_pert.loc[X_pert.index.intersection(keep)]

    n_rep = n_rep.loc[X_pert.index]

    # 聚合后的稳健标准化，便于后续线性模型收敛
    sd = X_pert.std(axis=0, ddof=0).replace(0, np.nan)
    X_pert = ((X_pert - X_pert.mean(axis=0)) / sd).fillna(0.0)

    log.info("[morph] X_morph: %d 扰动 x %d 特征（每扰动至少 %d 个重复）",
             X_pert.shape[0], X_pert.shape[1], min_rep)

    meta_pert = pd.DataFrame({
        "perturbation_id": X_pert.index,
        "n_replicates": n_rep.reindex(X_pert.index).astype(int).values,
    })
    return X_pert, meta_pert
