"""智能体 6 —— 批次效应诊断与校正（Batch Diagnosis Agent）。

核心思想
--------
批次效应 = 技术噪声伪装成生物学信号。本智能体做三件事：

1. **诊断**：用形态特征训练一个"批次分类器"。
   如果它很容易猜出孔板编号，说明数据里批次信息很重。
   判别标准：准确率越接近 `1/n_batches`（随机猜）越好。
   ⚠️ **批次分类准确率是越低越好**——这是本作业最容易搞反的地方。

2. **校正**：参数化 ComBat（见 `combat.py`）。

3. **校正前后对照**（关键，防止"过校正"）：
   同时监控两个指标
     * 批次可分性 ↓（希望降）
     * 扰动可分性 ↑/持平（希望别被一起抹掉）
   如果批次准确率降了、但扰动可分性也崩了，就是**过校正**。

输出
----
* 批次诊断指标表（校正前/后）
* 校正后的形态特征矩阵
* UMAP 坐标（供出图）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .combat import fit_transform
from .common import get_logger

log = get_logger("agent6.batch")


def _batch_labels(meta: pd.DataFrame, definition: str) -> np.ndarray:
    if definition == "dataset":
        return meta["dataset"].astype(str).to_numpy()
    return meta["batch"].astype(str).to_numpy()


def batch_classifier_accuracy(X: np.ndarray, labels: np.ndarray,
                              cfg: dict) -> dict:
    """交叉验证的批次分类准确率（含随机基线）。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    classes, counts = np.unique(labels, return_counts=True)
    n_batches = len(classes)
    chance = float(np.max(counts) / counts.sum())  # 多数类基线
    n_min = int(counts.min())

    if n_batches < 2 or n_min < 2 or X.shape[0] < 20:
        return {"n_batches": n_batches, "chance": chance, "accuracy": np.nan,
                "macro_f1": np.nan, "n_samples": int(X.shape[0]),
                "note": "批次不足，跳过"}

    b_cfg = cfg["batch"]
    if b_cfg["classifier"] == "knn":
        clf = Pipeline([("s", StandardScaler()),
                        ("c", KNeighborsClassifier(n_neighbors=int(b_cfg["knn_neighbors"])))])
    else:
        clf = Pipeline([("s", StandardScaler()),
                        ("c", LogisticRegression(max_iter=int(b_cfg["logistic_max_iter"]),
                                                 multi_class="multinomial",
                                                 n_jobs=-1))])

    n_splits = int(min(5, n_min))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(cfg["seed"]))
    try:
        acc = cross_val_score(clf, X, labels, cv=cv, scoring="accuracy", n_jobs=1)
        f1 = cross_val_score(clf, X, labels, cv=cv, scoring="f1_macro", n_jobs=1)
    except Exception as exc:  # noqa: BLE001
        log.warning("批次分类器失败: %s", exc)
        return {"n_batches": n_batches, "chance": chance, "accuracy": np.nan,
                "macro_f1": np.nan, "n_samples": int(X.shape[0]), "note": str(exc)}

    return {
        "n_batches": int(n_batches),
        "chance": chance,
        "accuracy": float(np.mean(acc)),
        "accuracy_std": float(np.std(acc)),
        "macro_f1": float(np.mean(f1)),
        "n_samples": int(X.shape[0]),
        "random_baseline_1_over_n": 1.0 / n_batches,
    }


def perturbation_retrieval(X: np.ndarray, pids: np.ndarray,
                           exclude_same_batch: bool = False,
                           batch: np.ndarray | None = None,
                           k: int = 1) -> tuple[float, float]:
    """扰动检索准确率：每个孔的 k 近邻里同扰动的比例。

    同时返回**随机基线**（把扰动标签视为等概率时的期望命中率），
    因为绝对数值本身没有意义——必须和基线比才知道生物学信号在不在。

    这是"生物学信号还在不在"的代理指标。校正过度时它会明显掉。
    """
    from sklearn.metrics import pairwise_distances

    n = X.shape[0]
    if n < 10:
        return np.nan, np.nan
    D = pairwise_distances(X)
    np.fill_diagonal(D, np.inf)
    if exclude_same_batch and batch is not None:
        same = batch[:, None] == batch[None, :]
        D[same] = np.inf
    nn = np.argsort(D, axis=1)[:, :k]
    hit = float((pids[nn] == pids[:, None]).mean())

    _, counts = np.unique(pids, return_counts=True)
    chance = float(((counts - 1) / (n - 1)).mean()) if n > 1 else np.nan
    return hit, chance


def run(X_rep: pd.DataFrame, meta_rep: pd.DataFrame, cfg: dict
        ) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """X_rep: 孔级形态特征；meta_rep: 与 X_rep 行对齐的统一元数据。"""
    b_cfg = cfg["batch"]
    definition = str(b_cfg["definition"])

    X = X_rep.to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    b = _batch_labels(meta_rep, definition)
    pids = meta_rep["perturbation_id"].astype(str).to_numpy()

    before = batch_classifier_accuracy(X, b, cfg)
    before["pert_retrieval_k1"], chance = perturbation_retrieval(X, pids, False, b, 1)
    before["pert_retrieval_chance"] = chance
    before["pert_retrieval_k1_cross_plate"], chance_x = perturbation_retrieval(
        X, pids, True, b, 1)
    before["pert_retrieval_cross_plate_chance"] = chance_x

    Xc, cstats = fit_transform(X, b, float(b_cfg["combat_strength"]),
                               bool(b_cfg.get("combat_use_eb", False)))

    after = batch_classifier_accuracy(Xc, b, cfg)
    after["pert_retrieval_k1"], _ = perturbation_retrieval(Xc, pids, False, b, 1)
    after["pert_retrieval_chance"] = before["pert_retrieval_chance"]
    after["pert_retrieval_k1_cross_plate"], _ = perturbation_retrieval(
        Xc, pids, True, b, 1)
    after["pert_retrieval_cross_plate_chance"] = before["pert_retrieval_cross_plate_chance"]

    diag = {
        "batch_definition": definition,
        "before": before,
        "after": after,
        "combat": cstats,
        "delta_accuracy": (float(before["accuracy"] - after["accuracy"])
                           if np.isfinite(before["accuracy"]) and np.isfinite(after["accuracy"])
                           else np.nan),
        "delta_pert_retrieval": (float(before["pert_retrieval_k1"] -
                                       after["pert_retrieval_k1"])
                                 if np.isfinite(before["pert_retrieval_k1"])
                                 and np.isfinite(after["pert_retrieval_k1"]) else np.nan),
    }
    log.info("[batch] 批次准确率 %.3f → %.3f（随机基线 %.3f）；扰动检索 %.3f → %.3f",
             before["accuracy"], after["accuracy"], before["chance"],
             before["pert_retrieval_k1"], after["pert_retrieval_k1"])

    Xc_df = pd.DataFrame(Xc, index=X_rep.index, columns=X_rep.columns)
    return Xc_df, diag, pd.DataFrame({"batch": b, "perturbation_id": pids},
                                     index=meta_rep.index)


def umap_embedding(X: np.ndarray, cfg: dict) -> np.ndarray:
    b_cfg = cfg["batch"]
    X = np.nan_to_num(np.asarray(X, dtype=np.float64),
                      nan=0.0, posinf=0.0, neginf=0.0)
    try:
        import umap

        reducer = umap.UMAP(
            n_neighbors=int(b_cfg["umap_neighbors"]),
            min_dist=float(b_cfg["umap_min_dist"]),
            n_components=2,
            random_state=int(cfg["seed"]),
            n_jobs=1,
        )
        return reducer.fit_transform(X)
    except Exception as exc:  # noqa: BLE001
        log.warning("UMAP 失败(%s)，退回 PCA 前两维", exc)
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=int(cfg["seed"])).fit_transform(X)
