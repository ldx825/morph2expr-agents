"""智能体 5 —— 跨模态预测（Cross-modal Prediction Agent）。

任务
----
学习 f(形态特征) → Δ基因表达，并在 **未见扰动** 上评估。

三种评估协议（难度递增）
------------------------
1. `perturbation_kfold`：把扰动随机分成 5 折，测试折里的扰动训练时完全没见过。
   ⇒ 回答"能不能外推到新药/新靶点"
2. `cluster_kfold`：先用**形态**无监督聚类，再整簇留出。
   ⇒ 回答"能不能外推到没见过的一类表型"（比 1 更难，也更接近真实应用）
3. `cross_dataset`：在一个细胞系上训练、另一个细胞系上测试。
   ⇒ 回答"换个细胞系还成立吗"（预期很差，**负面结果本身就是有价值的发现**）

指标
----
* 基因表达相关系数（逐基因 Pearson，取均值/中位）
* 基因 R² 分布
* 差异基因方向准确率（真值里"显著升降"的基因，方向猜对的比例）
* 近邻一致性（预测空间里的 top-k 邻居，与真值空间 top-k 的重合度）
* 置换负对照（打乱标签重跑，证明不是数据泄漏/偶然）

⚠️ **绝对不做的事**：把扰动身份（化合物 ID / 基因名）当作特征喂给模型。
   那在"未见扰动"任务上等于把答案写进输入，只能当"泄漏诊断"的对照组。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .common import get_logger, safe_pearson, set_seed

log = get_logger("agent5.predict")
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# 划分策略
# ---------------------------------------------------------------------------
def split_perturbation_kfold(n: int, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import KFold

    idx = np.arange(n)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in kf.split(idx)]


def split_cluster_kfold(X: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """按形态空间整簇留出。簇标签来自形态特征的无监督聚类。"""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    n = X.shape[0]
    k = int(min(n_splits, max(2, n // 5)))
    Z = PCA(n_components=int(min(50, X.shape[1], n - 1)), random_state=seed
            ).fit_transform(X)
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)

    folds = []
    for c in range(k):
        te = np.where(labels == c)[0]
        tr = np.where(labels != c)[0]
        if len(te) >= 5 and len(tr) >= 10:
            folds.append((tr, te))
    return folds or split_perturbation_kfold(n, n_splits, seed)


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
class MultiOutputLasso:
    """逐基因 Lasso（joblib 并行）。

    为什么不用 `MultiTaskLasso`：它把所有基因当成一个 group-lasso 同时求解，
    在 978 个输出上坐标下降极慢（单折需要数十分钟）。逐基因拟合在数学上
    是同一个“L1 正则多输出回归”的解施，且可完全并行，快 2~3 个数量级。
    """

    def __init__(self, alpha: float = 0.01, max_iter: int = 2000,
                 random_state: int = 0, n_jobs: int = 16):
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, Y):
        from joblib import Parallel, delayed
        from sklearn.linear_model import Lasso

        Y = np.asarray(Y, dtype=np.float64)

        def _one(j: int):
            m = Lasso(alpha=self.alpha, max_iter=self.max_iter,
                      random_state=self.random_state, tol=1e-3)
            m.fit(X, Y[:, j])
            return m

        self.models_ = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(_one)(j) for j in range(Y.shape[1]))
        self.coef_ = np.vstack([m.coef_ for m in self.models_]).T
        return self

    def predict(self, X) -> np.ndarray:
        return np.column_stack([m.predict(X) for m in self.models_])


def _make_model(name: str, cfg: dict, n_features: int, n_targets: int):
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    m_cfg = cfg["model"]
    seed = int(cfg["seed"])

    if name == "ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=10.0, random_state=seed)),
        ])

    if name == "lasso":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("reg", MultiOutputLasso(alpha=0.02, max_iter=2000,
                                      random_state=seed,
                                      n_jobs=int(m_cfg.get("lasso_n_jobs", 16)))),
        ])

    if name == "mlp":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("reg", MLPRegressor(
                hidden_layer_sizes=tuple(m_cfg["mlp_hidden"]),
                alpha=float(m_cfg["mlp_alpha"]),
                max_iter=int(m_cfg["mlp_max_iter"]),
                early_stopping=bool(m_cfg["mlp_early_stopping"]),
                validation_fraction=0.15,
                n_iter_no_change=15,
                random_state=seed,
                learning_rate_init=2e-3,
                batch_size=32,
            )),
        ])

    if name == "mean":
        return None  # 常数基线：预测训练集均值

    raise ValueError(f"未知模型: {name}")


def _fit_predict(name: str, Xtr, Ytr, Xte, cfg: dict) -> np.ndarray:
    if name == "mean":
        return np.tile(Ytr.mean(axis=0), (Xte.shape[0], 1))
    model = _make_model(name, cfg, Xtr.shape[1], Ytr.shape[1])
    model.fit(Xtr, Ytr)
    return np.asarray(model.predict(Xte))


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def gene_correlations(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    out = []
    for j in range(Y_true.shape[1]):
        out.append(safe_pearson(Y_true[:, j], Y_pred[:, j]))
    return np.asarray(out, dtype=float)


def gene_r2(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    ss_res = ((Y_true - Y_pred) ** 2).sum(axis=0)
    ss_tot = ((Y_true - Y_true.mean(axis=0)) ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - ss_res / ss_tot
    return np.where(np.isfinite(r2), r2, np.nan)


def dge_direction_accuracy(Y_true: np.ndarray, Y_pred: np.ndarray,
                           sd_mult: float = 1.0) -> tuple[float, int]:
    """真值中 |Δ| 超过 sd_mult 个标准差的基因视为"差异基因"，
    统计预测符号与真值符号一致的比例。"""
    accs, total = [], 0
    for i in range(Y_true.shape[0]):
        t = Y_true[i]
        thr = sd_mult * (np.nanstd(t) + 1e-12)
        sel = np.abs(t) > thr
        if sel.sum() < 3:
            continue
        agree = np.sign(Y_pred[i][sel]) == np.sign(t[sel])
        accs.append(float(agree.mean()))
        total += int(sel.sum())
    return (float(np.mean(accs)) if accs else np.nan), total


def neighbor_overlap(Y_true: np.ndarray, Y_pred: np.ndarray, k: int = 10) -> float:
    """预测空间与真值空间的 top-k 近邻重合度（越大越好，1.0 = 完全一致）。"""
    from sklearn.metrics import pairwise_distances

    n = Y_true.shape[0]
    if n < k + 2:
        return np.nan
    dt = pairwise_distances(Y_true)
    dp = pairwise_distances(Y_pred)
    np.fill_diagonal(dt, np.inf)
    np.fill_diagonal(dp, np.inf)
    nt = np.argsort(dt, axis=1)[:, :k]
    npp = np.argsort(dp, axis=1)[:, :k]
    scores = [len(set(a) & set(b)) / k for a, b in zip(nt, npp)]
    return float(np.mean(scores))


def summarize(Y_true, Y_pred, dge_sd: float) -> dict:
    r = gene_correlations(Y_true, Y_pred)
    r2 = gene_r2(Y_true, Y_pred)
    dge_acc, n_dge = dge_direction_accuracy(Y_true, Y_pred, dge_sd)
    return {
        "gene_r_mean": float(np.nanmean(r)) if np.isfinite(r).any() else np.nan,
        "gene_r_median": float(np.nanmedian(r)) if np.isfinite(r).any() else np.nan,
        "gene_r2_mean": float(np.nanmean(r2)) if np.isfinite(r2).any() else np.nan,
        "gene_r2_median": float(np.nanmedian(r2)) if np.isfinite(r2).any() else np.nan,
        "frac_genes_r_above_0.3": float(np.nanmean(r > 0.3)) if np.isfinite(r).any() else np.nan,
        "dge_direction_acc": dge_acc,
        "n_dge_entries": int(n_dge),
        "neighbor_overlap_k10": neighbor_overlap(Y_true, Y_pred, 10),
        "n_test_perturbations": int(Y_true.shape[0]),
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
@dataclass
class FoldResult:
    protocol: str
    model: str
    fold: int
    metrics: dict
    permuted: bool = False


def run_protocol(X: np.ndarray, Y: np.ndarray, splits, protocol: str,
                 models: list[str], cfg: dict) -> tuple[list[FoldResult], dict]:
    dge_sd = float(cfg["expr"]["dge_sd_multiplier"])
    results: list[FoldResult] = []
    oof = {m: np.full_like(Y, np.nan, dtype=float) for m in models}

    for fold, (tr, te) in enumerate(splits):
        for m in models:
            pred = _fit_predict(m, X[tr], Y[tr], X[te], cfg)
            oof[m][te] = pred
            met = summarize(Y[te], pred, dge_sd)
            results.append(FoldResult(protocol, m, fold, met))
            log.info("  [%s/%s] fold%d: r=%.3f r2=%.3f dge=%.3f",
                     protocol, m, fold, met["gene_r_mean"],
                     met["gene_r2_mean"],
                     met["dge_direction_acc"] if np.isfinite(met["dge_direction_acc"]) else -1)

    # ---- 置换负对照 ------------------------------------------------------
    rng = np.random.default_rng(int(cfg["seed"]))
    n_perm = int(cfg["model"]["permutation_repeats"])
    for rep in range(n_perm):
        perm = rng.permutation(Y.shape[0])
        Yp = Y[perm]
        for m in models:
            if m in ("mean",):
                continue
            preds, truths = [], []
            for tr, te in splits:
                p = _fit_predict(m, X[tr], Yp[tr], X[te], cfg)
                preds.append(p)
                truths.append(Yp[te])
            met = summarize(np.vstack(truths), np.vstack(preds), dge_sd)
            results.append(FoldResult(protocol, m, -1 - rep, met, permuted=True))

    return results, {"oof": oof}


def to_table(results: list[FoldResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        d = {"protocol": r.protocol, "model": r.model, "fold": r.fold,
             "is_permutation_control": r.permuted}
        d.update(r.metrics)
        rows.append(d)
    return pd.DataFrame(rows)
