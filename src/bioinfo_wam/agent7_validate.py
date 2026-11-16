"""智能体 7 —— 生物学验证（Biological Validation Agent）。

回答一个问题：**预测出来的表达向量，在生物学上是不是讲得通？**

三条验证线索
------------
1. **通路一致性**：把 978 个 landmark 基因按 MSigDB Hallmark 通路聚合成功效分数，
   比较"预测的通路分数"与"真实的通路分数"是否相关。
   相关系数高 ⇒ 模型抓住的是通路级别的生物学，不只是逐基因的数值噪声。
2. **富集方向一致性**：对每个扰动，把基因按 Δ 排序，
   检查通路基因是否富集在预期的上调/下调端（GSEA 的简化 running-sum）。
3. **因果分层**：把**化合物扰动**与**遗传扰动（ORF/CRISPR）**分开报告性能。
   遗传扰动是单点、可归因的干预，是"因果锚点"；
   化合物多靶点、脱靶效应多，形态-基因一致性天然更差。
   分层报告本身就是"相关 vs 因果"讨论的实证支撑。

依赖的外部参考表（自动下载并缓存，失败则优雅降级）
--------------------------------------------------
* `data/reference/lincs_gene_info.csv`   探针 ID → 基因符号
* `data/reference/h.all.symbols.gmt`     MSigDB Hallmark 通路
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .common import PROJECT_ROOT, get_logger, path_of, safe_pearson, safe_spearman

log = get_logger("agent7.validate")

GENE_INFO_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/"
    "GSE92742_Broad_LINCS_gene_info.txt.gz"
)
MYGENE_URL = "https://mygene.info/v3/query"


# ---------------------------------------------------------------------------
# 参考表
# ---------------------------------------------------------------------------
def _download(url: str, dest: Path, timeout: int = 180) -> bool:
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"Connection": "close"})
        with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.replace(dest)
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("下载失败 %s: %s", url.split("/")[-1], type(exc).__name__)
        if tmp.exists():
            tmp.unlink()
        return False


def _mygene_batch(probes: list[str], timeout: int = 90) -> dict[str, str]:
    """用 mygene.info 把 Affymetrix 探针 ID 批量注释为基因符号。"""
    import json as _json
    import urllib.parse

    out: dict[str, str] = {}
    for i in range(0, len(probes), 250):
        chunk = probes[i:i + 250]
        data = urllib.parse.urlencode({
            "q": ",".join(chunk),
            "scopes": "all",
            "fields": "symbol",
            "species": "human",
        }).encode()
        try:
            req = urllib.request.Request(MYGENE_URL, data=data,
                                         headers={"Connection": "close"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                hits = _json.loads(r.read().decode())
        except Exception as exc:  # noqa: BLE001
            log.warning("mygene 批次 %d 失败: %s", i // 250, type(exc).__name__)
            continue
        if isinstance(hits, dict):
            hits = [hits]
        for h in hits:
            if not isinstance(h, dict) or h.get("notfound"):
                continue
            sym = h.get("symbol")
            q = h.get("query")
            if sym and q:
                out[str(q)] = str(sym).upper()
    return out


def ensure_probe_mapping(cfg: dict, probes: list[str]) -> dict[str, str] | None:
    """探针 ID → 基因符号。优先读本地缓存；不足时调 mygene.info。"""
    dest = PROJECT_ROOT / "data/reference/probe2symbol.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    if dest.exists():
        try:
            tab = pd.read_csv(dest, dtype={"probe_id": str})
            mapping = dict(zip(tab["probe_id"].astype(str),
                               tab["gene_symbol"].astype(str)))
        except Exception:  # noqa: BLE001
            mapping = {}

    probes = [str(p) for p in probes]
    missing = [p for p in probes if p not in mapping]
    coverage = 1 - len(missing) / max(len(probes), 1)

    if missing and coverage < 1.0:
        log.info("探针映射缓存覆盖 %.1f%%，向 mygene.info 查询 %d 个探针",
                 100 * coverage, len(missing))
        fresh = _mygene_batch(missing)
        if fresh:
            mapping.update(fresh)
            pd.DataFrame({"probe_id": list(mapping),
                          "gene_symbol": list(mapping.values())}
                         ).to_csv(dest, index=False)
            log.info("探针映射已缓存: %d 条 → %s", len(mapping), dest.name)

    if not mapping:
        log.warning("无法获取探针→基因符号映射，通路分析将被跳过")
        return None
    got = sum(1 for p in probes if p in mapping)
    log.info("探针映射覆盖 %d/%d (%.1f%%)", got, len(probes), 100 * got / len(probes))
    return mapping


def ensure_hallmark_gmt(cfg: dict) -> dict[str, list[str]] | None:
    dest = path_of(cfg, cfg["validation"]["gmt_path"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        if not _download(cfg["validation"]["gmt_url"], dest):
            return None
    try:
        return read_gmt(dest)
    except Exception as exc:  # noqa: BLE001
        log.warning("读取 GMT 失败: %s", exc)
        return None


def read_gmt(path: str | Path) -> dict[str, list[str]]:
    sets: dict[str, list[str]] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = [p.strip() for p in line.rstrip("\n").split("\t") if p.strip()]
            if len(parts) < 3:
                continue
            name, _desc, genes = parts[0], parts[1], parts[2:]
            sets[name] = [g.upper() for g in genes]
    return sets


# ---------------------------------------------------------------------------
# 通路打分
# ---------------------------------------------------------------------------
def pathway_scores(Y: pd.DataFrame, gene_sets: dict[str, list[str]],
                   probe2sym: dict[str, str], cfg: dict) -> pd.DataFrame:
    """每个扰动 × 每条通路的平均 z 分数。"""
    v_cfg = cfg["validation"]
    lo, hi = int(v_cfg["min_genes_per_set"]), int(v_cfg["max_genes_per_set"])

    # 探针列 → 符号
    sym_by_col = {c: probe2sym.get(c, "") for c in Y.columns}
    inv: dict[str, list[str]] = {}
    for col, sym in sym_by_col.items():
        if sym:
            inv.setdefault(sym.upper(), []).append(col)

    rows = {}
    for name, genes in gene_sets.items():
        cols: list[str] = []
        for g in genes:
            cols.extend(inv.get(g, []))
        if not (lo <= len(cols) <= hi):
            continue
        rows[name] = Y[cols].mean(axis=1)
    if not rows:
        return pd.DataFrame(index=Y.index)
    return pd.DataFrame(rows, index=Y.index)


def pathway_consistency(true_scores: pd.DataFrame,
                        pred_scores: pd.DataFrame) -> dict:
    if true_scores.shape[1] < 3:
        return {"n_pathways": int(true_scores.shape[1]),
                "pathway_pearson": np.nan, "pathway_spearman": np.nan,
                "pathway_pearson_per_perturbation": np.nan,
                "top_pathway_jaccard": np.nan}
    t = true_scores.mean(axis=0)
    p = pred_scores.mean(axis=0)
    r = safe_pearson(t.values, p.values)
    rho = safe_spearman(t.values, p.values)

    # 更严格的指标：逐扰动地在 49 维通路空间求相关，再取均值/中位数。
    # 均值版本的相关系数容易被少数大幅值扰动主导（实测可到 0.99），
    # 逐扰动版本更接近"单个扰动预测得对不对"这一实际问题，报告应以前者为准。
    per_pert_vec = [safe_pearson(true_scores.loc[i].values, pred_scores.loc[i].values)
                    for i in true_scores.index]
    per_pert = np.asarray([x for x in per_pert_vec if np.isfinite(x)], dtype=float)

    k = max(3, true_scores.shape[1] // 3)
    top_t = set(t.abs().sort_values(ascending=False).head(k).index)
    top_p = set(p.abs().sort_values(ascending=False).head(k).index)
    jac = len(top_t & top_p) / len(top_t | top_p) if (top_t | top_p) else np.nan
    return {"n_pathways": int(true_scores.shape[1]),
            "pathway_pearson": float(r), "pathway_spearman": float(rho),
            "pathway_pearson_per_perturbation": (
                float(np.nanmean(per_pert)) if per_pert.size else np.nan),
            "pathway_pearson_per_perturbation_median": (
                float(np.nanmedian(per_pert)) if per_pert.size else np.nan),
            "frac_perturbations_positive": (
                float(np.mean(per_pert > 0)) if per_pert.size else np.nan),
            "top_pathway_jaccard": float(jac)}


def gsea_like_enrichment(true_scores: pd.DataFrame, pred_scores: pd.DataFrame,
                         Y_true: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """在通路分数层面比较：符号一致性 + 通路响应幅度。"""
    if true_scores.empty or pred_scores.empty:
        return pd.DataFrame(columns=["pathway", "sign_agreement",
                                     "true_abs_mean", "pred_abs_mean"])
    agree = np.sign(true_scores) == np.sign(pred_scores)
    out = pd.DataFrame({
        "pathway": true_scores.columns,
        "sign_agreement": agree.mean(axis=0).values,
        "true_abs_mean": true_scores.abs().mean(axis=0).values,
        "pred_abs_mean": pred_scores.abs().mean(axis=0).values,
    })
    return out.sort_values("true_abs_mean", ascending=False).head(top_n)


def stratified_by_type(metrics_by_pert: pd.DataFrame,
                       pert_meta: pd.DataFrame) -> pd.DataFrame:
    """按扰动类型（compound / orf / crispr）分层报告性能。"""
    df = metrics_by_pert.merge(pert_meta[["perturbation_id", "pert_type"]],
                               on="perturbation_id", how="left")
    agg = df.groupby("pert_type").agg(
        n=("perturbation_id", "size"),
        gene_r_mean=("gene_r", "mean"),
        gene_r_median=("gene_r", "median"),
        r2_mean=("r2", "mean"),
    ).reset_index()
    return agg


def per_perturbation_metrics(Y_true: np.ndarray, Y_pred: np.ndarray,
                             ids: list[str]) -> pd.DataFrame:
    rows = []
    for i, pid in enumerate(ids):
        t, p = Y_true[i], Y_pred[i]
        r = safe_pearson(t, p)
        ss_res = float(((t - p) ** 2).sum())
        ss_tot = float(((t - t.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rows.append({"perturbation_id": pid, "pearson_r": r, "r2": r2,
                     "gene_r": r, "r2_raw": r2,
                     "true_amplitude": float(np.abs(t).mean()),
                     "pred_amplitude": float(np.abs(p).mean())})
    return pd.DataFrame(rows)
