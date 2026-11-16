"""出图模块 —— 任务书要求的 4 张必放图 + 若干加分图。

fig1  预测 vs 真实 散点（并排 shuffle 负对照）
fig2  逐基因 R² / 相关系数 分布直方图（多模型 + 负对照）
fig3  UMAP 按批次着色（ComBat 校正前 / 后）
fig4  未见扰动 vs 已见扰动 性能柱状图（多协议）
fig5  通路一致性散点
fig6  扰动覆盖表
fig7  批次诊断前后对照（批次可分性 ↓、扰动可分性持平）
fig8  泄漏诊断：输入里加入扰动身份后的"虚假提升"
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import PROJECT_ROOT, get_logger

log = get_logger("figures")

# 中文字体候选（按优先级）。文件不存在时自动跳过。
_CJK_FONT_CANDIDATES = [
    PROJECT_ROOT / "data/reference/fonts/NotoSansCJKsc-Regular.otf",
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
]

_CJK_FONT_URL = (
    "https://raw.githubusercontent.com/notofonts/noto-cjk/main/"
    "Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf"
)


def _ensure_cjk_font(timeout: int = 180) -> None:
    """首次运行时把 Noto Sans CJK SC 下载到 `data/reference/fonts/`。

    字体是 16 MB 的二进制，不纳入版本库；但缺了它图上所有汉字都会变成方块，
    因此在这里自动补一次。下载失败不影响英文图，只打一条警告。
    """
    if any(p.exists() for p in _CJK_FONT_CANDIDATES):
        return
    import urllib.request

    target = _CJK_FONT_CANDIDATES[0]
    tmp = target.with_suffix(".part")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(_CJK_FONT_URL, headers={"Connection": "close"})
        with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.replace(target)
        log.info("已下载中文字体: %s (SIL OFL)", target.name)
    except Exception as exc:  # noqa: BLE001
        log.warning("中文字体下载失败(%s)，图中中文可能显示为方块；"
                    "可手动放到 data/reference/fonts/", type(exc).__name__)
        if tmp.exists():
            tmp.unlink()


def setup_matplotlib(cfg: dict):
    """初始化 matplotlib 并**确保中文字体可用**。

    中文字体缺失会让图上所有汉字变成方块（tofu），是交付物最常见的坑。
    这里按优先级寻找系统中可用的 CJK 字体并注册，找不到时退回 DejaVu Sans
    （英文图仍可正常使用）。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib import font_manager as fm

    _ensure_cjk_font()

    family = None
    for path in _CJK_FONT_CANDIDATES:
        try:
            if path.exists():
                fm.fontManager.addfont(str(path))
                cand = fm.FontProperties(fname=str(path)).get_name()
                # Droid Sans Fallback 只有 CJK 字形、没有拉丁字母，不能单独使用
                if "Droid" in cand:
                    continue
                family = cand
                break
        except Exception:  # noqa: BLE001
            continue

    sns.set_theme(style=cfg["plot"]["style"], context="talk")
    if family:
        plt.rcParams["font.family"] = family
        log.info("绘图字体: %s", family)
    else:
        plt.rcParams["font.family"] = cfg["plot"].get("font", "DejaVu Sans")
        log.warning("未找到中文字体，图中中文可能显示为方块；"
                    "可把字体放到 data/reference/fonts/")
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["savefig.dpi"] = int(cfg["plot"]["dpi"])
    plt.rcParams["figure.dpi"] = 110
    return plt, sns


def _setup(cfg: dict):
    return setup_matplotlib(cfg)


def _save(fig, path: Path, cfg: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(cfg["plot"]["dpi"]), bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    log.info("  已保存 %s", path.name)
    return path


# ---------------------------------------------------------------------------
def fig1_pred_vs_true(Y_true: np.ndarray, Y_pred: np.ndarray,
                      Y_perm: np.ndarray, cfg: dict, out: Path) -> Path:
    plt, _ = _setup(cfg)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharex=True, sharey=True)

    for ax, Yp, title in zip(
            axes, [Y_pred, Y_perm],
            ["跨模态预测", "负对照（打乱标签）"]):
        x, y = Y_true.ravel(), Yp.ravel()
        ax.scatter(x, y, s=4, alpha=0.15, edgecolors="none", rasterized=True)
        lim = np.nanpercentile(np.abs(np.concatenate([x, y])), 99.5)
        ax.plot([-lim, lim], [-lim, lim], "--", color="crimson", lw=1.4)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        r = np.corrcoef(x, y)[0, 1]
        ax.set_title(f"{title}\nr = {r:.3f}", fontsize=13, pad=8)
        ax.set_xlabel("真实 Δ 表达 (z)", fontsize=11)
    axes[0].set_ylabel("预测 Δ 表达 (z)", fontsize=11)
    fig.suptitle("图1  预测 vs 真实：模型与置换负对照", fontsize=15, y=1.03)
    fig.tight_layout()
    return _save(fig, out, cfg)


# ---------------------------------------------------------------------------
def fig2_gene_distributions(per_gene: dict[str, tuple[np.ndarray, np.ndarray]],
                            cfg: dict, out: Path) -> Path:
    """per_gene: {model_name: (r_array, r2_array)}"""
    plt, sns = _setup(cfg)
    names = list(per_gene)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    for ax, which in zip(axes, ["r", "r2"]):
        rows = []
        for name in names:
            arr = per_gene[name][0 if which == "r" else 1]
            arr = arr[np.isfinite(arr)]
            rows.append(pd.DataFrame({"模型": name, "值": arr}))
        df = pd.concat(rows)
        sns.histplot(data=df, x="值", hue="模型", bins=50, element="step",
                     fill=False, stat="density", common_norm=False, ax=ax,
                     alpha=0.85)
        ax.axvline(0, color="k", ls=":", lw=1)
        ax.set_ylabel("密度", fontsize=11)
        ax.set_xlabel("逐基因 Pearson r" if which == "r" else "逐基因 R²",
                      fontsize=11)
        ax.set_title("逐基因相关系数分布" if which == "r" else "逐基因 R² 分布",
                     fontsize=12)
    fig.suptitle("图2  逐基因性能分布：真实信号 vs 负对照", fontsize=14)
    fig.tight_layout()
    return _save(fig, out, cfg)


# ---------------------------------------------------------------------------
def fig3_umap_batch(emb_before: np.ndarray, emb_after: np.ndarray,
                    batch: np.ndarray, cfg: dict, out: Path,
                    acc_before: float | None = None,
                    acc_after: float | None = None) -> Path:
    plt, _ = _setup(cfg)
    levels = pd.Categorical(batch).categories
    cmap = plt.get_cmap("tab20")
    color_of = {b: cmap(i % 20) for i, b in enumerate(levels)}
    colors = [color_of[b] for b in batch]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, emb, tag, acc in zip(
            axes, [emb_before, emb_after], ["校正前", "ComBat 校正后"],
            [acc_before, acc_after]):
        ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=9, alpha=0.7,
                   edgecolors="none", rasterized=True)
        sub = f"批次分类准确率 = {acc:.3f}" if acc is not None and np.isfinite(acc) else ""
        ax.set_title(f"{tag}\n{sub}", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP-1", fontsize=10)
        ax.set_ylabel("UMAP-2", fontsize=10)
    fig.suptitle(f"图3  形态空间 UMAP 按批次着色（共 {len(levels)} 个批次）",
                 fontsize=14)
    fig.tight_layout()
    return _save(fig, out, cfg)


# ---------------------------------------------------------------------------
PROTOCOL_LABELS = {
    "perturbation_kfold": "新扰动\n(同分布)",
    "cluster_kfold": "新表型簇\n(更难)",
    "cross_dataset": "跨细胞系\n(最难)",
}

MODEL_LABELS = {"ridge": "Ridge", "lasso": "Lasso", "mlp": "MLP"}


def _proto_label(p: str) -> str:
    if p in PROTOCOL_LABELS:
        return PROTOCOL_LABELS[p]
    if p.startswith("cross_dataset"):
        return PROTOCOL_LABELS["cross_dataset"]
    return p


def fig4_seen_vs_unseen(metrics: pd.DataFrame, cfg: dict, out: Path) -> Path:
    plt, sns = _setup(cfg)
    df = metrics[(~metrics.is_permutation_control) &
                 (metrics.model != "mean")].copy()
    df["protocol"] = df["protocol"].map(_proto_label)
    agg = df.groupby(["protocol", "model"], as_index=False).agg(
        gene_r_mean=("gene_r_mean", "mean"),
        gene_r2_mean=("gene_r2_mean", "mean"),
        dge_direction_acc=("dge_direction_acc", "mean"),
    )
    metric_cols = ["gene_r_mean", "gene_r2_mean", "dge_direction_acc"]
    label_map = {"gene_r_mean": "逐基因相关 r", "gene_r2_mean": "逐基因 R²",
                 "dge_direction_acc": "差异基因方向准确率"}
    order = [m for m in ["ridge", "lasso", "mlp"] if m in set(agg.model)]
    proto_order = [_proto_label(p) for p in
                   ["perturbation_kfold", "cluster_kfold", "cross_dataset"]]
    proto_order = [p for p in proto_order if p in set(agg.protocol)]
    proto_order += [p for p in agg.protocol.unique() if p not in proto_order]

    long = agg.melt(id_vars=["protocol", "model"], value_vars=metric_cols,
                    var_name="指标", value_name="数值")
    long["指标"] = long["指标"].map(label_map)

    fig, axes = plt.subplots(1, len(metric_cols), figsize=(16, 5.2))
    for k, (ax, m) in enumerate(zip(axes, metric_cols)):
        sub = long[long["指标"] == label_map[m]]
        sns.barplot(data=sub, x="protocol", y="数值", hue="model",
                    hue_order=order, order=proto_order, errorbar=None,
                    ax=ax, palette="Set2")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"({chr(97+k)}) {label_map[m]}", fontsize=13, pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=10)
        if ax.get_legend() is not None:
            ax.get_legend().set_title("模型")
        ax.margins(x=0.08)
    fig.suptitle("图4  未见扰动泛化：三种 OOD 协议 × 三个指标（越高越好）",
                 fontsize=15, y=1.02)
    fig.tight_layout()
    return _save(fig, out, cfg)


# ---------------------------------------------------------------------------
def fig5_pathway(true_scores: pd.DataFrame, pred_scores: pd.DataFrame,
                 cfg: dict, out: Path) -> Path:
    plt, _ = _setup(cfg)
    if true_scores.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "未能获取通路基因集\n（MSigDB GMT 下载失败）",
                ha="center", va="center", fontsize=13)
        ax.axis("off")
        return _save(fig, out, cfg)

    from .common import safe_pearson

    t = true_scores.mean(axis=0)
    p = pred_scores.mean(axis=0)
    r_agg = safe_pearson(t.values, p.values)

    per_pert = np.asarray(
        [safe_pearson(true_scores.loc[i].values, pred_scores.loc[i].values)
         for i in true_scores.index], dtype=float)
    per_pert = per_pert[np.isfinite(per_pert)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.3))

    # ---- 左：通路层面的均值散点（sanity check）--------------------------
    ax = axes[0]
    ax.scatter(t.values, p.values, s=34, alpha=0.85, color="#2E7D32",
               edgecolors="k", linewidths=0.4)
    lim = np.nanpercentile(np.abs(np.concatenate([t.values, p.values])), 99)
    ax.plot([-lim, lim], [-lim, lim], "--", color="crimson", lw=1.3)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_title(f"（a）通路均值层面  r = {r_agg:.3f}\n"
                 f"{true_scores.shape[1]} 条 Hallmark 通路", fontsize=12)
    ax.set_xlabel("真实通路分数（均值）", fontsize=11)
    ax.set_ylabel("预测通路分数（均值）", fontsize=11)
    ax.text(0.03, 0.95, "受全局均值主导，仅作 sanity check",
            transform=ax.transAxes, fontsize=9, va="top", color="#666")

    # ---- 右：逐扰动通路一致性分布（主报告指标）--------------------------
    ax = axes[1]
    if per_pert.size:
        ax.hist(per_pert, bins=50, color="#4C72B0", alpha=0.85,
                edgecolor="white", linewidth=0.4)
        med = float(np.median(per_pert))
        ax.axvline(0, color="k", ls=":", lw=1.2)
        ax.axvline(med, color="crimson", ls="--", lw=1.8)
        ax.text(med, ax.get_ylim()[1] * 0.92, f"  中位数 {med:.3f}",
                color="crimson", fontsize=11, va="top")
        frac_pos = float(np.mean(per_pert > 0))
        ax.set_title(f"（b）逐扰动通路一致性分布（n={per_pert.size}）\n"
                     f"{100*frac_pos:.1f}% 的扰动为正相关", fontsize=12)
    else:
        ax.text(0.5, 0.5, "无有效扰动", ha="center", va="center")
    ax.set_xlabel("单个扰动的通路分数相关系数 r", fontsize=11)
    ax.set_ylabel("扰动数", fontsize=11)

    fig.suptitle("图5  通路一致性验证", fontsize=14)
    fig.tight_layout()
    return _save(fig, out, cfg)


# ---------------------------------------------------------------------------
def fig6_coverage(coverage: pd.DataFrame, cfg: dict, out: Path) -> Path:
    plt, sns = _setup(cfg)

    SHORT = {
        "LINCS-Pilot1": "LINCS-Pilot1\n(A549 · 化合物)",
        "TA-ORF-BBBC037-Rohban": "TA-ORF\n(U2OS · 遗传/ORF)",
        "LUAD-BBBC041-Caicedo": "LUAD\n(A549 · 遗传/shRNA)",
        "CDRP-BBBC047-Bray": "CDRP\n(U2OS · 化合物)",
    }
    df = coverage.copy()
    df["短名"] = df["dataset"].map(lambda d: SHORT.get(d, d))
    order = list(df["短名"])

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))

    ax = axes[0]
    bars = ax.bar(df["短名"], df["n_perturbations_matched"],
                  color="#4C72B0", width=0.55)
    for bar, v in zip(bars, df["n_perturbations_matched"]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(df["n_perturbations_matched"]) * 0.02,
                str(int(v)), ha="center", fontsize=12)
    ax.set_title("（a）配对成功的扰动数", fontsize=13, pad=8)
    ax.set_ylabel("扰动数", fontsize=12)
    ax.set_ylim(0, max(df["n_perturbations_matched"]) * 1.15)
    ax.tick_params(axis="x", labelsize=10)

    ax = axes[1]
    long = df.melt(id_vars=["短名"],
                   value_vars=["n_perturbations_morph", "n_perturbations_expr",
                               "n_perturbations_matched"],
                   var_name="类型", value_name="数量")
    long["类型"] = long["类型"].map({
        "n_perturbations_morph": "仅形态可用",
        "n_perturbations_expr": "仅转录组可用",
        "n_perturbations_matched": "两模态配对",
    })
    sns.barplot(data=long, x="短名", y="数量", hue="类型", order=order,
                ax=ax, palette="muted")
    ax.set_title("（b）模态覆盖与配对率", fontsize=13, pad=8)
    ax.set_xlabel("")
    ax.set_ylabel("扰动数", fontsize=12)
    ax.tick_params(axis="x", labelsize=10)
    ax.legend(fontsize=10, title="", loc="upper right",
              bbox_to_anchor=(1.0, 0.98), framealpha=0.95)

    fig.suptitle(f"图6  数据覆盖表（两模态配对成功合计 {int(df['n_perturbations_matched'].sum())} 个扰动）",
                 fontsize=15, y=1.02)
    fig.tight_layout()
    return _save(fig, out, cfg)


# ---------------------------------------------------------------------------
def fig7_batch_diagnosis(diag: dict, cfg: dict, out: Path) -> Path:
    """批次诊断双面板：
    (a) 批次分类准确率（越低越好，与随机基线对比）
    (b) 扰动检索相对随机基线的倍数（越高越好，log 轴）
    两者量纲差 1~2 个数量级，放同一坐标轴会失去可读性，因此分面。
    """
    plt, _ = _setup(cfg)
    b, a = diag["before"], diag["after"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    # ---- (a) 批次可分性 -------------------------------------------------
    ax = axes[0]
    vals = [b.get("accuracy", np.nan), a.get("accuracy", np.nan)]
    bars = ax.bar(["校正前", "ComBat 校正后"], vals,
                  color=["#C44E52", "#55A868"], width=0.55)
    chance = b.get("chance", np.nan)
    if np.isfinite(chance):
        ax.axhline(chance, ls="--", color="k", lw=1.3,
                   label=f"随机基线 {chance:.3f}")
        ax.legend(fontsize=10, loc="upper right")
    for bar, v in zip(bars, vals):
        if np.isfinite(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}",
                    ha="center", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("批次分类准确率", fontsize=12)
    ax.set_title("（a）批次可分性：越低越好\n越接近随机基线 = 批次信息被抹干净",
                 fontsize=12)

    # ---- (b) 生物学保留度：相对随机基线的倍数 ----------------------------
    ax = axes[1]
    def _fold(d):
        r, c = d.get("pert_retrieval_k1"), d.get("pert_retrieval_chance")
        if r is None or c in (None, 0) or not np.isfinite(c):
            return np.nan
        return float(r) / float(c)

    def _fold_x(d):
        r = d.get("pert_retrieval_k1_cross_plate")
        c = d.get("pert_retrieval_cross_plate_chance")
        if r is None or c in (None, 0) or not np.isfinite(c):
            return np.nan
        return float(r) / float(c)

    labels = ["同板扰动检索", "跨板扰动检索"]
    before_vals = [_fold(b), _fold_x(b)]
    after_vals = [_fold(a), _fold_x(a)]
    x = np.arange(len(labels))
    w = 0.36
    ax.bar(x - w / 2, before_vals, w, label="校正前", color="#C44E52")
    ax.bar(x + w / 2, after_vals, w, label="ComBat 校正后", color="#55A868")
    for xi, v0, v1 in zip(x, before_vals, after_vals):
        for off, v in ((-w / 2, v0), (w / 2, v1)):
            if np.isfinite(v):
                ax.text(xi + off, v * 1.08, f"{v:.0f}×", ha="center",
                        fontsize=11)
    ax.set_yscale("log")
    ax.axhline(1.0, ls="--", color="k", lw=1.3, label="随机水平 (1×)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("相对随机基线的倍数（log）", fontsize=12)
    ax.set_title("（b）生物学保留度：越高越好\n远高于 1× = 扰动信号仍在", fontsize=12)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_ylim(0.8, max([v for v in before_vals + after_vals
                          if np.isfinite(v)] + [10]) * 1.6)

    fig.suptitle("图7  批次效应诊断：校正应只抹掉批次、不抹掉生物学",
                 fontsize=15, y=1.02)
    fig.tight_layout()
    return _save(fig, out, cfg)


# ---------------------------------------------------------------------------
def fig8_leakage(leak: dict, cfg: dict, out: Path) -> Path:
    plt, _ = _setup(cfg)
    labels = ["形态 \n(正确做法)", "+ 扰动身份 \n(泄漏)", "+ 扰动身份 \n(负对照)"]
    vals = [leak.get("morph_only", np.nan), leak.get("morph_plus_identity", np.nan),
            leak.get("permuted_identity", np.nan)]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, vals, color=["#4C72B0", "#C44E52", "#999999"])
    for bar, v in zip(bars, vals):
        if np.isfinite(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.3f}",
                    ha="center", fontsize=11)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("基因相关 r（未见扰动）", fontsize=11)
    ax.set_title("图8  泄漏诊断：把扰动身份喂进模型会制造虚假性能", fontsize=13)
    fig.tight_layout()
    return _save(fig, out, cfg)


# ---------------------------------------------------------------------------
def fig9_metrics_summary(metrics: pd.DataFrame, cfg: dict, out: Path) -> Path:
    plt, sns = _setup(cfg)
    df = metrics.copy()
    df["protocol"] = df["protocol"].map(_proto_label)
    df["类型"] = np.where(df.is_permutation_control, "负对照", "真实模型")
    piv = df.groupby(["protocol", "model", "类型"], as_index=False)[
        "gene_r_mean"].mean()
    piv["键"] = piv["model"].map(lambda m: MODEL_LABELS.get(m, m)) + " | " + piv["类型"]
    keys = [k for k in piv["键"].unique()]
    keys.sort(key=lambda s: ("负对照" in s, s))

    fig, ax = plt.subplots(figsize=(12, 5.2))
    sns.barplot(data=piv, x="protocol", y="gene_r_mean", hue="键",
                hue_order=keys, ax=ax, palette="husl")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("图9  全部协议 × 模型 × 置换负对照 汇总", fontsize=14)
    ax.set_ylabel("逐基因相关 r", fontsize=12)
    ax.set_xlabel("")
    ax.legend(fontsize=9, ncol=2, title="模型 | 类型")
    fig.tight_layout()
    return _save(fig, out, cfg)
