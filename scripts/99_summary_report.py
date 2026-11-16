#!/usr/bin/env python3
"""结果摘要生成器 —— 把 results/ 下的所有产物汇成一份可直接引用的 Markdown。

用法
----
    .venv/bin/python scripts/99_summary_report.py
    # 输出: 结果摘要.md

设计意图
--------
报告和 PPT 需要的每个数字都在这里集中出现一次，避免手抄 CSV 抄错。
每次重跑流水线后重新执行本脚本即可自动更新。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
def _jload(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    if df is None or df.empty:
        return "_（无数据）_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if not np.isfinite(v) else floatfmt.format(v))
            elif isinstance(v, (int, np.integer)):
                cells.append(str(int(v)))
            else:
                cells.append("" if v is None else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _section(title: str, level: int = 2) -> str:
    return f"\n{'#' * level} {title}\n\n"


# ---------------------------------------------------------------------------
def build_report() -> str:
    out: list[str] = []
    build = _jload(RES / "metrics/build_summary.json")
    batch = _jload(RES / "metrics/batch_diagnosis.json")
    valid = _jload(RES / "metrics/validation.json")
    setup = _jload(RES / "metrics/predict_setup.json")

    out.append("# 结果摘要（自动生成）\n")
    out.append("> 由 `scripts/99_summary_report.py` 生成。"
               "所有数字直接读自 `results/`，重跑流水线后重新生成本文件即可。\n")

    # ---------------- 1. 数据 ----------------
    out.append(_section("1. 数据集与覆盖"))
    cov = RES / "tables/coverage.csv"
    if cov.exists():
        out.append(_md_table(pd.read_csv(cov)))
    if build:
        out.append(f"\n- 数据集：{', '.join(build.get('datasets', []))}\n")
        out.append(f"- 跨数据集共有形态特征：**{build.get('n_common_features')}** 维\n")
        out.append(f"- 共有 landmark 基因：**{build.get('n_common_genes')}** 个\n")
        out.append(f"- **最终建模扰动数：{build.get('n_perturbations')}**\n")

    # ---------------- 2. 质控 ----------------
    if build and build.get("qc"):
        out.append(_section("2. 智能体 1 质控结果"))
        rows = []
        for ds, r in build["qc"].items():
            rows.append({
                "dataset": ds,
                "原始孔数": r.get("n_replicates_raw"),
                "质控后孔数": r.get("n_replicates_final"),
                "原始特征数": r.get("n_features_raw"),
                "保留特征数": r.get("n_features_final"),
                "缺失值插补": r.get("n_missing_imputed", 0),
                "离群孔": (r.get("outlier_stats") or {}).get("n_outlier"),
                "扰动数": r.get("n_perturbations"),
                "对照孔": r.get("n_controls"),
                "批次数": r.get("batches"),
            })
        out.append(_md_table(pd.DataFrame(rows)))
        dropped = {ds: r.get("dropped_counts", {}) for ds, r in build["qc"].items()}
        out.append("\n**各步剔除数量**\n\n")
        out.append(_md_table(pd.DataFrame(dropped).T.reset_index()
                             .rename(columns={"index": "dataset"})))

    # ---------------- 3. 转录组 ----------------
    if build and build.get("expr"):
        out.append(_section("3. 智能体 3 转录组预处理"))
        rows = []
        for ds, r in build["expr"].items():
            rows.append({
                "dataset": ds,
                "L1000 孔数": r.get("n_wells_raw"),
                "基因数": r.get("n_genes"),
                "原始扰动": r.get("n_perturbations_raw"),
                "一致性过滤后": r.get("n_perturbations_final"),
                "噪声剔除": r.get("n_perturbations_dropped_noise"),
                "重复中位谱相关": r.get("median_replicate_spearman"),
                "对照孔": r.get("n_control_wells"),
            })
        out.append(_md_table(pd.DataFrame(rows)))

    # ---------------- 4. 扰动编码 ----------------
    if build and build.get("encoding"):
        out.append(_section("4. 智能体 4 扰动编码"))
        rows = []
        for ds, r in build["encoding"].items():
            rows.append({
                "dataset": ds,
                "扰动数": r.get("n_perturbations"),
                "化合物数": r.get("compound_total"),
                "有 SMILES": r.get("compound_with_smiles"),
                "化合物编码方式": r.get("compound_encoded_by"),
                "遗传扰动数": r.get("genetic_total"),
                "遗传编码方式": r.get("genetic_encoded_by"),
            })
        out.append(_md_table(pd.DataFrame(rows)))

    # ---------------- 5. 批次 ----------------
    if batch:
        out.append(_section("5. 智能体 6 批次效应诊断"))
        b, a = batch.get("before", {}), batch.get("after", {})
        rows = [
            {"指标": "批次分类准确率（**越低越好**）",
             "校正前": b.get("accuracy"), "校正后": a.get("accuracy")},
            {"指标": "随机基线（多数类比例）", "校正前": b.get("chance"),
             "校正后": a.get("chance")},
            {"指标": "扰动检索准确率 k=1（越高越好）",
             "校正前": b.get("pert_retrieval_k1"), "校正后": a.get("pert_retrieval_k1")},
            {"指标": "扰动检索随机基线",
             "校正前": b.get("pert_retrieval_chance"),
             "校正后": a.get("pert_retrieval_chance")},
            {"指标": "跨板扰动检索（越高越好）",
             "校正前": b.get("pert_retrieval_k1_cross_plate"),
             "校正后": a.get("pert_retrieval_k1_cross_plate")},
            {"指标": "批次数", "校正前": b.get("n_batches"), "校正后": a.get("n_batches")},
        ]
        out.append(_md_table(pd.DataFrame(rows), floatfmt="{:.4f}"))
        c = batch.get("combat", {})
        out.append(f"\n- 校正方法：位置-尺度校正（ComBat 核心）；"
                   f"经验贝叶斯收缩 = {c.get('use_eb')}；"
                   f"尺度下限 = {c.get('sd_floor_frac')}\n")
        out.append(f"- 批次分类准确率下降 **{batch.get('delta_accuracy'):.4f}**\n"
                   if isinstance(batch.get("delta_accuracy"), (int, float))
                   else "")

    # ---------------- 6. 预测 ----------------
    out.append(_section("6. 智能体 5 跨模态预测"))
    if setup:
        out.append(f"- 建模规模：{setup.get('n_perturbations')} 扰动 × "
                   f"{setup.get('n_features')} 形态特征 → {setup.get('n_genes')} 基因\n")
        out.append(f"- 评估协议：{', '.join(setup.get('protocols', []))}\n")
        out.append(f"- 模型：{', '.join(setup.get('models', []))}\n")
    ms = RES / "tables/metrics_summary.csv"
    if ms.exists():
        df = pd.read_csv(ms)
        cols = [c for c in ["protocol", "model", "gene_r_mean", "gene_r2_mean",
                            "dge_direction_acc", "neighbor_overlap_k10",
                            "frac_genes_r_above_0.3", "gene_r_mean_permuted"]
                if c in df.columns]
        out.append("\n**主指标表**\n\n")
        out.append(_md_table(df[cols].sort_values(["protocol", "model"])))
        out.append("\n> `*_permuted` 列是**置换负对照**（打乱标签重跑）。"
                   "真实模型的指标必须明显高于它才有意义。\n")

    per_gene = RES / "tables/per_gene_metrics.csv"
    if per_gene.exists():
        pg = pd.read_csv(per_gene)
        out.append("\n**逐基因指标（主协议最佳模型）**\n\n")
        out.append(f"- 基因数 {len(pg)}；"
                   f"r>0.3 的基因占比 {100*np.mean(pg.gene_r > 0.3):.1f}%；"
                   f"r>0.5 占比 {100*np.mean(pg.gene_r > 0.5):.1f}%\n")
        if "probe_id" in pg.columns:
            top = pg.nlargest(8, "gene_r")[["probe_id", "gene_r", "gene_r2"]]
            out.append("\n预测最好的 8 个基因：\n\n" + _md_table(top))

    bpt = RES / "tables/metrics_by_pert_type.csv"
    if bpt.exists():
        out.append("\n**按扰动类型分层**\n\n")
        out.append(_md_table(pd.read_csv(bpt)))

    # ---------------- 7. 生物验证 ----------------
    if valid and valid.get("pathway_available"):
        out.append(_section("7. 智能体 7 生物验证"))
        rows = [
            {"指标": "匹配到的 Hallmark 通路数", "值": valid.get("n_pathways")},
            {"指标": "通路分数相关（均值版，Pearson）", "值": valid.get("pathway_pearson")},
            {"指标": "通路分数相关（均值版，Spearman）", "值": valid.get("pathway_spearman")},
            {"指标": "通路分数相关（逐扰动均值）",
             "值": valid.get("pathway_pearson_per_perturbation")},
            {"指标": "通路分数相关（逐扰動中位数）",
             "值": valid.get("pathway_pearson_per_perturbation_median")},
            {"指标": "通路相关为正的扰动占比", "值": valid.get("frac_perturbations_positive")},
            {"指标": "top 通路 Jaccard 重合度", "值": valid.get("top_pathway_jaccard")},
        ]
        out.append(_md_table(pd.DataFrame(rows), floatfmt="{:.4f}"))
        out.append("\n> 均值版相关系数容易被少数大幅值扰动主导，"
                   "**逐扰动版本**（中位数）更能代表「单个扰动预测得对不对」，"
                   "报告中建议两个都报。\n")

    vpt = RES / "tables/validation_by_pert_type.csv"
    if vpt.exists():
        out.append("\n**因果分层（按扰动类型）**\n\n")
        out.append(_md_table(pd.read_csv(vpt)))

    pe = RES / "tables/pathway_enrichment.csv"
    if pe.exists():
        df = pd.read_csv(pe).head(10)
        out.append("\n**响应最强的 10 条通路**\n\n")
        out.append(_md_table(df))

    # ---------------- 8. 文件清单 ----------------
    out.append(_section("8. 产物清单"))
    rows = []
    for sub in ("figures", "tables", "metrics"):
        d = RES / sub
        if not d.exists():
            continue
        for f in sorted(d.glob("*")):
            rows.append({"目录": sub, "文件": f.name,
                         "大小(KB)": round(f.stat().st_size / 1024, 1)})
    out.append(_md_table(pd.DataFrame(rows)))

    out.append("\n---\n\n"
               "**引用建议**：报告正文所有数字直接复制上表；"
               "PPT 用 `results/figures/` 里的图。\n")
    return "".join(out)


def main() -> int:
    text = build_report()
    dest = ROOT / "结果摘要.md"
    dest.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n[已写入] {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
