#!/usr/bin/env python3
"""主流水线编排 —— 七个智能体按依赖顺序串联。

用法
----
python scripts/run_pipeline.py --stage build       # 智能体 1-4：造表
python scripts/run_pipeline.py --stage batch       # 智能体 6：批次诊断与校正
python scripts/run_pipeline.py --stage predict     # 智能体 5：跨模态预测
python scripts/run_pipeline.py --stage validate    # 智能体 7：生物验证
python scripts/run_pipeline.py --stage figures     # 出图
python scripts/run_pipeline.py --stage all         # 一条龙

中间产物
--------
data/processed/
    X_morph.parquet                扰动级形态特征
    Y_expr.parquet                 扰动级 Δ 表达
    perturbation_meta.parquet      扰动元数据
    perturbation_encoding.parquet  扰动编码
    X_morph_replicate.parquet      孔级形态（供批次分析）
    meta_replicate.parquet         孔级元数据
    X_morph_combat.parquet         批次校正后形态（孔级）
results/
    tables/coverage.csv
    metrics/metrics_by_fold.csv
    metrics/oof.npz
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from bioinfo_wam import agent1_qc, agent2_morph, agent3_expr, agent4_pertencode  # noqa: E402
from bioinfo_wam import agent5_predict, agent6_batch, agent7_validate, figures  # noqa: E402
from bioinfo_wam.common import PROJECT_ROOT, ensure_dirs, get_logger, load_config  # noqa: E402
from bioinfo_wam.common import available_datasets, load_pair, save_json, set_seed  # noqa: E402

PROC = PROJECT_ROOT / "data/processed"
RES = PROJECT_ROOT / "results"
LOG = PROJECT_ROOT / "logs"

log = get_logger("pipeline", LOG / "pipeline.log")


# ===========================================================================
# Stage 1 —— 智能体 1/2/3/4
# ===========================================================================
def stage_build(cfg: dict) -> dict:
    ensure_dirs(cfg)
    set_seed(int(cfg["seed"]))
    datasets = available_datasets(cfg)
    if not datasets:
        raise SystemExit("没有可用的配对数据集，请先运行 scripts/00_download.py")

    log.info("=" * 78)
    log.info("阶段 1/5  造表（智能体 1 质控 → 2 形态 → 3 转录组 → 4 扰动编码）")
    log.info("可用子集: %s", datasets)
    log.info("=" * 78)

    per_ds: dict[str, dict] = {}

    smiles_table = None
    try:
        smiles_table = agent4_pertencode.ensure_smiles_table(cfg)
    except Exception as exc:  # noqa: BLE001
        log.warning("化合物结构表不可用: %s", exc)

    for ds in datasets:
        try:
            pair = load_pair(cfg, ds)
        except Exception as exc:  # noqa: BLE001
            log.warning("[" + ds + "] 读取失败（可能尚未下载完成）: %s", exc)
            continue
        if pair is None:
            continue
        log.info("-" * 78)
        log.info("[%s] Cell Painting %s | L1000 %s", ds, pair.cp.shape, pair.l1k.shape)

        # 遗传扰动的连接键：两侧都提供基因符号时用基因，否则回退 pert_id。
        def _has_gene(df: pd.DataFrame) -> bool:
            return any(c in df.columns for c in
                       ("Metadata_genesymbol", "Metadata_genesymbol_mutation"))

        genetic_key = "gene" if (_has_gene(pair.cp) and _has_gene(pair.l1k)) else "pert_id"
        log.info("[%s] 遗传扰动连接键: %s", ds, genetic_key)

        # ---- 智能体 1 ---------------------------------------------------
        qc = agent1_qc.run(pair.cp, ds, cfg, genetic_key=genetic_key)

        # ---- 智能体 2 ---------------------------------------------------
        X_pert, meta_morph = agent2_morph.build(qc.features, qc.meta, cfg)

        # ---- 智能体 3 ---------------------------------------------------
        Y_pert, Y_rep, expr_out = agent3_expr.build(pair.l1k, ds, cfg,
                                                    genetic_key=genetic_key)

        # 化合物结构：Rosetta 的 L1000 表内自带 Metadata_SMILES，直接抽出来
        smiles_map: dict[str, str] = {}
        l1k_meta = expr_out["meta_replicate"]
        if "smiles" in l1k_meta.columns:
            bad = {"", "nan", "None", "-666", "NA", "N/A"}
            for pid, smi in zip(l1k_meta["pert_id_raw"].astype(str),
                                l1k_meta["smiles"].astype(str)):
                if smi not in bad and pid not in bad:
                    smiles_map.setdefault(pid, smi)

        # 孔级形态特征（供智能体 6）
        X_rep = qc.features.copy()
        X_rep.index = qc.meta["replicate_id"].values

        per_ds[ds] = {
            "qc": qc,
            "X_pert": X_pert,
            "Y_pert": Y_pert,
            "X_rep": X_rep,
            "Y_rep": Y_rep,
            "meta_rep": expr_out["meta_replicate"],
            "meta_pert_expr": expr_out["meta_perturbation"],
            "expr_stats": expr_out["stats"],
            "morph_meta": meta_morph,
            # 必须按数据集存：编码在第二个循环才执行，用外层变量会只剩最后一个数据集的值
            "smiles_map": smiles_map,
        }

    # ---- 跨数据集对齐 ----------------------------------------------------
    feat_sets = [set(v["X_pert"].columns) for v in per_ds.values()]
    common_feats = sorted(set.intersection(*feat_sets)) if feat_sets else []
    gene_sets = [set(v["Y_pert"].columns) for v in per_ds.values()]
    common_genes = sorted(set.intersection(*gene_sets)) if gene_sets else []

    log.info("跨数据集共有形态特征: %d | 共有基因: %d", len(common_feats), len(common_genes))
    if not common_feats or not common_genes:
        raise SystemExit("跨数据集没有共同特征或基因，无法合并")

    X_parts, Y_parts, meta_parts, coverage_rows = [], [], [], []
    enc_parts = []

    for ds, v in per_ds.items():
        Xp = v["X_pert"][common_feats]
        Yp = v["Y_pert"][common_genes]
        ids = Xp.index.intersection(Yp.index)

        coverage_rows.append({
            "dataset": ds,
            "cell_line": v["qc"].report["cell_lines"][0] if v["qc"].report["cell_lines"] else "?",
            "n_replicates": v["qc"].report["n_replicates_final"],
            "n_features_morph": v["qc"].report["n_features_final"],
            "n_perturbations_morph": int(Xp.shape[0]),
            "n_perturbations_expr": int(Yp.shape[0]),
            "n_perturbations_matched": int(len(ids)),
            "match_rate": round(len(ids) / max(Xp.shape[0], 1), 4),
        })

        if len(ids) == 0:
            log.warning("[%s] 两模态没有交集，跳过", ds)
            continue

        X_parts.append(Xp.loc[ids])
        Y_parts.append(Yp.loc[ids])

        type_map = dict(zip(v["qc"].meta["perturbation_id"],
                            v["qc"].meta["pert_type"]))
        gene_map = dict(zip(v["qc"].meta["perturbation_id"],
                            v["qc"].meta.get("pert_gene",
                                              pd.Series("", index=v["qc"].meta.index))))
        mp = pd.DataFrame({"perturbation_id": ids})
        mp["dataset"] = ds
        mp["pert_type"] = [type_map.get(i, "compound") for i in ids]
        mp["pert_target"] = [i.split("::", 1)[1].replace("GENE::", "")
                             if "::" in i else i for i in ids]
        mp["pert_gene"] = [gene_map.get(i, "") or "" for i in ids]
        mp["cell_line"] = v["qc"].report["cell_lines"][0] if v["qc"].report["cell_lines"] else "?"
        mp["n_replicates_morph"] = v["morph_meta"].set_index(
            "perturbation_id").reindex(ids)["n_replicates"].values
        mp["replicate_spearman"] = v["meta_pert_expr"].set_index(
            "perturbation_id").reindex(ids)["replicate_spearman"].values
        meta_parts.append(mp)

        # ---- 智能体 4：扰动编码 -----------------------------------------
        enc, enc_diag = agent4_pertencode.build(mp, cfg,
                                                smiles_table=smiles_table,
                                                smiles_map=v.get("smiles_map") or {})
        enc_parts.append(enc)
        v["encoding_diag"] = enc_diag

    X_morph = pd.concat(X_parts).sort_index() if X_parts else pd.DataFrame()
    if not X_parts:
        raise SystemExit("所有数据集两模态均无交集；请检查 pert_id 是否一致")
    Y_expr = pd.concat(Y_parts).sort_index()
    meta_pert = pd.concat(meta_parts, ignore_index=True).set_index("perturbation_id")
    meta_pert = meta_pert.loc[X_morph.index]

    X_morph.index.name = "perturbation_id"
    Y_expr.index.name = "perturbation_id"

    # ---- 写盘 ------------------------------------------------------------
    PROC.mkdir(parents=True, exist_ok=True)
    X_morph.to_parquet(PROC / "X_morph.parquet")
    Y_expr.to_parquet(PROC / "Y_expr.parquet")
    meta_pert.reset_index().to_parquet(PROC / "perturbation_meta.parquet", index=False)

    if enc_parts:
        pd.concat(enc_parts).to_parquet(PROC / "perturbation_encoding.parquet")

    X_rep = pd.concat([v["X_rep"][common_feats] for v in per_ds.values()])
    meta_rep_parts = []
    for ds, v in per_ds.items():
        mr = v["qc"].meta.copy()
        mr.index = mr["replicate_id"].values
        meta_rep_parts.append(mr)
    meta_rep = pd.concat(meta_rep_parts)
    X_rep = X_rep.reindex(meta_rep.index)
    X_rep = X_rep.fillna(0.0)
    X_rep.to_parquet(PROC / "X_morph_replicate.parquet")
    meta_rep.reset_index(drop=True).to_parquet(PROC / "meta_replicate.parquet", index=False)

    coverage = pd.DataFrame(coverage_rows)
    (RES / "tables").mkdir(parents=True, exist_ok=True)
    coverage.to_csv(RES / "tables/coverage.csv", index=False, encoding="utf-8-sig")

    summary = {
        "datasets": datasets,
        "n_common_features": len(common_feats),
        "n_common_genes": len(common_genes),
        "n_perturbations": int(X_morph.shape[0]),
        "coverage": coverage_rows,
        "qc": {ds: v["qc"].report for ds, v in per_ds.items()},
        "expr": {ds: v["expr_stats"] for ds, v in per_ds.items()},
        "encoding": {ds: v.get("encoding_diag", {}) for ds, v in per_ds.items()},
    }
    save_json(summary, RES / "metrics/build_summary.json")

    log.info("X_morph: %s | Y_expr: %s", X_morph.shape, Y_expr.shape)
    log.info("阶段 1 完成")
    return {"X_morph": X_morph, "Y_expr": Y_expr, "meta_pert": meta_pert,
            "coverage": coverage, "summary": summary}


# ===========================================================================
# Stage 2 —— 智能体 6（批次）
# ===========================================================================
def stage_batch(cfg: dict) -> dict:
    log.info("=" * 78)
    log.info("阶段 2/5  批次效应诊断与校正（智能体 6）")
    log.info("=" * 78)

    X_rep = pd.read_parquet(PROC / "X_morph_replicate.parquet")
    meta_rep = pd.read_parquet(PROC / "meta_replicate.parquet")
    meta_rep.index = meta_rep["replicate_id"].values
    X_rep = X_rep.reindex(meta_rep.index)

    # 用子采样控制运行时间（批次分类器是 O(n) 线性模型，但仍要限规模）
    max_n = 20000
    if len(X_rep) > max_n:
        rng = np.random.default_rng(int(cfg["seed"]))
        sel = rng.choice(len(X_rep), size=max_n, replace=False)
        sel.sort()
        X_rep = X_rep.iloc[sel]
        meta_rep = meta_rep.iloc[sel]

    Xc, diag, labels = agent6_batch.run(X_rep, meta_rep, cfg)

    emb_before = agent6_batch.umap_embedding(X_rep.to_numpy(float), cfg)
    emb_after = agent6_batch.umap_embedding(Xc.to_numpy(float), cfg)

    Xc.to_parquet(PROC / "X_morph_combat.parquet")
    np.savez_compressed(RES / "metrics/umap_batch.npz",
                        before=emb_before, after=emb_after,
                        batch=labels["batch"].to_numpy(),
                        perturbation_id=labels["perturbation_id"].to_numpy())

    save_json(diag, RES / "metrics/batch_diagnosis.json")
    log.info("阶段 2 完成")
    return {"diag": diag, "emb_before": emb_before, "emb_after": emb_after,
            "labels": labels}


# ===========================================================================
# Stage 3 —— 智能体 5（预测）
# ===========================================================================
def _protocols(X: np.ndarray, Y: np.ndarray, meta: pd.DataFrame,
               cfg: dict) -> dict[str, list]:
    seed = int(cfg["seed"])
    n_splits = int(cfg["model"]["outer_folds"])
    out: dict[str, list] = {}

    out["perturbation_kfold"] = agent5_predict.split_perturbation_kfold(
        X.shape[0], n_splits, seed)
    out["cluster_kfold"] = agent5_predict.split_cluster_kfold(X, n_splits, seed)

    heldout = cfg["datasets"].get("heldout_for_transfer")
    if heldout:
        te = np.where(meta["dataset"].to_numpy() == heldout)[0]
        tr = np.where(meta["dataset"].to_numpy() != heldout)[0]
        if len(te) >= 10 and len(tr) >= 30:
            out[f"cross_dataset({heldout})"] = [(tr, te)]
        else:
            log.warning("跨数据集协议样本不足（train=%d, test=%d），跳过", len(tr), len(te))
    return out


def stage_predict(cfg: dict) -> dict:
    log.info("=" * 78)
    log.info("阶段 3/5  跨模态预测（智能体 5）")
    log.info("=" * 78)

    X_df = pd.read_parquet(PROC / "X_morph.parquet")
    Y_df = pd.read_parquet(PROC / "Y_expr.parquet")
    meta = pd.read_parquet(PROC / "perturbation_meta.parquet").set_index("perturbation_id")
    ids = X_df.index.intersection(Y_df.index)
    X_df, Y_df, meta = X_df.loc[ids], Y_df.loc[ids], meta.loc[ids]

    log.info("建模规模: %d 扰动 × %d 形态特征 → %d 基因",
             X_df.shape[0], X_df.shape[1], Y_df.shape[1])

    min_pert = int(cfg["model"]["min_perturbations"])
    if X_df.shape[0] < min_pert:
        raise SystemExit(f"扰动数 {X_df.shape[0]} 少于阈值 {min_pert}")

    X = X_df.to_numpy(np.float64)
    Y = Y_df.to_numpy(np.float64)

    models = [m for m in ("ridge", "lasso", "mlp")]
    protos = _protocols(X, Y, meta, cfg)

    all_results = []
    oof_store: dict[str, dict[str, np.ndarray]] = {}

    for pname, splits in protos.items():
        log.info("协议 %s（%d 折）", pname, len(splits))
        res, extra = agent5_predict.run_protocol(X, Y, splits, pname, models, cfg)
        all_results.extend(res)
        oof_store[pname] = extra["oof"]

    metrics = agent5_predict.to_table(all_results)
    (RES / "metrics").mkdir(parents=True, exist_ok=True)
    metrics.to_csv(RES / "metrics/metrics_by_fold.csv", index=False, encoding="utf-8-sig")

    # 汇总表
    summ = (metrics[~metrics.is_permutation_control]
            .groupby(["protocol", "model"], as_index=False)
            .agg({c: "mean" for c in [
                "gene_r_mean", "gene_r_median", "gene_r2_mean", "gene_r2_median",
                "frac_genes_r_above_0.3", "dge_direction_acc",
                "neighbor_overlap_k10", "n_test_perturbations"]}))
    perm = (metrics[metrics.is_permutation_control]
            .groupby(["protocol", "model"], as_index=False)
            .agg({c: "mean" for c in ["gene_r_mean", "gene_r2_mean",
                                      "dge_direction_acc"]}))
    perm = perm.rename(columns={c: f"{c}_permuted" for c in perm.columns
                                if c not in ("protocol", "model")})
    summ = summ.merge(perm, on=["protocol", "model"], how="left")
    summ.to_csv(RES / "tables/metrics_summary.csv", index=False, encoding="utf-8-sig")

    log.info("\n%s", summ.round(3).to_string(index=False))

    # 逐基因指标（主协议 + 主模型）
    main_proto = "perturbation_kfold"
    oof = oof_store.get(main_proto, {})
    per_gene_out = {}
    if oof:
        best_model = "lasso" if "lasso" in oof else list(oof)[0]
        P = oof[best_model]
        mask = ~np.isnan(P).any(axis=1)
        r = agent5_predict.gene_correlations(Y[mask], P[mask])
        r2 = agent5_predict.gene_r2(Y[mask], P[mask])
        pg = pd.DataFrame({"probe_id": Y_df.columns, "gene_r": r, "gene_r2": r2})
        pg = pg.sort_values("gene_r", ascending=False)
        pg.to_csv(RES / "tables/per_gene_metrics.csv", index=False, encoding="utf-8-sig")
        per_gene_out["main"] = pg

        np.savez_compressed(
            RES / "metrics/oof_predictions.npz",
            y_true=Y[mask], y_pred=P[mask],
            # 负对照：把真实标签的行顺序随机打乱（保留边缘分布、破坏配对）
            y_perm=Y[np.random.default_rng(int(cfg["seed"])).permutation(mask.sum())],
            perturbation_id=np.asarray(ids)[mask],
            model=best_model, protocol=main_proto,
        )

    save_json({"protocols": list(protos), "models": models,
               "n_perturbations": int(len(ids)),
               "n_features": int(X_df.shape[1]),
               "n_genes": int(Y_df.shape[1])},
              RES / "metrics/predict_setup.json")

    # ---- 扰动类型分层 ------------------------------------------------
    strat = None
    if oof:
        pg_all = agent7_validate.per_perturbation_metrics(
            Y[mask], P[mask], list(np.asarray(ids)[mask]))
        strat = agent7_validate.stratified_by_type(pg_all, meta.reset_index())
        strat.to_csv(RES / "tables/metrics_by_pert_type.csv", index=False,
                     encoding="utf-8-sig")
        log.info("\n按扰动类型分层：\n%s", strat.round(3).to_string(index=False))

    log.info("阶段 3 完成")
    return {"metrics": metrics, "summary": summ, "oof": oof, "Y": Y,
            "Y_df": Y_df, "ids": list(ids), "strat": strat}


# ===========================================================================
# Stage 4 —— 智能体 7（生物验证）
# ===========================================================================
def stage_validate(cfg: dict, predict_out: dict | None = None) -> dict:
    log.info("=" * 78)
    log.info("阶段 4/5  生物学验证（智能体 7）")
    log.info("=" * 78)

    oof_path = RES / "metrics/oof_predictions.npz"
    if not oof_path.exists():
        log.warning("缺少 OOF 预测，跳过生物验证")
        return {}

    z = np.load(oof_path, allow_pickle=True)
    Y_true, Y_pred = z["y_true"], z["y_pred"]
    ids = list(z["perturbation_id"])

    Y_df = pd.read_parquet(PROC / "Y_expr.parquet")
    Y_true_df = pd.DataFrame(Y_true, index=ids, columns=Y_df.columns[:Y_true.shape[1]])
    Y_pred_df = pd.DataFrame(Y_pred, index=ids, columns=Y_true_df.columns)

    gene_info = agent7_validate.ensure_probe_mapping(cfg, list(Y_df.columns))
    gene_sets = agent7_validate.ensure_hallmark_gmt(cfg)

    result: dict = {}
    if gene_info is None or not gene_sets:
        log.warning("通路基因集/基因映射不可用，只输出逐扰动指标")
        result["pathway_available"] = False
        per_pert = agent7_validate.per_perturbation_metrics(Y_true, Y_pred, ids)
        per_pert.to_csv(RES / "tables/per_perturbation_metrics.csv", index=False,
                        encoding="utf-8-sig")
        save_json({"pathway_available": False}, RES / "metrics/validation.json")
        return result

    probe2sym = dict(gene_info)

    t_scores = agent7_validate.pathway_scores(Y_true_df, gene_sets, probe2sym, cfg)
    p_scores = agent7_validate.pathway_scores(Y_pred_df, gene_sets, probe2sym, cfg)

    cons = agent7_validate.pathway_consistency(t_scores, p_scores)
    log.info("通路一致性: %s", cons)

    t_scores.to_csv(RES / "tables/pathway_scores_true.csv", encoding="utf-8-sig")
    p_scores.to_csv(RES / "tables/pathway_scores_pred.csv", encoding="utf-8-sig")

    enr = agent7_validate.gsea_like_enrichment(t_scores, p_scores, Y_true_df,
                                               int(cfg["validation"]["top_n_pathways"]))
    enr.to_csv(RES / "tables/pathway_enrichment.csv", index=False, encoding="utf-8-sig")

    per_pert = agent7_validate.per_perturbation_metrics(Y_true, Y_pred, ids)
    per_pert.to_csv(RES / "tables/per_perturbation_metrics.csv", index=False,
                    encoding="utf-8-sig")

    meta = pd.read_parquet(PROC / "perturbation_meta.parquet")
    strat = agent7_validate.stratified_by_type(per_pert, meta)
    strat.to_csv(RES / "tables/validation_by_pert_type.csv", index=False,
                 encoding="utf-8-sig")

    out = {"pathway_available": True, **cons,
           "n_perturbations_scored": int(len(ids)),
           "stratified": strat.to_dict(orient="records")}
    save_json(out, RES / "metrics/validation.json")
    result.update(out)
    result["t_scores"], result["p_scores"] = t_scores, p_scores
    log.info("阶段 4 完成")
    return result


# ===========================================================================
# Stage 5 —— 出图
# ===========================================================================
def stage_figures(cfg: dict, batch_out: dict | None = None,
                  validate_out: dict | None = None) -> dict:
    log.info("=" * 78)
    log.info("阶段 5/5  出图")
    log.info("=" * 78)
    FIG = RES / "figures"
    FIG.mkdir(parents=True, exist_ok=True)
    made = {}

    # fig1 / fig2 ---------------------------------------------------------
    oof_path = RES / "metrics/oof_predictions.npz"
    if oof_path.exists():
        z = np.load(oof_path, allow_pickle=True)
        Y_true, Y_pred, Y_perm = z["y_true"], z["y_pred"], z["y_perm"]
        made["fig1"] = figures.fig1_pred_vs_true(Y_true, Y_pred, Y_perm, cfg,
                                                 FIG / "fig1_pred_vs_true.png")
        r = agent5_predict.gene_correlations(Y_true, Y_pred)
        r2 = agent5_predict.gene_r2(Y_true, Y_pred)
        rp = agent5_predict.gene_correlations(Y_true, Y_perm)
        r2p = agent5_predict.gene_r2(Y_true, Y_perm)
        made["fig2"] = figures.fig2_gene_distributions(
            {"模型": (r, r2), "负对照": (rp, r2p)}, cfg,
            FIG / "fig2_gene_r2_distribution.png")

    # fig3 ----------------------------------------------------------------
    umap_path = RES / "metrics/umap_batch.npz"
    if umap_path.exists():
        u = np.load(umap_path, allow_pickle=True)
        diag = {}
        bd = RES / "metrics/batch_diagnosis.json"
        if bd.exists():
            from bioinfo_wam.common import read_json

            diag = read_json(bd)
        made["fig3"] = figures.fig3_umap_batch(
            u["before"], u["after"], u["batch"].astype(str), cfg,
            FIG / "fig3_umap_batch.png",
            diag.get("before", {}).get("accuracy"),
            diag.get("after", {}).get("accuracy"))
        if bd.exists():
            made["fig7"] = figures.fig7_batch_diagnosis(diag, cfg,
                                                        FIG / "fig7_batch_diagnosis.png")

    # fig4 / fig9 ---------------------------------------------------------
    met_path = RES / "metrics/metrics_by_fold.csv"
    if met_path.exists():
        met = pd.read_csv(met_path)
        made["fig4"] = figures.fig4_seen_vs_unseen(met, cfg,
                                                   FIG / "fig4_ood_protocols.png")
        made["fig9"] = figures.fig9_metrics_summary(met, cfg,
                                                    FIG / "fig9_metrics_summary.png")

    # fig5 ----------------------------------------------------------------
    t_path = RES / "tables/pathway_scores_true.csv"
    p_path = RES / "tables/pathway_scores_pred.csv"
    if validate_out and validate_out.get("pathway_available"):
        t_scores, p_scores = validate_out["t_scores"], validate_out["p_scores"]
    elif t_path.exists() and p_path.exists():
        t_scores = pd.read_csv(t_path, index_col=0)
        p_scores = pd.read_csv(p_path, index_col=0)
    else:
        t_scores, p_scores = pd.DataFrame(), pd.DataFrame()
    made["fig5"] = figures.fig5_pathway(t_scores, p_scores, cfg,
                                        FIG / "fig5_pathway_consistency.png")

    # fig6 ----------------------------------------------------------------
    cov_path = RES / "tables/coverage.csv"
    if cov_path.exists():
        made["fig6"] = figures.fig6_coverage(pd.read_csv(cov_path), cfg,
                                             FIG / "fig6_coverage.png")

    log.info("阶段 5 完成，共 %d 张图", len(made))
    return made


# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--stage", default="all",
                    choices=["build", "batch", "predict", "validate", "figures", "all"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    t0 = time.time()

    need = (["build", "batch", "predict", "validate", "figures"]
            if args.stage == "all" else [args.stage])

    batch_out = validate_out = None
    if args.stage in ("batch", "figures", "all"):
        if (RES / "metrics/batch_diagnosis.json").exists() and args.stage != "all":
            batch_out = {"diag": __import__("json").loads(
                (RES / "metrics/batch_diagnosis.json").read_text())}

    if "build" in need:
        stage_build(cfg)
    if "batch" in need:
        batch_out = stage_batch(cfg)
    if "predict" in need:
        stage_predict(cfg)
    if "validate" in need:
        validate_out = stage_validate(cfg)
    if "figures" in need:
        stage_figures(cfg, batch_out, validate_out)

    log.info("=" * 78)
    log.info("全部完成，用时 %.1f 分钟", (time.time() - t0) / 60)
    log.info("结果目录: %s", RES)
    log.info("=" * 78)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
