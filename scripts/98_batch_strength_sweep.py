#!/usr/bin/env python3
"""附加实验：ComBat 校正强度扫描 —— 量化「过校正」风险。

思路
----
把校正强度 λ 从 0（不校正）扫到 1（完整校正），每一步同时测量：

  * 批次可分性  —— 批次分类准确率（**越低越好**）
  * 生物学保留度 —— 扰动检索准确率 ÷ 随机基线（**越高越好**）

理想曲线：λ↑ 时前者迅速降到随机水平，后者基本持平。
如果后者也明显下降，就说明**在抹批次的同时把生物学信号一起抹掉了**（过校正）。

输出
----
results/figures/fig8_batch_strength_sweep.png
results/tables/batch_strength_sweep.csv

用法
----
    .venv/bin/python scripts/98_batch_strength_sweep.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bioinfo_wam.agent6_batch import (  # noqa: E402
    batch_classifier_accuracy,
    perturbation_retrieval,
)
from bioinfo_wam.combat import fit_transform  # noqa: E402
from bioinfo_wam.common import load_config, get_logger, save_json  # noqa: E402

log = get_logger("sweep", ROOT / "logs" / "sweep.log")

STRENGTHS = [0.0, 0.25, 0.5, 0.75, 1.0]
MAX_SAMPLES = 6000
SEED = 20260115


def main() -> int:
    cfg = load_config()
    cfg["_resolve_paths"] = False
    rng = np.random.default_rng(SEED)

    X = pd.read_parquet(ROOT / "data/processed/X_morph_replicate.parquet")
    M = pd.read_parquet(ROOT / "data/processed/meta_replicate.parquet")

    sel = rng.choice(len(X), size=min(MAX_SAMPLES, len(X)), replace=False)
    sel.sort()
    A = np.nan_to_num(X.iloc[sel].to_numpy(np.float64), nan=0.0,
                      posinf=0.0, neginf=0.0)
    batch = M.iloc[sel]["batch"].astype(str).to_numpy()
    pids = M.iloc[sel]["perturbation_id"].astype(str).to_numpy()
    log.info("扫描样本数 %d × %d 特征，批次数 %d", A.shape[0], A.shape[1],
             len(np.unique(batch)))

    # 批次分类器降迭代次数以控制耗时（趋势判断足够）
    cfg["batch"]["logistic_max_iter"] = 600

    rows = []
    for lam in STRENGTHS:
        if lam <= 0:
            Z = A.copy()
            info = {"applied": False}
        else:
            Z, info = fit_transform(A, batch, strength=lam, use_eb=False)

        b = batch_classifier_accuracy(Z, batch, cfg)
        ret, chance = perturbation_retrieval(Z, pids, False, batch, 1)
        row = {
            "strength": lam,
            "batch_accuracy": b.get("accuracy"),
            "batch_chance": b.get("chance"),
            "pert_retrieval": ret,
            "pert_retrieval_chance": chance,
            "retrieval_fold_over_chance": (ret / chance) if chance and chance > 0 else np.nan,
        }
        rows.append(row)
        log.info("λ=%.2f | 批次准确率 %.4f（随机 %.4f）| 扰动检索 %.4f（随机 %.5f，%.0f×）",
                 lam, row["batch_accuracy"], row["batch_chance"],
                 ret, chance, row["retrieval_fold_over_chance"])

    df = pd.DataFrame(rows)
    (ROOT / "results/tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(ROOT / "results/tables/batch_strength_sweep.csv", index=False,
              encoding="utf-8-sig")
    save_json(rows, ROOT / "results/metrics/batch_strength_sweep.json")

    # ---------------- 出图 ----------------
    from bioinfo_wam.figures import setup_matplotlib

    plt, _sns = setup_matplotlib(cfg)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(df.strength, df.batch_accuracy, "o-", lw=2.4, ms=9,
            color="#C44E52", label="批次分类准确率")
    ax.axhline(df.batch_chance.iloc[0], ls="--", color="k", lw=1.2,
               label=f"随机基线 ({df.batch_chance.iloc[0]:.3f})")
    ax.set_xlabel("ComBat 校正强度 λ", fontsize=12)
    ax.set_ylabel("批次分类准确率（越低越好）", fontsize=12)
    ax.set_title("批次信息去除效果", fontsize=13)
    ax.set_ylim(-0.02, 1.0)
    ax.legend(fontsize=10)

    ax = axes[1]
    ax.plot(df.strength, df.retrieval_fold_over_chance, "s-", lw=2.4, ms=9,
            color="#55A868")
    ax.axhline(1.0, ls="--", color="k", lw=1.2, label="随机水平 (1×)")
    ax.set_xlabel("ComBat 校正强度 λ", fontsize=12)
    ax.set_ylabel("扰动检索 / 随机基线（越高越好）", fontsize=12)
    ax.set_title("生物学信号保留度", fontsize=13)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=10)

    fig.suptitle("图8  校正强度扫描：批次去除 vs 生物学保留的权衡", fontsize=14)
    fig.tight_layout()
    out = ROOT / "results/figures/fig8_batch_strength_sweep.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("已保存 %s", out)
    print(f"\n[已保存] {out}")
    print(df.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
