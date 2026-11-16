"""公共工具：配置加载、统一 schema、数据读写、随机种子。"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_project_root"] = str(PROJECT_ROOT)
    return cfg


def path_of(cfg: dict, key: str) -> Path:
    """config 里的相对路径统一相对项目根目录解析。"""
    resolve = cfg.get("_resolve_paths", True)
    p = Path(key)
    return p if p.is_absolute() or not resolve else PROJECT_ROOT / p


def ensure_dirs(cfg: dict) -> None:
    for sub in ("data/processed", "data/reference", "results/figures",
                "results/tables", "results/metrics", "results/models", "logs"):
        (PROJECT_ROOT / sub).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
def get_logger(name: str, log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(name)-22s | %(levelname)-7s | %(message)s",
                            datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# 随机种子
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# 统一 schema
# ---------------------------------------------------------------------------
# 所有中间表的主键，字段表
#   replicate_id      每个实验重复（一孔）唯一标识
#   perturbation_id   扰动唯一标识（跨数据集全局唯一）
#   dataset           来源子集
#   cell_line         细胞系
#   timepoint         处理时长（小时）
#   pert_type         compound | orf | crispr | control
#   pert_target       化合物 ID 或基因符号
#   batch             批次（孔板）
META_COLS = ["replicate_id", "perturbation_id", "dataset", "cell_line",
             "timepoint", "pert_type", "pert_target", "batch"]

# Cell Painting 元数据列（source of truth）
CP_META_MAP = {
    "Metadata_Plate": "batch",
    "Metadata_Well": "well",
    "Metadata_pert_id": "pert_id_raw",
    "Metadata_pert_type": "pert_type_raw",
    "Metadata_cell_id": "cell_line",
    "Metadata_genesymbol": "gene_symbol",
    "Metadata_genesymbol_mutation": "gene_symbol_mutation",
    "Metadata_pert_timepoint": "timepoint",
    "Metadata_pert_dose_micromolar": "dose",
    "Metadata_pert_iname": "pert_iname",
}

L1K_META_MAP = {
    "Metadata_Plate": "batch",
    "Metadata_Well": "well",
    "Metadata_pert_id": "pert_id_raw",
    "Metadata_pert_type": "pert_type_raw",
    "Metadata_cell_id": "cell_line",
    "Metadata_genesymbol_mutation": "gene_symbol_mutation",
    "Metadata_ARP_ID": "arp_id",
    "Metadata_pert_timepoint": "timepoint",
    # Rosetta 的 L1000 表内直接带化合物结构，无需另拉外部参考表
    "Metadata_SMILES": "smiles",
    "Metadata_pert_iname": "pert_iname",
}

CONTROL_LABELS = {
    "negcon", "negcon_", "EMPTY", "EMPTY_", "control", "DMSO", "none",
    "Luciferase", "LacZ", "eGFP",
}


def is_control_label(value: Any) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip() in CONTROL_LABELS


def infer_pert_type(dataset: str, pert_id: Any, gene_symbol: Any,
                    raw_type: Any) -> str:
    """把原始标签归一化为 compound / orf / crispr / control。

    扰动模态是**筛选实验的属性**，因此以数据集名为准；只有编号前缀
    （BRDN* = ORF 构建体）作为补充证据。不能用基因符号列判断——
    化合物数据集里该列也可能被填入化合物名称。
    """
    if is_control_label(pert_id) or str(raw_type).lower() != "trt":
        return "control"
    ds = dataset.upper()
    if "ORF" in ds:
        return "orf"
    if "CRISPR" in ds or "CRISPER" in ds:
        return "crispr"
    pid = str(pert_id)
    if pid.startswith("BRDN"):
        return "orf"
    # TRCN* 是 Broad RNAi 联盟的 shRNA 敲低构建体（LUAD-BBBC041 即此类）
    if pid.startswith("TRCN"):
        return "shrna"
    return "compound"


@dataclass
class DatasetPair:
    """一个子集的两模态原始表。"""
    name: str
    cp: pd.DataFrame
    l1k: pd.DataFrame
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def n_cp(self) -> int:
        return len(self.cp)

    @property
    def n_l1k(self) -> int:
        return len(self.l1k)


def raw_path(cfg: dict, dataset: str, modality: str) -> Path:
    fn = {"cp": "replicate_level_cp_augmented.parquet",
          "l1k": "replicate_level_l1k.parquet"}[modality]
    return path_of(cfg, f"data/raw/{dataset}/{fn}")


def available_datasets(cfg: dict) -> list[str]:
    out = []
    for ds in cfg["datasets"]["include"]:
        if raw_path(cfg, ds, "cp").exists() and raw_path(cfg, ds, "l1k").exists():
            out.append(ds)
    return out


def load_pair(cfg: dict, dataset: str) -> DatasetPair | None:
    cp_p, l1k_p = raw_path(cfg, dataset, "cp"), raw_path(cfg, dataset, "l1k")
    if not (cp_p.exists() and l1k_p.exists()):
        return None
    cp = pd.read_parquet(cp_p)
    l1k = pd.read_parquet(l1k_p)
    return DatasetPair(name=dataset, cp=cp, l1k=l1k)


# ---------------------------------------------------------------------------
# 元数据标准化
# ---------------------------------------------------------------------------
def standardize_meta(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """只保留 mapping 里声明过的列，并重命名为统一名字。"""
    keep = {k: v for k, v in mapping.items() if k in df.columns}
    out = df[list(keep)].rename(columns=keep).copy()
    return out


def canonical_meta(raw: pd.DataFrame, dataset: str, kind: str,
                   genetic_key: str = "gene") -> pd.DataFrame:
    """把原始元数据转换为统一 schema。

    kind = "cp" | "l1k"
    genetic_key = "gene" | "pert_id"
        遗传扰动用什么做主键。两侧模态都提供基因符号时用基因（同一基因的多个
        构建体属于同一生物学扰动，配对率更高）；只要有一侧没有基因列，就必须
        回退到 pert_id，否则两模态永远对不上。
    """
    mapping = CP_META_MAP if kind == "cp" else L1K_META_MAP
    m = standardize_meta(raw, mapping)

    if "batch" not in m.columns or "well" not in m.columns:
        raise KeyError(f"{dataset}/{kind} 缺少 batch 或 well 列，实际列: {list(raw.columns[:12])}")

    m["dataset"] = dataset
    if "cell_line" not in m.columns:
        m["cell_line"] = "unknown"
    if "timepoint" not in m.columns:
        m["timepoint"] = np.nan
    if "pert_id_raw" not in m.columns:
        m["pert_id_raw"] = np.nan
    if "pert_type_raw" not in m.columns:
        m["pert_type_raw"] = "trt"

    # 批次名加数据集前缀，避免不同数据集孔板编号撞车
    m["batch"] = m["batch"].astype(str).map(lambda s: f"{dataset}::{s}")
    m["replicate_id"] = m["batch"] + "::" + m["well"].astype(str)

    def _clean(s: pd.Series) -> pd.Series:
        bad = {"EMPTY", "EMPTY_", "", "nan", "None", "NaN"}
        return s.where(~s.astype(str).isin(bad), np.nan)

    if "gene_symbol" in m.columns:
        genes = _clean(m["gene_symbol"])
    else:
        genes = pd.Series(np.nan, index=m.index, dtype=object)
    if "gene_symbol_mutation" in m.columns:
        genes_mut = _clean(m["gene_symbol_mutation"])
    else:
        genes_mut = pd.Series(np.nan, index=m.index, dtype=object)
    gene = genes.fillna(genes_mut)

    m["pert_type"] = [
        infer_pert_type(dataset, pid, g, t)
        for pid, g, t in zip(m["pert_id_raw"], gene, m["pert_type_raw"])
    ]

    # 连接键：化合物用 pert_id（不含剂量的基础编号”，见 agent4），
    # 遗传扰动用**基因符号**——同一基因的多个构建体属于同一生物学扰动，
    # 且两模态的构建体编号体系不一定一致，用基因名能显著提高配对率。
    m["pert_gene"] = [
        str(g) if isinstance(g, str) and g not in ("nan", "None") else ""
        for g in gene
    ]
    targets, keys = [], []
    for pid, pt, g in zip(m["pert_id_raw"], m["pert_type"], m["pert_gene"]):
        if pt == "control":
            targets.append("control"); keys.append("control")
        elif pt in ("orf", "crispr", "shrna") and g and genetic_key == "gene":
            targets.append(g); keys.append(f"GENE::{g}")
        else:
            targets.append(str(pid)); keys.append(str(pid))
    m["pert_target"] = targets
    m["perturbation_id"] = m["dataset"] + "::" + pd.Series(keys, index=m.index)
    return m


# 向后兼容别名
def build_ids(meta: pd.DataFrame, dataset: str) -> pd.DataFrame:
    return meta


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def feature_columns(df: pd.DataFrame, meta: Iterable[str]) -> list[str]:
    meta = set(meta)
    cols = [c for c in df.columns if c not in meta]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    a, b = a[ok], b[ok]
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    rho, _ = spearmanr(a[ok], b[ok])
    return float(rho)


def save_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def describe_table(df: pd.DataFrame, name: str) -> dict:
    return {
        "name": name,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns_sample": list(df.columns[:10]),
        "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 1e6, 2),
    }
