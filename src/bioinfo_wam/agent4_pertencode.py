"""智能体 4 —— 扰动信息编码（Perturbation Encoding Agent）。

把"我对细胞做了什么"编码成向量，供三类用途：

  a) **机制解释**：化合物结构相似 / 靶点相同 → 编码相近
  b) **辅助模型**：morphology + 扰动编码 → 表达（性能上界参考）
  c) **泄漏诊断**：说明把扰动身份喂给模型在"未见扰动"任务上是作弊
     ——这是本作业"相关 vs 因果"讨论的关键证据

三种编码
--------
* 化合物：RDKit Morgan 指纹（radius=2, 1024 bit）→ TruncatedSVD 降维。
  需要 SMILES 表（`data/reference/compound_smiles.csv`）。
  Rosetta 发布表里**不含 SMILES**，因此本脚本会自动从 LINCS 官方
  pert_info 下载并缓存；如果拿不到，则退化为 **结构无关的 ID 哈希编码**
  并在报告里标注该降级。
* 遗传扰动（ORF/CRISPR）：基因符号 one-hot → TruncatedSVD 降到 32 维。
* 扰动物种类型：compound / orf / crispr 的 one-hot。

输出 `perturbation_encoding.parquet`（行 = perturbation_id）
"""

from __future__ import annotations

import gzip
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .common import PROJECT_ROOT, get_logger, path_of

log = get_logger("agent4.encode")

LINCS_PERT_INFO_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/"
    "GSE92742_Broad_LINCS_pert_info.txt.gz"
)


# ---------------------------------------------------------------------------
# SMILES 参考表
# ---------------------------------------------------------------------------
def ensure_smiles_table(cfg: dict, force: bool = False) -> Path | None:
    """确保 `data/reference/compound_smiles.csv` 存在（列: pert_id, smiles）。"""
    out = PROJECT_ROOT / "data/reference/compound_smiles.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not force:
        return out
    tmp = out.with_suffix(".txt.gz")
    try:
        log.info("[encode] 下载 LINCS 化合物结构信息 ...")
        req = urllib.request.Request(LINCS_PERT_INFO_URL,
                                     headers={"Connection": "close"})
        with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    except Exception as exc:  # noqa: BLE001
        log.warning("[encode] 化合物结构信息下载失败(%s)，将使用降级编码",
                    type(exc).__name__)
        if tmp.exists():
            tmp.unlink()
        return None

    try:
        df = pd.read_csv(tmp, sep="\t", low_memory=False)
        col_map = {c.lower(): c for c in df.columns}
        pid = col_map.get("pert_id")
        smi = col_map.get("canonical_smiles") or col_map.get("smiles")
        if pid is None or smi is None:
            log.warning("[encode] pert_info 缺少 pert_id/smiles 列: %s", list(df.columns))
            return None
        sub = df[[pid, smi]].dropna().drop_duplicates(pid)
        sub.columns = ["pert_id", "smiles"]
        sub = sub[sub.smiles.astype(str).str.len() > 0]
        sub.to_csv(out, index=False)
        log.info("[encode] 化合物结构表已缓存: %d 条 → %s", len(sub), out.name)
        return out
    except Exception as exc:  # noqa: BLE001
        log.warning("[encode] 解析 pert_info 失败: %s", exc)
        return None
    finally:
        if tmp.exists():
            tmp.unlink()


# ---------------------------------------------------------------------------
# 编码
# ---------------------------------------------------------------------------
def _base_pert_id(pid: str) -> str:
    """去掉剂量的后缀：BRD-K12345678-001-01-1 → BRD-K12345678。"""
    parts = str(pid).split("-")
    if len(parts) >= 3 and parts[0].upper() == "BRD":
        return "-".join(parts[:3])
    return str(pid)


def _morgan_matrix(smiles: list[str], n_bits: int, radius: int) -> tuple[np.ndarray, list[int]]:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")
    rows, ok_idx = [], []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi)) if isinstance(smi, str) else None
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.float32)
        for bit in fp.GetOnBits():
            arr[bit] = 1.0
        rows.append(arr)
        ok_idx.append(i)
    if not rows:
        return np.zeros((0, n_bits), dtype=np.float32), []
    return np.vstack(rows), ok_idx


def _onehot_svd(keys: list[str], dim: int, seed: int) -> np.ndarray:
    from sklearn.decomposition import TruncatedSVD

    n = len(keys)
    if n == 0:
        return np.zeros((0, dim), dtype=np.float32)
    cats = sorted({k for k in keys})
    idx = {k: i for i, k in enumerate(cats)}
    M = np.zeros((n, len(cats)), dtype=np.float32)
    for i, k in enumerate(keys):
        M[i, idx[k]] = 1.0
    k = int(min(dim, max(1, min(M.shape) - 1)))
    Z = TruncatedSVD(n_components=k, random_state=seed).fit_transform(M)
    if Z.shape[1] < dim:
        Z = np.pad(Z, ((0, 0), (0, dim - Z.shape[1])))
    return Z.astype(np.float32)


def build(meta_pert: pd.DataFrame, cfg: dict,
          smiles_table: Path | None = None,
          smiles_map: dict[str, str] | None = None
          ) -> tuple[pd.DataFrame, dict]:
    """meta_pert 需要列: perturbation_id, pert_type, pert_target, pert_gene

    smiles_map 优先级最高（Rosetta 的 L1000 表内自带 Metadata_SMILES）；
    smiles_table 作为后备（外部下载的 LINCS pert_info）。
    """
    p_cfg = cfg["perturbation"]
    seed = int(cfg["seed"])
    n_bits = int(p_cfg["morgan_bits"])
    radius = int(p_cfg["morgan_radius"])

    meta = meta_pert.drop_duplicates("perturbation_id").copy()
    pids = meta["perturbation_id"].tolist()
    types = meta["pert_type"].tolist()
    targets = meta["pert_target"].astype(str).tolist()
    genes = (meta["pert_gene"].fillna("").astype(str).tolist()
             if "pert_gene" in meta.columns else [""] * len(pids))

    blocks: list[np.ndarray] = []
    colnames: list[str] = []
    diagnostics: dict = {"n_perturbations": len(pids)}

    # ---- 1. 化合物结构编码 ----------------------------------------------
    smiles_map: dict[str, str] = dict(smiles_map or {})
    if smiles_table and Path(smiles_table).exists():
        try:
            tab = pd.read_csv(smiles_table)
            raw = dict(zip(tab["pert_id"].astype(str), tab["smiles"].astype(str)))
            for k, v in raw.items():
                smiles_map.setdefault(k, v)
                # LINCS pert_info 里是去掉剂量的基础编号，而 Rosetta 表里
                # 常带 -001-01-1 这类剂量后缀，两种键都建上。
                smiles_map.setdefault(_base_pert_id(k), v)
        except Exception as exc:  # noqa: BLE001
            log.warning("[encode] 读取结构表失败: %s", exc)
    # 同时为传入的映射补上基础编号键
    for k, v in list(smiles_map.items()):
        smiles_map.setdefault(_base_pert_id(k), v)

    is_compound = np.array([t == "compound" for t in types])
    n_comp = int(is_compound.sum())
    comp_dim = int(p_cfg["compound_svd_dim"])
    comp_block = np.zeros((len(pids), comp_dim), dtype=np.float32)

    if n_comp:
        comp_idx = np.where(is_compound)[0]
        comp_ids = [targets[i] for i in comp_idx]
        smis = [smiles_map.get(pid) or smiles_map.get(_base_pert_id(pid))
                for pid in comp_ids]
        n_has_smiles = sum(1 for s in smis if isinstance(s, str))
        diagnostics["compound_total"] = n_comp
        diagnostics["compound_with_smiles"] = int(n_has_smiles)

        used_rdkit = False
        try:
            import rdkit  # noqa: F401

            if n_has_smiles >= max(3, 0.05 * n_comp):
                fp, ok = _morgan_matrix(smis, n_bits, radius)
                if fp.shape[0] >= 3:
                    from sklearn.decomposition import TruncatedSVD

                    k = int(min(comp_dim, fp.shape[0] - 1, fp.shape[1]))
                    Z = TruncatedSVD(n_components=k, random_state=seed).fit_transform(fp)
                    Z = Z / (Z.std(axis=0, keepdims=True) + 1e-9)
                    comp_block[comp_idx[ok], :k] = Z.astype(np.float32)
                    used_rdkit = True
                    diagnostics["compound_encoded_by"] = "morgan_rdkit"
                    log.info("[encode] 化合物 Morgan 指纹编码: %d/%d 条有 SMILES",
                             len(ok), n_comp)
        except ImportError:
            pass

        if not used_rdkit:
            Z = _onehot_svd(comp_ids, comp_dim, seed)
            comp_block[comp_idx] = Z
            diagnostics["compound_encoded_by"] = "id_hash_svd(降级: 无 SMILES 或无 RDKit)"
            log.warning("[encode] 化合物使用 ID 哈希降级编码（缺 SMILES 或 RDKit）")

    blocks.append(comp_block)
    colnames += [f"cmp_{i:03d}" for i in range(comp_dim)]

    # ---- 2. 遗传扰动基因编码 --------------------------------------------
    gene_dim = int(p_cfg["gene_svd_dim"])
    gene_block = np.zeros((len(pids), gene_dim), dtype=np.float32)
    is_genetic = np.array([t in ("orf", "crispr", "shrna") for t in types])
    if is_genetic.any():
        g_idx = np.where(is_genetic)[0]
        g_keys = [genes[i] or targets[i] for i in g_idx]
        Z = _onehot_svd(g_keys, gene_dim, seed)
        gene_block[g_idx] = Z
        diagnostics["genetic_total"] = int(len(g_idx))
        diagnostics["genetic_encoded_by"] = "gene_symbol_svd"
    blocks.append(gene_block)
    colnames += [f"gene_{i:03d}" for i in range(gene_dim)]

    # ---- 3. 扰动类型 one-hot --------------------------------------------
    type_order = ["compound", "orf", "crispr", "shrna"]
    type_block = np.zeros((len(pids), len(type_order)), dtype=np.float32)
    for i, t in enumerate(types):
        if t in type_order:
            type_block[i, type_order.index(t)] = 1.0
    blocks.append(type_block)
    colnames += [f"type_{t}" for t in type_order]

    enc = pd.DataFrame(np.hstack(blocks), index=pd.Index(pids, name="perturbation_id"),
                       columns=colnames)
    log.info("[encode] 编码完成: %d 扰动 x %d 维", enc.shape[0], enc.shape[1])
    return enc, diagnostics
