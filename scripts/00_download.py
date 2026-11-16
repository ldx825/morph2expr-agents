#!/usr/bin/env python3
"""00 — 数据下载智能体（DownLoader）

从公开 S3 桶 `cellpainting-gallery` 拉取 **Rosetta 配对数据集**
（同一批扰动同时做了 Cell Painting 形态学成像 与 L1000 转录组测序）。

来源
----
Cell Painting Gallery — cpg0003-rosetta
  https://registry.opendata.aws/cellpainting-gallery/
  论文: Way et al., "Morphology and gene expression profiling provide
        complementary information for mapping cell state" (bioRxiv 2021)

四个配对子集
-----------
LINCS-Pilot1          A549 细胞系，化合物扰动（主数据集）
CDRP-BBBC047-Bray     U2OS 细胞系，化合物扰动（大，~3 GB）
LUAD-BBBC041-Caicedo  LUAD 细胞系，化合物扰动（小）
TA-ORF-BBBC037-Rohban A549 细胞系，ORF 过表达 = 遗传扰动（因果锚点）

每个子集包含两份"replicate 级"表
  CellPainting/replicate_level_cp_augmented.parquet  形态学特征  X_morph
  L1000/replicate_level_l1k.parquet                  基因表达    Y_expr

用法
----
python scripts/00_download.py                # 下载 3 个必需子集（~600 MB）
python scripts/00_download.py --all          # 额外下载 CDRP（+3 GB）
python scripts/00_download.py --subsets LINCS-Pilot1 TA-ORF-BBBC037-Rohban
python scripts/00_download.py --verify       # 只校验已有文件
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import urllib.request
from pathlib import Path

BUCKET = "https://cellpainting-gallery.s3.amazonaws.com"
ROOT_PREFIX = "cpg0003-rosetta/broad/workspace/curated_preprocessed_data"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

SUBSETS = {
    "LINCS-Pilot1": ["CellPainting", "L1000"],
    "CDRP-BBBC047-Bray": ["CellPainting", "L1000"],
    "LUAD-BBBC041-Caicedo": ["CellPainting", "L1000"],
    "TA-ORF-BBBC037-Rohban": ["CellPainting", "L1000"],
}

# 小文件优先，保证最小可行流程尽早可用；CDRP 最大(3 GB)，作为可选扩展
DOWNLOAD_ORDER = [
    "TA-ORF-BBBC037-Rohban",
    "LUAD-BBBC041-Caicedo",
    "LINCS-Pilot1",
    "CDRP-BBBC047-Bray",
]
ESSENTIAL = ["TA-ORF-BBBC037-Rohban", "LUAD-BBBC041-Caicedo", "LINCS-Pilot1"]

FILES = {
    "CellPainting": "replicate_level_cp_augmented.parquet",
    "L1000": "replicate_level_l1k.parquet",
}


def remote_key(subset: str, modality: str) -> str:
    return f"{ROOT_PREFIX}/{subset}/{modality}/{FILES[modality]}"


def local_path(subset: str, modality: str) -> Path:
    return RAW_DIR / subset / FILES[modality]


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _remote_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD", headers={"Connection": "close"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return int(r.headers.get("Content-Length", 0))


def download_one(subset: str, modality: str, retries: int = 200) -> Path:
    """带无限断点续传的下载（本机代理会周期性断流，必须能续）。"""
    key = remote_key(subset, modality)
    dest = local_path(subset, modality)
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BUCKET}/{key}"

    total = 0
    for _ in range(20):
        try:
            total = _remote_size(url)
            break
        except Exception:  # noqa: BLE001
            time.sleep(2)
    if total <= 0:
        raise RuntimeError(f"无法获取文件大小: {url}")

    if dest.exists() and dest.stat().st_size == total:
        print(f"  [skip] {subset}/{FILES[modality]}  已完整 ({human(total)})", flush=True)
        return dest
    if dest.exists() and dest.stat().st_size > total:
        dest.unlink()  # 上次写坏了，重来

    t_start = time.time()
    attempt = 0
    while True:
        have = dest.stat().st_size if dest.exists() else 0
        if have >= total:
            break
        attempt += 1
        if attempt > retries:
            raise RuntimeError(f"下载失败(重试{retries}次): {subset}/{modality} @ {human(have)}")
        headers = {"Connection": "close"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=180) as r, open(dest, "ab") as f:
                while True:
                    chunk = r.read(1 << 20)  # 1 MiB
                    if not chunk:
                        break
                    f.write(chunk)
        except Exception as exc:  # noqa: BLE001
            got = dest.stat().st_size if dest.exists() else 0
            pct = 100.0 * got / total
            print(f"  [retry#{attempt}] {type(exc).__name__} @ {human(got)} ({pct:.1f}%)",
                  flush=True)
            time.sleep(min(2 + attempt, 15))
            continue

        got = dest.stat().st_size if dest.exists() else 0
        if got < total:
            print(f"  [retry#{attempt}] 提前结束 @ {human(got)} ({100.0*got/total:.1f}%)", flush=True)
            time.sleep(2)

    dt = time.time() - t_start
    print(f"  [ ok ] {subset}/{FILES[modality]}  {human(total)}  耗时 {dt:.0f}s "
          f"({total/1e6/max(dt,1):.2f} MB/s)", flush=True)
    return dest


def md5_of(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while blk := f.read(chunk):
            h.update(blk)
    return h.hexdigest()


def verify() -> int:
    ok, missing_ess = 0, []
    for subset in SUBSETS:
        for modality in SUBSETS[subset]:
            p = local_path(subset, modality)
            tag = "必需" if subset in ESSENTIAL else "可选"
            if p.exists() and p.stat().st_size > 0:
                print(f"  [ ok ] {p.relative_to(PROJECT_ROOT)}  {human(p.stat().st_size)}")
                ok += 1
            else:
                print(f"  [MISS] {p.relative_to(PROJECT_ROOT)}   ({tag})")
                if subset in ESSENTIAL:
                    missing_ess.append(f"{subset}/{modality}")
    need = len(ESSENTIAL) * 2
    print(f"\n共 {ok}/{len(SUBSETS) * 2} 个文件就位；必需文件缺失 {len(missing_ess)}")
    if missing_ess:
        print(f"缺失: {missing_ess}")
    return 0 if not missing_ess else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="*", default=None,
                    help="要下载的子集名（默认先下 3 个必需子集）")
    ap.add_argument("--all", action="store_true", help="包括可选的大数据集 CDRP")
    ap.add_argument("--verify", action="store_true", help="只校验，不下载")
    ap.add_argument("--md5", action="store_true", help="下载后计算 md5（慢）")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if args.verify:
        return verify()

    if args.subsets:
        subsets = args.subsets
    elif args.all:
        subsets = DOWNLOAD_ORDER
    else:
        subsets = ESSENTIAL
    unknown = [s for s in subsets if s not in SUBSETS]
    if unknown:
        print(f"未知子集: {unknown}\n可选: {list(SUBSETS)}", file=sys.stderr)
        return 2

    print("=" * 72)
    print("Rosetta 配对数据下载 (Cell Painting + L1000)")
    print(f"目标目录: {RAW_DIR}")
    print("=" * 72)

    t0 = time.time()
    for subset in subsets:
        print(f"\n[{subset}]")
        for modality in SUBSETS[subset]:
            p = download_one(subset, modality)
            if args.md5:
                print(f"  md5  {md5_of(p)}")

    print(f"\n总耗时 {time.time()-t0:.0f}s")
    verify()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
