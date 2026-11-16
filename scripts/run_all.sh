#!/usr/bin/env bash
# 一键复现全流程：下载 → 造表 → 批次 → 预测 → 验证 → 出图
#
#   bash scripts/run_all.sh              # 标准全流程（约 20–40 分钟）
#   bash scripts/run_all.sh --no-download # 数据已就位，跳过下载
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

PY=".venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "[错误] 未找到虚拟环境 .venv，请先执行："
  echo "  /home/dongxu/.conda/envs/EfficientWAM/bin/python -m venv --system-site-packages .venv"
  echo "  .venv/bin/python -m pip install scipy seaborn pyyaml statsmodels umap-learn gseapy"
  exit 1
fi

echo "================================================================"
echo " 项目目录: $HERE"
echo " 解释器  : $($PY -V)"
echo "================================================================"

SKIP_DOWNLOAD=0
SKIP_SWEEP=0
for arg in "$@"; do
  [[ "$arg" == "--no-download" ]] && SKIP_DOWNLOAD=1
  [[ "$arg" == "--no-sweep" ]] && SKIP_SWEEP=1
done

if [[ $SKIP_DOWNLOAD -eq 0 ]]; then
  echo ""
  echo ">>> [0/5] 下载 Rosetta 配对数据"
  $PY scripts/00_download.py || echo "[警告] 下载未完全成功，继续尝试用已有数据"
else
  echo ""
  echo ">>> [0/5] 跳过下载"
  $PY scripts/00_download.py --verify || true
fi

for stage in build batch predict validate figures; do
  case $stage in
    build)    idx="1" ;;
    batch)    idx="2" ;;
    predict)  idx="3" ;;
    validate) idx="4" ;;
    figures)  idx="5" ;;
  esac
  echo ""
  echo ">>> [$idx/5] $stage"
  $PY scripts/run_pipeline.py --stage "$stage"
done

if [[ $SKIP_SWEEP -eq 0 ]]; then
  echo ""
  echo ">>> [附加] ComBat 校正强度扫描（约 10 分钟，生成 fig8）"
  $PY scripts/98_batch_strength_sweep.py || echo "[警告] 扫描失败，不影响主结果"
fi

echo ""
echo ">>> [附加] 生成结果摘要 结果摘要.md"
$PY scripts/99_summary_report.py > /dev/null && echo "[已写入] 结果摘要.md"

echo ""
echo "================================================================"
echo " 完成。"
echo "   图   : results/figures/"
echo "   表   : results/tables/"
echo "   指标 : results/metrics/"
echo "   日志 : logs/pipeline.log"
echo "================================================================"
