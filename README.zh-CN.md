# 细胞形态 → 转录组 扰动响应预测（多智能体流水线）

> 生物信息学大作业 · 3 人 × 3 周
> 用 **Cell Painting 形态特征** 预测 **L1000 基因表达变化**，并在未见扰动上验证。

---

## 一分钟上手

```bash
cd "/home/dongxu/EfficientWAM/26.11.15临时交大作业用/生物信息学"

# 1) 下载配对数据（约 600 MB，首选 3 个子集）
.venv/bin/python scripts/00_download.py

# 2) 一键跑完全流程（造表 → 批次 → 预测 → 验证 → 出图 → 扫描 → 摘要）
bash scripts/run_all.sh --no-download

# 想快一点（跳过 10 分钟的校正强度扫描）
bash scripts/run_all.sh --no-download --no-sweep
```

跑完后看这三个地方：

| 位置 | 内容 |
|---|---|
| **`结果摘要.md`** | **所有指标已汇总，报告直接抄** |
| `results/figures/` | fig1 ~ fig9，直接塞进 PPT |
| `logs/pipeline.log` | 全过程日志（含每折指标） |

**先读这两份文档**：
- `方法与实验说明.md` —— 方法底稿，报告正文可直接改写
- `报告与PPT大纲.md` —— 给负责汇报同学的页面级大纲

---

## 环境

已为你建好 `.venv`（基于 conda 环境 `EfficientWAM` 的 Python 3.10，复用其中的
pandas / scikit-learn / matplotlib，额外装了 scipy / seaborn / statsmodels /
umap-learn / gseapy）。

```bash
.venv/bin/python -V                       # Python 3.10.20
.venv/bin/python -c "import umap, gseapy" # 确认可选依赖
```

需要重建环境时：

```bash
/home/dongxu/.conda/envs/EfficientWAM/bin/python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
```

> RDKit 未安装；缺失时化合物编码会自动降级为 ID 哈希（流水线不会中断）。
> 要启用真正的 Morgan 指纹：`.venv/bin/python -m pip install rdkit`

---

## 数据

来自 **Cell Painting Gallery** 的 `cpg0003-rosetta`（Broad Institute 公开 S3 桶）：

| 子集 | 细胞系 | 扰动类型 | 角色 |
|---|---|---|---|
| `LINCS-Pilot1` | A549 | 化合物 | 主数据集 |
| `TA-ORF-BBBC037-Rohban` | U2OS | ORF 过表达 | 遗传扰动 / 因果锚点 |
| `LUAD-BBBC041-Caicedo` | LUAD | 化合物 | 跨细胞系留出测试 |
| `CDRP-BBBC047-Bray` | U2OS | 化合物 | 可选（+3 GB） |

每个子集含两份 replicate 级表：形态特征（`replicate_level_cp_augmented.parquet`）
与 L1000 表达（`replicate_level_l1k.parquet`）。

> 两个模态是**分开做的实验**，只能在**扰动级**配对。这一点决定了全部评估协议的设计。

---

## 目录

```
├── config.yaml                  全部超参 / 阈值（唯一调参入口）
├── 方法与实验说明.md             方法底稿
├── 报告与PPT大纲.md              汇报大纲
├── data/
│   ├── raw/<子集>/              原始 parquet
│   ├── reference/               探针映射 / Hallmark 通路 / 化合物 SMILES
│   └── processed/               X_morph / Y_expr / meta / encoding
├── src/bioinfo_wam/
│   ├── agent1_qc.py             ① 质控
│   ├── agent2_morph.py          ② 形态特征
│   ├── agent3_expr.py           ③ 转录组预处理
│   ├── agent4_pertencode.py     ④ 扰动编码
│   ├── agent5_predict.py        ⑤ 跨模态预测
│   ├── agent6_batch.py          ⑥ 批次诊断与校正
│   ├── agent7_validate.py       ⑦ 生物验证
│   ├── combat.py                ComBat 校正实现
│   └── figures.py               9 张图
├── scripts/
│   ├── 00_download.py           数据下载（断点续传）
│   ├── run_pipeline.py          主流水线（--stage build/batch/predict/validate/figures）
│   └── run_all.sh               一键复现
└── results/{figures,tables,metrics}
```

---

## 分阶段运行

```bash
P=".venv/bin/python scripts/run_pipeline.py"
$P --stage build      # 智能体 1-4：造 X_morph / Y_expr / 覆盖表
$P --stage batch      # 智能体 6：批次诊断 + ComBat + UMAP
$P --stage predict    # 智能体 5：三种 OOD 协议 × 三个模型 × 置换负对照
$P --stage validate   # 智能体 7：Hallmark 通路一致性 + 按扰动类型分层
$P --stage figures    # 出图
```

单个阶段可独立重跑；已生成的中间表不会自动删除。

---

## 常见问题

**Q：`predict` 阶段太慢？**
把 `config.yaml` 里 `model.mlp_max_iter` 调小（如 200），或把
`model.permutation_repeats` 设为 1。`lasso_n_jobs` 可调到 32。

**Q：通路一致性算不出来（`n_pathways: 0`）？**
检查 `data/reference/h.all.symbols.gmt` 与 `probe2symbol.csv` 是否存在。
后者需要联网调 mygene.info 一次，之后离线可用。

**Q：化合物编码走了降级分支？**
说明没有 RDKit 或没有 SMILES。装 `rdkit` 即可：
`.venv/bin/python -m pip install rdkit`。

**Q：想改实验？**
只改 `config.yaml` 和 `src/bioinfo_wam/agent5_predict.py` 里的 `split_*` 函数，
其余代码不用动。

---

## 复现性

* 所有随机种子统一来自 `config.yaml` 的 `seed`（默认 `20260115`）
* 所有指标写入 `results/metrics/*.json` 与 `results/tables/*.csv`
* 智能体是**确定性脚本**，不调用任何 LLM 做计算——同一条命令得到同一份数字
