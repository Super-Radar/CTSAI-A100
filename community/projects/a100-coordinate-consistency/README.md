# a100-coordinate-consistency：RawTarget/TraTarget 目标日志坐标约定与几何闭合分析

一个针对 CTSAI-A100 仓库自带目标日志（`*_RawTarget.csv` / `*_TraTarget.csv`）的**离线 Python 分析项目**：回答一个非常具体的问题——**日志中的笛卡尔列 X/Y 与极坐标列 range/angle 之间，究竟满足哪一条坐标约定？该约定在全语料上是否几何闭合？**

## 1. 项目目标

1. 解析 RadarTools 上位机导出的多帧目标日志（自写流式 CSV 解析，不依赖 pandas），提取每帧的 `FrameNb`、时间戳、epoch 毫秒与逐目标行；
2. 对 TraTarget 行检验 4 条候选坐标约定（假设族 H1–H4），以「预测点与 logged (X,Y) 的平面欧氏距离」为残差，逐文件给出最优假设、残差中位数/p95/max、以及次优假设与可区分裕度；
3. 可选扫描角度零点偏移（±5°、0.5° 步长），检验语料是否存在系统性角度零点偏差；
4. 对全部 41 个目标日志做字段有效性普查（重点 X、Y、heigh、speed、RCS、snr、DbfMag），Raw 与 Tra 分开汇总；
5. 产出可复现的图表（4 张）与结果表（4 份 + 摘要）。

## 2. 数据来源

* 目录：`RadarTools/RadarTools_Release/data/`；
* 文件名模式：`<YYYY_M_D_H_M_S>_Tatget_CTLRR-300-V2_1_RawTarget.csv` 与 `..._TraTarget.csv`（注意原文里 `Tatget` 的拼写即如此），共 **21 个 RawTarget + 20 个 TraTarget**（20 组同名配对 + 1 个仅 Raw 的会话）；
* 帧格式：首行 CSV 表头（不同上位机版本为 58/59/62 列，关键列索引一致）→ 每帧以 `START,FrameNb:N,...` 开始，随后是 `YYYY/MM/DD HH:MM:SS:mmm---->` 时间戳行、`heightOffset:0,<epoch_ms>,...` 元数据行、逐目标数值行，帧块以 `END,Total:N` 结束，文件末尾有 `BB Hw:...` 固件版本页脚；
* 这些文件是 **RadarTools 上位机记录的示例目标日志**（文件名标注 CTLRR-300-V2_1），属仓内官方附带资产，官方 Python 可视化文档亦用同款 CSV 作为示例数据。**它们不是 A100 硬件实测数据**，本项目的全部结论仅针对这份导出语料本身；
* 全部 41 个文件的 SHA-256 清单见 [`results/data_sha256.txt`](results/data_sha256.txt)。

## 3. 环境依赖

* Python 3.12；
* 第三方依赖仅 **numpy、matplotlib（Agg 后端）、pytest**——**不使用 pandas**，多帧 CSV 解析为本项目自写（`src/coord_consistency/parser.py`）；
* 开发与验证使用共享 venv（`numpy 2.5.3` / `matplotlib 3.11.2` / `pytest 9.1.1`，该 venv 亦含 scipy，但本项目未用到）。

## 4. 运行步骤

在项目目录（本 README 所在目录）下：

```bash
# 运行测试（合成数据，不需要语料也能全绿；语料存在时附带冒烟测试）
python -m pytest tests/ -q

# 一键复现全部结果与图表（随机种子固定 20261003，逐字节可复现）
PYTHONPATH=src python -m coord_consistency.cli
```

CLI 默认读取 `<仓库根>/RadarTools/RadarTools_Release/data/`，可用 `--data-dir` / `--out-dir` 覆盖。运行后 `figures/` 与 `results/` 内所有产物均为重新生成。

## 5. 结果

完整数值见 [`results/summary.md`](results/summary.md)（中文，自动生成）。关键结论：

* **最优坐标约定为 H1：`X = -r·sin(θ)`，`Y = r·cos(θ)`**（θ 单位为度、顺时针为正、X 指左、Y 指前）——20/20 个 TraTarget 文件一致选中，无一例外；
* H1 残差中位数 **0.0015 ~ 0.0041 m**（p95 0.0044 ~ 0.0097 m，最大 0.0216 m），即 logged X/Y 与极坐标反算点位在毫米~厘米级闭合；
* 可区分裕度（次优中位数 − 最优中位数）**4.02 ~ 22.59 m**，判别无歧义（次优恒为 H2，即仅 X 符号相反的镜像约定）；
* 角度零点偏移扫描（±5°、0.5° 步长）：全部 20 文件最优偏移 **0.0°**，无一因偏移改善——语料不支持存在系统性角度零点偏差；
* 字段普查：**RawTarget 的 X/Y 有效填充率为 0.0000%**（全为 0 或 1e-31 量级浮点垃圾值），不可用作笛卡尔坐标；TraTarget 的 X/Y 为 99.99%/100%。另注意 Tra 的 `snr`、`DbfMag`、`heigh` 有效率亦为 0%，Raw 的 `heigh` 为 0%（见 `results/field_census.csv` 与 fig4）。

图表（英文标签、含单位）：

| 图 | 内容 |
|---|---|
| `figures/fig1_closure_scatter.png` | Tra 全语料 logged X/Y 对 H1 预测的恒等散点（对角线） |
| `figures/fig2_residual_distribution.png` | H1 残差分布：合并直方图 + 按文件小提琴（log 轴） |
| `figures/fig3_hypothesis_margins.png` | 每文件最优 vs 次优中位残差（log 轴）与裕度标注 |
| `figures/fig4_field_census_heatmap.png` | 字段 × 文件有效填充率热图（Raw/Tra 分块） |

结果表：`results/closure_by_file.csv`（逐文件闭合统计与偏移扫描）、`results/hypothesis_margins.csv`（四假设逐文件中位数与裕度）、`results/field_census.csv`（字段普查，含 AGGREGATE 行）、`results/data_sha256.txt`（语料哈希）。

## 6. 局限

* **纯离线分析**：只读取仓内既有 CSV，不做任何采集或硬件在环验证；
* 语料为上位机示例目标日志（非 A100 硬件实测），且数量有限（20 个 Tra 会话、约 11.8 万有效行）；
* **导出自洽 ≠ 物理标定正确**：H1 胜出只说明上位机导出 X/Y 时使用的公式与 H1 一致，不能证明雷达安装角/轴方向在物理世界同样如此（例如 X 左 Y 前的物理含义依赖安装朝向）；
* RawTarget 的 X/Y 无有效值，Raw 语料只能做极坐标层面的分析；
* 坐标约定可能随上位机/固件版本变化（语料内已见 58/59/62 三种表头变体与逐帧 `END` 标记），换版本后应重跑本流水线核验。

## 7. 作者

chennaji9

## 8. 与现有社区项目的差异

`community/projects/` 下已合并的 5 个项目——`a100-indoor-cart-mti`、`a100-indoor-outdoor-radar-analysis`、`a100-matlab-adc-pipeline-validation`、`a100-matlab-adc-signal-processing`、`a100-perimeter-vehicle-radar-analysis`——分别聚焦 ADC 信号处理、MTI/CFAR、Range/Doppler FFT 与实测场景分析，**均不涉及目标日志（RawTarget/TraTarget）坐标约定的核验**。本项目是首个对 X/Y ↔ (range, angle) 几何闭合关系做系统检验与字段有效性普查的社区项目，且不修改任何官方工具，只读取官方附带数据。

## 项目结构

```
a100-coordinate-consistency/
├── README.md
├── .gitignore
├── src/coord_consistency/
│   ├── parser.py       # 多帧目标日志流式解析（无 pandas）
│   ├── hypotheses.py   # 坐标假设族 H1–H4 与角度偏移
│   ├── closure.py      # 闭合残差、统计、判别、弃权、偏移扫描
│   ├── census.py       # 字段有效性普查
│   └── cli.py          # 一键复现（结果表 + 摘要 + 4 张图）
├── tests/              # pytest：解析/假设/闭合/普查，30 个断言组用例
├── figures/            # fig1–fig4（cli 重新生成）
└── results/            # closure_by_file / hypothesis_margins / field_census / data_sha256 / summary.md
```
