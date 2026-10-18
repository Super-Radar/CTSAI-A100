# a100-filter-selection-bias：点云筛选阈值的选择偏差与交互作用

## 目标

研究 CTSAI-A100 目标日志（RawTarget 点云）在**距离 / 角度 / 速度 / SNR 阈值筛选**下产生的
选择偏差（selection bias）：每一维阈值删掉了多少观测、删在了哪些空间区域、不同维度的删除
集合重叠到什么程度、两个阈值叠加时是"独立损失"还是"双重打击"，以及筛选后幸存分布相对
原始分布的偏移。全部为**离线 Python 分析**，不含任何硬件在环步骤。

四个核心实验：

1. **近零速度过滤的整体隐藏程度**：`|speed| >= v_min` 从 0 扫到 2 m/s，看静态（近零速）
   观测被整体移除的比例与其空间分布（距离带 / 角度扇区分层 + 极坐标删除分布图）；
2. **各筛选维度删除集合的交集**：参考设置下逐维取删除集，做两两 Jaccard 相似度，
   找出"删的是同一批点"的维度对；
3. **snr_min 与 range_max 的联合作用**：二维联合网格上比较联合保留率与
   "两维独立"乘积基线，量化远距离低 SNR 观测被两维同时命中的双重打击；
4. **分布偏移**：筛选前后 range / angle / snr / speed 的均值与分位数曲线变化。

## 数据来源

- 主输入：`Python雷达可视化工具示例/负30度_Tatget_2026_03_10_15_34_39_CTLRR-300-V2_1_RawTarget.csv`（2463 帧）；
- 扩展集：`RadarTools/RadarTools_Release/data/` 下全部 21 个 `*_RawTarget.csv`；
- 共 22 个文件，全部为仓内 RadarTools 示例目标日志（设备标注 **CTLRR-300-V2_1**，
  **非 A100 硬件实测**）。22 个文件的逐文件 SHA256 见
  [`results/data_sha256.txt`](results/data_sha256.txt)；
- 帧格式：首行表头（命名列数有 57/58/59/62 四种变体，前 5 列恒为 `ObjId,range,speed,angle,snr`）
  → `START,FrameNb:N` → 时间戳行 → `heightOffset:` 元数据行 → 目标行 → `END,Total:N`。
  解析按**列位置**取值，不依赖表头名；任一关键字段缺失/NaN 的行整行剔除，不填零。

## 环境

- Python 3.12（仅在标准库 + 下列依赖上运行，**不使用 pandas**）
- numpy、scipy、matplotlib（Agg 后端，无显示环境要求）、pytest

## 运行步骤

```bash
cd community/projects/a100-filter-selection-bias

# 测试（20 项，含合成混合分布的解析解校验）
python -m pytest tests/ -q

# 复现全部结果（results/ + figures/，固定种子 20261003，确定性输出）
PYTHONPATH=src python -m filter_bias.cli
# 或：python src/filter_bias/cli.py
```

CLI 自动定位仓库根（亦可用 `--repo-root` 显式指定），读取 22 个 CSV，写出：
`results/retention_sweep.csv`、`results/jaccard.csv`、`results/joint_grid.csv`、
`results/distribution_shift.csv`、`results/data_sha256.txt`、`results/summary.md`
（中文关键结论）与 `figures/` 下 5 张英文标签图。

## 结果

| 文件 | 内容 |
|---|---|
| `results/retention_sweep.csv` | 五个维度的单阈值扫描（speed_min 含距离带/角度扇区分层、snr_min 含距离带分层）：总体与分层保留率 |
| `results/jaccard.csv` | 参考设置下各维度删除集合的两两 Jaccard 与包含率（含交集/并集计数；Jaccard 低但包含率高=单向覆盖） |
| `results/joint_grid.csv` | `v_min x s_min` 与 `R_max x s_min` 两个联合网格：联合保留率、独立乘积基线、交互量 |
| `results/distribution_shift.csv` | 多组阈值下筛选前后 range/angle/snr/speed 的均值与分位数（q05–q95）偏移 |
| `results/summary.md` | 中文关键结论（近零删除比例、最高 Jaccard 维度对、联合作用量级、分布偏移要点） |
| `figures/fig1_retention_curves.png` | 各维度保留率曲线（5 面板） |
| `figures/fig2_jaccard_matrix.png` | 删除集合两两 Jaccard 矩阵（热图） |
| `figures/fig3_joint_heatmap.png` | `v_min x s_min` 联合保留率热图 + `R_max x s_min` 交互量热图 |
| `figures/fig4_distribution_shift.png` | 筛选前后分位数曲线对比（range/angle/snr） |
| `figures/fig5_nearzero_spatial.png` | 近零速度过滤的空间删除分布（极坐标散点，删除红 / 保留蓝） |

## 合成基准（正确性验证）

`tests/test_filter_bias.py` 用已知混合分布（静态簇 + 运动簇 + 均匀背景，各簇字段为
meshgrid 网格，簇内字段独立）构造**解析可算**的保留率：任一阈值组合下各簇保留率
为逐字段通过率的乘积，总体保留率为按簇大小的加权平均。测试断言计算值与解析值
精确一致（误差 < 1e-12），并断言分层计数守恒（各带之和 = 总体）。另覆盖：
三表头变体解析、`END,Total` 帧界、时间戳/元数据行跳过、缺失/NaN 行剔除不填零、
全关=恒等、收紧单一阈值幸存集单调包含、Jaccard 代数性质（自反=1、不相交=0、对称）、
联合网格与直接组合掩码一致、分位数单调。

## 局限

- **无真值标注**：本工作只评价保留率与分布变化，不给出准确率/召回率，**不评选"最佳检测阈值"**；
- 数据为 RadarTools 示例目标日志（CTLRR-300-V2_1 标注），**非 A100 硬件实测**；
- 全部为**离线分析**：无实时性、无硬件在环验证；
- 语料为特定场景录制，结论的数值（删除比例、Jaccard、交互量）只对该语料成立，
  换场景需重跑 CLI。

## 作者

chennaji9

## 与仓内已有内容的关系（差异声明）

- 已合并的 5 个社区项目（a100-indoor-cart-mti、a100-indoor-outdoor-radar-analysis、
  a100-matlab-adc-signal-processing、a100-matlab-adc-pipeline-validation、
  a100-perimeter-vehicle-radar-analysis）聚焦 ADC 信号处理 / MTI / CFAR / 场景实测，
  **均不做目标导出流（RawTarget）筛选阈值的选择偏差研究**；本项目填补该空白，
  与它们无代码或数据依赖；
- 官方 `Python雷达可视化工具示例/radarViewer.py` 中的 `RadarFilter` 类仅为**只读概念参考**
  （其距离/角度/速度/SNR 筛选维度启发了本研究的阈值算子集合）：本项目不修改、不执行
  官方工具，也不重复其聚类 / 跟踪 / 可视化功能，是独立的筛选偏差研究；
- 本目录为纯新增，不触碰仓内任何其他文件。
