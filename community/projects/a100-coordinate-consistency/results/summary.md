# a100-coordinate-consistency 结果摘要

本文件由 `PYTHONPATH=src python -m coord_consistency.cli` 自动生成（随机种子 20261003），请勿手工编辑。

## 语料

- 目标日志共 41 个文件（21 个 `*_RawTarget.csv`、20 个 `*_TraTarget.csv`），来自 `RadarTools/RadarTools_Release/data/`，文件名标注 CTLRR-300-V2_1。
- 各文件 SHA-256 见 `data_sha256.txt`。
- Tra 语料有效目标行合计 117571 行（range/angle/X/Y 均可解析且非缺失）。

## 坐标约定闭合结论（TraTarget，残差 = 预测点与 logged (X,Y) 的平面欧氏距离）

- **最优假设：H1**（20/20 个 TraTarget 文件一致），即 **X = -r·sin(θ)，Y = r·cos(θ)**（θ 为度、顺时针为正、X 左 Y 前）。
- 最优残差中位数范围：**0.0015 ~ 0.0041 m**（20 个文件中位数的中位数 0.0036 m）。
- 最优残差 p95 范围：0.0044 ~ 0.0097 m；最大值范围：0.0054 ~ 0.0216 m。
- 可区分裕度（次优中位数 − 最优中位数）：**4.02 ~ 22.59 m**，全部文件裕度 > 0，判别无歧义。
- 角度零点偏移扫描（±5°、0.5° 步长）：最优偏移集合 [0.0]°，0/20 个文件因偏移而改善——即语料不支持存在系统性角度零点偏移。

## 字段有效性普查（有效 = 可解析且 |值| > 1e-6）

- **RawTarget 的 X/Y 有效填充率为 0.0000% / 0.0000%（全为 0 或 1e-31 量级浮点垃圾值），不可用作笛卡尔坐标**。
- TraTarget 的 X/Y 有效填充率为 99.9872% / 100.0000%。

| 字段 | Raw 有效填充率 | Tra 有效填充率 |
|---|---|---|
| ObjId | 0.9777 | 0.9978 |
| range | 1.0000 | 1.0000 |
| speed | 0.4844 | 0.4106 |
| angle | 0.9995 | 0.9543 |
| snr | 1.0000 | 0.0000 |
| X | 0.0000 | 0.9999 |
| Y | 0.0000 | 1.0000 |
| heigh | 0.0000 | 0.0000 |
| RCS | 0.9736 | 0.6593 |
| DbfMag | 1.0000 | 0.0000 |

## 图

- `figures/fig1_closure_scatter.png`：Tra 全语料 logged X/Y 与 H1 预测的恒等散点。
- `figures/fig2_residual_distribution.png`：H1 残差分布（合并直方图 + 按文件小提琴）。
- `figures/fig3_hypothesis_margins.png`：每文件最优 vs 次优中位残差与裕度。
- `figures/fig4_field_census_heatmap.png`：字段 × 文件有效填充率热图。

## 复现

```bash
PYTHONPATH=src python -m coord_consistency.cli
```
