# a100-target-stream-alignment：RawTarget / TraTarget 双流时间对齐与可配对边界

[CTSAI-A100](https://github.com/Super-Radar/CTSAI-A100) 社区项目。RadarTools 上位机会为同一次采集输出两份目标日志：`..._RawTarget.csv`（原始检测目标流）与 `..._TraTarget.csv`（跟踪目标流）。两份文件**帧数不同、不逐帧对应**（例如会话 `2026_05_20_16_24_32`：Raw 2456 帧、Tra 2909 帧），直接按行号或 FrameNb 盲合并会制造未经证实的对应关系。本项目回答三个问题：

1. 两流之间什么样的帧级对应是**可以被数据证实**的？
2. 配对成立的**边界**在哪里（容差、帧数差、时间分布）？
3. 什么情况下**不该**做配对？

## 方法

`src/stream_align/` 提供三种配对方法，全部满足"一对一、时间单调、跨分段不配对"三条硬约束：

| 方法 | 规则 | 角色 |
| --- | --- | --- |
| A. FrameNb 相等 | 同分段内 `(segment, FrameNb)` 键相等才配对 | "朴素合并"基线，用于暴露风险 |
| B. 时间戳精确相等 | `epoch_ms` 完全相同才配对 | 零假设检验：两流是否同刻输出 |
| C. 容差匹配 | 分段内双指针贪心扫描，`\|t_raw − t_tra\| ≤ tol` 才配对，一步前瞻优先更近邻 | 通用方法，tol 从 0 扫到 50 ms |

解析器（`parser.py`）从每个 CSV 帧中提取 `START` 行的 FrameNb、`---->` 行的墙钟时间、`heightOffset` 元数据行的第二个字段 `epoch_ms`（实测为 UTC+8 北京时间墙钟，与 `---->` 行逐帧一致）以及目标行数；FrameNb 重置/回退自动分段，墙钟与 epoch 的一致性逐帧校验。指标（`metrics.py`）输出覆盖率、歧义数、残差分布（中位数/p95/max）与未匹配帧的时间分布（相对会话位置、到另一流最近帧的 KD-tree 最近邻距离）。

## 数据来源

- 目录：`RadarTools/RadarTools_Release/data/`（仓库官方附带的上位机示例目标日志）。
- 文件名模式：`<YYYY_MM_DD_HH_MM_SS>_Tatget_CTLRR-300-V2_1_{RawTarget,TraTarget}.csv`，共 **20 组同名配对**；另有 1 个只有 Raw 无配对（`2025_11_20_11_33_31`）。
- 文件名中的 `CTLRR-300-V2_1` 为 RadarTools 日志标注的雷达型号，**这些数据是上位机示例目标日志，不是 A100 硬件实测**，结论仅对这批数据本身成立。
- 每帧结构：60 列表头 → `START,FrameNb:N,...` → `YYYY/MM/DD HH:MM:SS:mmm---->` → `heightOffset:0,<epoch_ms>,...` → 每目标一行。
- 40 个配对输入文件的 SHA-256 见 [`results/data_sha256.txt`](results/data_sha256.txt)。
- 辅助单流：`Python雷达可视化工具示例/负30度_Tatget_2026_03_10_15_34_39_CTLRR-300-V2_1_RawTarget.csv`（2463 帧，仅演示长单流解析，不参与配对）。

## 环境

- Python 3.12（开发与验证使用 3.12.3）
- 依赖仅：标准库 + numpy + scipy + matplotlib + pytest；**不使用 pandas**，CSV 帧结构由 `parser.py` 自写解析
- matplotlib 使用 Agg 后端，无显示环境也能出图

## 运行步骤

```bash
# 1) 测试（16 项，含合成双流与真实数据冒烟）
python -m pytest community/projects/a100-target-stream-alignment/tests/ -v

# 2) 一键复现全部结果（约数秒）
python community/projects/a100-target-stream-alignment/src/stream_align/cli.py
```

`cli.py` 自动定位仓库内数据目录（可用 `--data-dir` 覆盖），把全部产物写入 `results/`：`session_summary.csv`（20 会话全指标）、`frame_pairs_<会话>.csv`（2 个代表会话完整配对表）、`data_sha256.txt`、`summary.md` 与 `figures/` 下 4 张图（固定种子 20261003，可复现）。

## 结果

关键结论（完整版与数字见 [`results/summary.md`](results/summary.md)）：

1. **20/20 个会话在 tol = 0 ms 即可把全部 Raw 帧精确配对**（Raw 侧覆盖率 1.0，残差全 0 ms）。最佳 tol 就是 0 ms：放宽到 50 ms 配对数也不增加。
2. **2025-11-20 批次（10 会话）**两流帧数相等、逐帧同一毫秒——Raw/Tra 同周期成对输出，FrameNb 配对也严格成立。
3. **2026-05 批次（10 会话）Tra 比 Raw 多 2~453 帧**（Tra 侧覆盖率 0.800~0.941）。未匹配 Tra 帧到最近 Raw 帧的中位距离约 34~48 ms，均匀散布在整段时间轴上——它们是**落在 Raw 输出间隔缝隙里的额外跟踪输出**，这就是可配对边界；不是抖动，加大 tol 无济于事。
4. **FrameNb 盲合并是陷阱**：方法 A 在 2026-05 批次的配对"数量"看起来同样高，但残差中位数 -301 ms（负值 = 同帧号 Tra 更早）、最大 |Δt| 61824 ms（ pooled 中位数 3936 ms、p95 22006 ms），属于系统性错配（见 fig3 左右对比）。
5. 这批数据中未出现 FrameNb 重置/回退、墙钟-epoch 失配或流内重复时间戳；相关检测均实现并由合成数据测试覆盖。

| 图 | 内容 |
| --- | --- |
| `results/figures/fig1_timeband_2026_05_20_16_24_32.png` | 代表会话双流时间带：两行帧时刻散点 + 抽样匹配连线 + 未匹配 Tra 帧（橙 ×） |
| `results/figures/fig2_tol_coverage.png` | 20 会话的 tol-覆盖率曲线（0~50 ms，实线 Raw 侧 / 虚线 Tra 侧） |
| `results/figures/fig3_residual_hist.png` | 匹配残差直方图：方法 A 的秒级长尾 vs 方法 C 的 0 ms 尖峰 |
| `results/figures/fig4_coverage_by_method.png` | 会话级覆盖率汇总条形图（A/B/C 分组，Raw 侧与 Tra 侧） |

## 局限

- 时间戳（`---->` 行与 `epoch_ms`）可能来自上位机记录时钟，而非雷达固件时间。
- Raw/Tra 配对帧的 |Δt| **不能**解释为固件处理延迟——两流时间在本批数据中完全同刻，且时钟来源未知。
- 两流 FrameNb 的语义未获官方文档证实，跨批次不可当作公共帧号（结果 4 即反例）。
- 纯离线分析，未做硬件实测；结论只覆盖仓库附带示例数据。

## 与已有社区项目的关系

- 已合并的 5 个社区项目（`a100-indoor-cart-mti`、`a100-indoor-outdoor-radar-analysis`、`a100-matlab-adc-pipeline-validation`、`a100-matlab-adc-signal-processing`、`a100-perimeter-vehicle-radar-analysis`）分别处理 ADC/点云/MTI/围界等主题，**均不做 RawTarget/TraTarget 双流帧级时序对应**；本项目的双流解析-对齐-指标链路是新增维度。
- 本项目不修改官方 `radarViewer.py` 等任何既有文件，只新增 `community/projects/a100-target-stream-alignment/`。

## 作者

chennaji9
