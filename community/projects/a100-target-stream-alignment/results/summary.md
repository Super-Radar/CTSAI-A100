# a100-target-stream-alignment 关键结论

分析对象：`RadarTools/RadarTools_Release/data/` 下 20 组同名 `<ts>_Tatget_CTLRR-300-V2_1_{RawTarget,TraTarget}.csv` 会话对（解析器与指标由本项目 `src/stream_align/` 提供，全部产物可由 `cli.py` 重跑生成）。

## 总体结论

1. **20/20 个会话在 tol=0 ms 即可把全部 Raw 帧精确配对到 Tra 帧**（方法 B/C 的 Raw 侧覆盖率均为 1.0；最佳 tol = 0 ms）。两流时间戳来自同一记录时钟，配对不依赖任何容差假设。

2. **2025-11-20 批次（10 个会话）两流帧数完全相等、逐帧同一毫秒**：Raw 与 Tra 是同一采集周期成对输出的，FrameNb 配对（方法 A）也严格成立。

3. **2026-05 批次（10 个会话）Tra 帧数比 Raw 多 2~453 帧**（Tra 侧配对覆盖率 0.800~0.941）：Tra 流存在大量没有同刻 Raw 帧的额外输出，这是可配对边界所在，而不是时间抖动——把 tol 从 0 ms 放宽到 50 ms 配对数不增加（见 fig2）。

4. **不要用 FrameNb 盲合并**：方法 A 在 2026-05 批次仍会给出 14~2456 个“高数量”配对，但残差中位数 -301 ms（负值 = 同一 FrameNb 的 Tra 帧时刻更早）、最大 61824 ms（跨秒级），属于系统性错配；两流 FrameNb 语义未经官方文档证实，跨批次不可当作公共帧号。

5. 数据质量：20 组会话共 0 次 FrameNb 重置/回退、0 帧墙钟与 epoch 不一致、0 个流内重复时间戳（解析器的重置分段与一致性检查在这些数据上未触发，相关行为由合成数据测试覆盖）。

## 会话可配对性

- 全部 20 个会话 Raw 侧可稳定配对（覆盖率 1.0，残差全为 0 ms）。
- Tra 侧覆盖率最低的会话：`2026_05_22_13_41_24_Tatget_CTLRR-300-V2_1`（0.800，4 个未匹配 Tra 帧）；未匹配 Tra 帧在整段时间轴上均匀出现（相对位置均值约 0.48），到最近 Raw 帧的中位距离约 41 ms，即它们落在 Raw 输出间隔的缝隙里，是额外输出而非坏点。

- 未参与配对：`2025_11_20_11_33_31_Tatget_CTLRR-300-V2_1_RawTarget`（只有 RawTarget、无同名 TraTarget）。

- 辅助单流演示：`Python雷达可视化工具示例/负30度_..._RawTarget.csv` 解析出 2463 帧、1 个分段、0 次重置（不参与双流配对）。

## 产物索引

- `session_summary.csv`：全部会话指标表

- `frame_pairs_2025_11_20_18_41_24.csv`、`frame_pairs_2026_05_20_16_24_32.csv`：两个代表会话的完整配对表

- `data_sha256.txt`：40 个配对输入文件的 SHA-256

- `figures/fig1_timeband_2026_05_20_16_24_32.png`：双流时间带

- `figures/fig2_tol_coverage.png`：tol-覆盖率曲线

- `figures/fig3_residual_hist.png`：匹配残差直方图（A 对比 C）

- `figures/fig4_coverage_by_method.png`：会话级覆盖率汇总


> 局限：时间戳可能来自上位机记录时钟；Raw/Tra 的 |Δt| 不能解释为固件处理延迟；两流 FrameNb 语义未获官方证实；本分析为离线复现，未做硬件实测。
