# a100-log-buffer-budget — CAN-FD 日志突发性与离线消费缓冲预算

对 RadarTools 上位机随目标日志一并记录的 CAN-FD 总线日志（Vector ASC 格式）做**离线**突发性统计，
并用确定性队列模拟回答一个工程问题：**假设有一个按固定速率逐条消费这些日志报文的离线消费器，
它的内存队列需要多大的消费速率与容量才不丢报文？**

本项目只做离线统计与内存队列模拟：不解码任何协议语义、不推断总线真实占用率、
不涉及向总线发送报文或任何渗透/漏洞方向。

## 1. 目标

- 解析仓内 ASC 日志（含头/注释/空文件容忍；DLC 字段与实际字节 token 不符时以 token 计数为准）；
- 多尺度滑动窗口到达率包络（10 ms / 50 ms / 100 ms / 500 ms / 1 s，messages/s 与 bytes/s）；
- 同时间戳批次（burst）与按 ID 的流量占比统计；
- 确定性单服务台内存队列模拟：消费速率 r 扫描、容量 c 有限时的溢出丢弃，
  输出最大积压、溢出数与容量-溢出关系，给出量化的缓冲预算建议（全部以"假设的消费器"表述）。

## 2. 数据来源

目录：`RadarTools/RadarTools_Release/data/`（仓库官方附带资产）。
该目录共 56 个 `*_CanData.asc`，其中 **10 个含报文、共 201,123 条 CAN-FD 记录**
（其余 46 个为空文件或仅含文件头，解析器自动跳过）。

| 文件 | 报文数 | 时长 (s) | 均值 (msg/s) | 峰值@10ms (msg/s) | 最大同时间戳批次 |
|---|---:|---:|---:|---:|---:|
| 2026_05_20_15_47_18_Tatget__CanData.asc | 9,039 | 12.93 | 699.0 | 3,800 | 38 |
| 2026_05_20_15_47_40_Tatget__CanData.asc | 73,313 | 150.84 | 486.0 | 3,800 | 38 |
| 2026_05_20_16_24_32_Tatget__CanData.asc（代表文件） | 100,131 | 145.37 | 688.8 | 3,700 | 37 |
| 2026_05_22_13_41_24_Tatget__CanData.asc | 675 | 1.01 | 668.3 | 3,600 | 36 |
| 2026_05_22_13_41_37_Tatget__CanData.asc | 540 | 0.79 | 684.4 | 3,400 | 33 |
| 2026_05_22_13_41_39_Tatget__CanData.asc | 1,406 | 2.06 | 683.5 | 3,600 | 36 |
| 2026_05_22_13_41_43_Tatget__CanData.asc | 2,125 | 3.16 | 672.7 | 3,600 | 36 |
| 2026_05_22_16_21_05_Tatget__CanData.asc | 3,007 | 4.48 | 670.6 | 3,600 | 36 |
| 2026_05_22_16_31_18_Tatget__CanData.asc | 2,733 | 4.09 | 667.6 | 3,500 | 35 |
| 2026_05_25_15_05_16_Tatget__CanData.asc | 8,154 | 12.68 | 643.2 | 3,900 | 39 |

10 个文件合计 12,119,171 个日志载荷字节 token。SHA-256（与 `results/data_sha256.txt` 一致）：

```text
aee624bf2ca538cfefa64b043fea5249a4e1c8c61a7d2312ed1e2115db2945a5  2026_05_20_15_47_18_Tatget__CanData.asc
677b4b8e64ba2bf8e4d4af9c59f86f7845899d6b6fa6622dae6631d66dd00e49  2026_05_20_15_47_40_Tatget__CanData.asc
3ef9967457b86596a39a91b21fa643fd61174cb3cceaec44a1ed6d6cf74eb2f4  2026_05_20_16_24_32_Tatget__CanData.asc
bfe5e689e0757eb5cc24062e74a6c922dc0e1f4932f9b0c1254f6abde514e4cc  2026_05_22_13_41_24_Tatget__CanData.asc
3eb165128e7037f135cb4e0861bc1427def2cfed71577413a87acdec97cc9c00  2026_05_22_13_41_37_Tatget__CanData.asc
81a0ea708919ed4abca93c26a26114ad5cdf0f6365f6bd505df58665b1a08eef  2026_05_22_13_41_39_Tatget__CanData.asc
76111c24ed2d82c84e15d91376c8f092fcf64b613b6a8fb092c1713fbfe70f2f  2026_05_22_13_41_43_Tatget__CanData.asc
45f7102450926337cc9e1c2bc08452dd0b36a6a88ab31297b76e188214a1c5ff  2026_05_22_16_21_05_Tatget__CanData.asc
fec371a4a9b0e77470626cf921045c9520274d4946cd68bd9cbcc1115b697cf2  2026_05_22_16_31_18_Tatget__CanData.asc
6dc8d60c7fe89d738981a970fd96555ba63076d9b6184c3ae50b16c71bdb4ae4  2026_05_25_15_05_16_Tatget__CanData.asc
```

格式说明（实测）：报文行为
`<相对秒> CANFD <通道> <ID hex> Rx <BRS> <ESI> d <DLC> <数据字节 token...>`。
两点重要事实：**DLC 字段是长度代码，与行内实际字节 token 数不一致**（实测每个文件中
token 数恒等于 DLC 代码对应的载荷长度 +1，如 DLC=15 → 65 个 token），解析一律以实际
token 计数为准；**同一时间戳多条报文常见**（日志器把整簇报文记在同一时刻）。

> 声明：该数据是 RadarTools 上位机随目标日志记录的 CAN 总线日志，属于仓内官方附带资产，
> 但本项目的分析**不是 A100 硬件实测**，只是对既有日志文件的离线统计。

## 3. 环境

- Python 3.12；仅依赖标准库 + numpy + scipy + matplotlib + pytest（**不使用 pandas**）；
- 图形后端 matplotlib Agg（无显示环境亦可运行）；
- 随机种子固定 20261003（仅用于图 2 从最繁忙的 3 个 100 ms 窗口中选定展示片段；
  其余统计与模拟完全确定）。

## 4. 运行步骤

```bash
cd community/projects/a100-log-buffer-budget

# 测试（23 项，含 ASC 解析 / 解析解核对 / 守恒 / 单调性等）
python3 -m pytest tests/ -q

# 一键复现全部结果与图（约 10 s）
PYTHONPATH=src python3 -m buffer_budget.cli
```

CLI 会重建 `results/`（CSV、`data_sha256.txt`、`summary.md`）与 `figures/`（5 张图）。

## 5. 结果

详细数字见 `results/summary.md`（中文关键结论）与各 CSV；图在 `figures/`：

| 图 | 内容 |
|---|---|
| fig1_multiscale_rate_envelope.png | 代表文件五尺度到达率包络（log-y） |
| fig2_burst_segment_batch_stems.png | 突发片段：10 ms 窗到达率 + 同时间戳批次竖线 |
| fig3_backlog_vs_rate.png | 最大积压 vs 消费速率（10 文件曲线，log-log） |
| fig4_capacity_overflow_heatmap.png | 容量×速率溢出热图与零溢出边界（代表文件） |
| fig5_per_id_share.png | 按 ID 报文量占比（代表文件，0x601 占 20.5%） |

要点（代表文件 `2026_05_20_16_24_32`，均值到达率 688.8 msg/s）：

- **突发性**：1 s 尺度峰值/均值 ≈ 1.0（长时近平稳），10 ms 尺度突发倍数 **5.4x**
  （10 文件范围 5.0–7.8x）；最大同时间戳批次 **37** 条（平均 17.5 条/批）。
- **假设消费器（内存队列）**：r < 均值到达率时积压发散（r=0.8x 时日志结束积压 20,035 条）；
  r=1.00x 时零溢出最小容量 **143** 条（按 64B/条 ≈ 8.9 KiB）；r=1.05x 降到 **44** 条；
  r ≥ 1.35x 后最小容量恒为 **37** 条 = 最大同时间戳批次——消费速率足够快时，
  缓冲预算由**批次规模**而非平均流量决定。
- 上述全部针对"假设的离线消费器"，不是总线带宽利用率。

## 6. 局限

- 本项目度量的是**日志载荷吞吐与假设消费器的积压**，不是 CAN-FD 总线真实利用率；
- 不建模位填充、仲裁、驱动/内核环形缓冲等总线与系统层机制；
- ASC 时间戳分辨率受日志器限制（同簇报文共享同一时间戳，毫秒级），
  真实突发间隔可能比记录值更短，据此得到的峰值到达率是**下界**；
- 离线分析，未做硬件实测。

## 7. 作者

chennaji9

## 8. 与现有内容的差异

- 已合并的 5 个社区项目（`a100-indoor-cart-mti`、`a100-indoor-outdoor-radar-analysis`、
  `a100-matlab-adc-pipeline-validation`、`a100-matlab-adc-signal-processing`、
  `a100-perimeter-vehicle-radar-analysis`）均聚焦雷达信号处理/测量分析，
  **没有项目处理 ASC 日志的消费负载与缓冲预算**；
- 本项目与官方 RadarTools 上位机**无任何代码或配置修改**，只读取其日志数据，
  全部新增文件位于 `community/projects/a100-log-buffer-budget/` 内。
