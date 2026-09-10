---
doc_id: "2064911496346779648"
title: "ADC数据采集"
parent_id: "2064910734866694144"
sort: 3
content_type: 1
---


ADC 数据是雷达信号处理链路中的底层数据，适合用于 FFT、滤波、检测、微多普勒分析、特征提取等研究。

### 启动 ADC 采集

ADC 数据采集通过 RadarTools 完成。

操作流程：

1. 连接雷达并完成上位机配置。
2. 点击 `Start`，确认雷达数据流正常输出。
3. 在 RadarTools 中点击 `ADC采集` 按钮。
4. 进入 ADC 采集界面。
5. 按实验需求配置采集参数。
6. 点击开始采集。
7. 等待采集完成后，检查输出文件。

###  ADC 采集参数

ADC 采集界面中通常包含以下配置项：

| 配置项       | 说明           |
| --------- | ------------ |
| FrameType | 波形类型 / 帧类型   |
| RxCh      | 接收通道选择       |
| Sample    | 每 chirp 采样点数 |
| Chirp_N   | chirp 数量     |
| 文件名       | 当前采集文件名称     |
| 目标目录      | ADC 数据保存目录   |
| DataType  | 数据类型         |

不同波形对应的 ADC 数据需要使用对应配置文件进行后续解析。

### ADC 数据保存目录

ADC 数据采集完成后，数据文件默认保存至：

```text
RadarTools/RadarTools_Release/adcData/
```

采集完成后，请确认该目录下已生成对应数据文件。RadarTools 版本不同，
输出可能采用会话子目录或旧版平铺目录。

当前仓库同时包含以下两种格式：

```text
RadarTools/RadarTools_Release/adcData/
├── adc_test_20260707154459626/
│   ├── run_001_Pf0_Rx0.txt
│   ├── run_001_Pf0_Rx1.txt
│   ├── run_001_Pf0_Rx2.txt
│   └── run_001_Pf0_Rx3.txt
├── adc_test_20260522110324_Pf0_Rx0.txt
├── adc_test_20260522110332_Pf0_Rx1.txt
├── adc_test_20260522110341_Pf0_Rx2.txt
└── adc_test_20260522110350_Pf0_Rx3.txt
```

文件名中的 `Pf0` 表示采集时使用的波形配置（profile），`Rx0`～`Rx3` 表示接收通道。
后续分析时，应选择同一会话、同一波形配置的文件，并保留实际文件名。
