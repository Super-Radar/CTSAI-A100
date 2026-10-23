# CTSAI-A100 室内 / 室外场景实测与 LFMCW 雷达综合仿真



本项目基于 **CTSAI-A100**，针对室内与室外环境开展 ADC 实测数据分析，并结合 MATLAB 建立 LFMCW 雷达综合仿真。



项目主要包含：



* CTSAI-A100 ADC TXT 数据解析

* Range FFT

* Range-Doppler / 2D-FFT

* MTI 静态及近静态杂波抑制

* 室内 / 室外代表场景结果展示

* MATLAB LFMCW 发射信号仿真

* 运动目标回波与差频信号

* LFM 时频分析

* 仿真 Range-Doppler

* MTI 前后对比

* LFM 频宽、时宽、PRF、目标数量、目标速度和 MTI 阶数参数实验

* MATLAB 仿真与 CTSAI-A100 实测结果对比



本项目对应 CTSAI-A100 Community Issue #8。



---
## 1. 项目目标



本项目通过真实 CTSAI-A100 ADC 数据和 MATLAB 理想化雷达仿真完成以下实验：



1. 采集并解析室内、室外场景 ADC 数据；

2. 对实测 ADC 数据进行 Range FFT；

3. 进行 Range-Doppler / 2D-FFT 处理；

4. 对比 MTI 处理前后的静态背景；

5. 建立 MATLAB LFMCW 综合仿真；

6. 生成发射 LFM chirp；

7. 模拟目标传播时延及 Doppler；

8. 生成目标回波和差频信号；

9. 生成 LFM 时频图；

10. 研究 LFM 频宽、时宽、PRF、目标数量、目标速度及 MTI 阶数变化；

11. 对理想仿真和真实室内 / 室外实测结果进行比较。



---
## 2. 开发环境



* Radar：CTSAI-A100

* Operating System：Windows

* MATLAB：2024b

* Data：ADC TXT

* Profile：Pf0

* Receive channels：Rx0-Rx3



当前项目使用的主要 Pf0 参数：



| Parameter         |    Value |

| ----------------- | -------: |

| Start frequency   | 76.3 GHz |

| FMCW bandwidth    |  300 MHz |

| Chirp ramp-up     |    43 us |

| Chirp period      |    48 us |

| ADC frequency     |   25 MHz |

| Samples per chirp |     1024 |

| Chirps per frame  |      256 |

| Range FFT size    |     1024 |

| Doppler FFT size  |      256 |

| TX mode           |     SISO |



根据当前配置计算：



* Range resolution：约 `0.5245 m/bin`

* Velocity resolution：约 `0.1599 m/s/bin`

* 当前处理对应的无模糊速度范围：约 `±20.46 m/s`



---
## 3. 项目目录



```text

community/projects/a100-indoor-outdoor-radar-analysis/

├── README.md

├── main.m

│

├── config/

│   ├── sensor_config_init0.hxx

│   └── simulation_config.m

│

├── data/

│   ├── README.md

│   └── measured/

│       ├── indoor_case_01/

│       ├── indoor_case_02/

│       ├── indoor_case_03/

│       └── outdoor_case_01/

│

├── functions/

│   ├── common/

│   ├── measured/

│   └── simulation/

│

├── docs/

│   ├── SCENE_AND_ACQUISITION.md

│   ├── LFMCW_SIMULATION.md

│   └── SIMULATION_VS_MEASUREMENT.md

│

├── results/

│   ├── measured/

│   └── simulation/

│       └── parameter_study/

│

└── tests/

```



---
## 4. 实测数据

完整实验共包含四组实测数据：

| 数据组               | 场景说明                               |
| ----------------- | ---------------------------------- |
| `indoor_case_01`  | 室内场景，被试者在雷达观测区域内左右移动               |
| `indoor_case_02`  | 室内场景，被试者坐在椅子上；采集开始阶段有小幅动作，后半段保持静止  |
| `indoor_case_03`  | 室内空旷静态场景，全程未人为设置运动目标               |
| `outdoor_case_01` | 室外自然环境场景，用于观察室外背景杂波、环境反射及可能存在的运动响应 |

各场景的现场照片、目标状态、运动方式以及实验信息边界详见：

`docs/SCENE_AND_ACQUISITION.md`



### 完整数据规模



每个场景实际采集：



* 20 个 run

* 每个 run 包含 Rx0-Rx3

* 80 个 ADC TXT 文件 / scene



完整实验：



```text

4 scenes × 20 runs × 4 RX

= 320 ADC TXT files

```



完整原始 ADC TXT 数据总量约为：



```text

379 MB

```



---
## 5. 仓库中的代表数据



考虑到 320 个原始 ADC TXT 文件总体积约为 379 MB，本 Community Project 不直接提交完整原始数据。



仓库为每个场景提供 `run_001`：



```text

data/measured/

├── indoor_case_01/

│   └── run_001_Pf0_Rx0~Rx3.txt

├── indoor_case_02/

│   └── run_001_Pf0_Rx0~Rx3.txt

├── indoor_case_03/

│   └── run_001_Pf0_Rx0~Rx3.txt

└── outdoor_case_01/

    └── run_001_Pf0_Rx0~Rx3.txt

```



因此仓库包含：



```text

4 scenes × 1 representative run × 4 RX

= 16 ADC TXT files

```



这些代表数据用于复现本项目主要 MATLAB 实测信号处理流程。



完整的 20-run 数据保留在原始实验数据存储位置，可用于进一步的本地批量分析。



---
## 6. ADC 数据检查



仓库中的 16 个代表 ADC TXT 文件已完成解析测试。



四个场景均满足：



* Rx index：`0, 1, 2, 3`

* Samples per chirp：`1024`

* Chirps per frame：`256`

* ADC payload 可完整解析



每个 TXT 文件在标准 ADC payload 后还存在 1 个 trailer 数值。



当前解析器将其作为尾部数据忽略，不影响 ADC payload 的正常读取与解包。



---
## 7. 实测处理流程



项目对四个场景分别选择 `run_001_Pf0_Rx0.txt` 作为代表性实测样本。



处理流程：



```text

CTSAI-A100 ADC TXT

        ↓

ADC TXT parsing

        ↓

Packed uint32 unpacking

        ↓

Fast-time DC removal

        ↓

Range FFT

        ↓

Range-slow-time data

        ↓

MTI2

        ↓

Doppler FFT

        ↓

Range-Doppler Map

```



Range FFT 使用一帧中的 fast-time samples。



Doppler FFT 和 MTI 使用同一帧内部的 256 个 slow-time chirps。



---
## 8. 实测 Range Spectrum



四个代表场景均生成 Range Spectrum：



```text

results/measured/indoor_case_01/range_spectrum.png

results/measured/indoor_case_02/range_spectrum.png

results/measured/indoor_case_03/range_spectrum.png

results/measured/outdoor_case_01/range_spectrum.png

```



距离轴使用实际雷达配置转换为 `Range (m)`。



---
## 9. 实测 Range-Doppler / 2D-FFT



四个代表场景均生成：



```text

range_doppler_before_mti.png

```



结果使用：



* Range：m

* Radial Velocity：m/s



作为物理坐标。



结果分别位于：



```text

results/measured/indoor_case_01/

results/measured/indoor_case_02/

results/measured/indoor_case_03/

results/measured/outdoor_case_01/

```



---
## 10. 实测 MTI



项目使用二脉冲差分型 MTI：



```text

y[n] = x[n] - x[n-1]

```



每个代表场景均生成：



```text

range_doppler_before_mti.png

range_doppler_after_mti2.png

```



Before MTI 和 After MTI2 使用相同的功率参考值和显示动态范围，以便直接观察静态和近静态背景在 MTI 前后的变化。



MTI 主要用于削弱 slow-time 中变化较小的静态及近静态分量。



---
## 11. 室内 / 室外场景



本项目包含三组室内数据以及一组室外数据。



室内环境可能包含墙面、固定结构和其他物体形成的复杂反射与多径。



室外环境具有不同的传播条件和背景杂波组成。



因此不同场景中的 Range Spectrum 和 Range-Doppler 分布可能存在明显差异。



本项目通过代表性实测数据展示这些场景中的雷达响应特征，不将单个代表样本直接解释为 CTSAI-A100 的统一产品性能指标。



详细说明：



`docs/SCENE_AND_ACQUISITION.md`



---
## 12. MATLAB LFMCW 综合仿真



除实测数据外，本项目建立了独立的 MATLAB LFMCW 雷达综合仿真。



默认参数：



* Carrier frequency：76.3 GHz

* Bandwidth：300 MHz

* Chirp duration：43 us

* Chirp period：48 us

* Target initial range：30 m

* Target radial velocity：-5 m/s



仿真中的速度符号约定：



```text

v < 0 : approaching

v > 0 : receding

```



仿真流程：



```text

LFMCW transmitted chirp

        ↓

Propagation delay

        ↓

Doppler phase

        ↓

Target echo

        ↓

Dechirp

        ↓

Beat signal

        ↓

Range FFT

        ↓

Doppler FFT

        ↓

Range-Doppler Map

```



---
## 13. LFMCW 发射信号、回波和差频信号



生成：



### 发射信号



`results/simulation/lfmcw_tx_waveform.png`



### 目标回波



`results/simulation/lfmcw_target_echo.png`



### 差频信号



`results/simulation/lfmcw_beat_signal.png`



### 时频图



`results/simulation/lfmcw_tx_spectrogram.png`



时频图展示 LFM chirp 的频率随时间变化特征。



---
## 14. 仿真 Range-Doppler 与 MTI



仿真场景包含：



* 运动目标

* 多个静态反射体

* 复高斯噪声



结果：



```text

results/simulation/simulation_range_doppler_before_mti.png

results/simulation/simulation_range_doppler_after_mti.png

```



未进行 MTI 时，人工设置的静态反射体主要集中在零 Doppler 附近。



经过 MTI 后，静态及近静态背景受到明显抑制。



---
## 15. 参数影响实验



参数实验位于：



```text

results/simulation/parameter_study/

```



### LFM 频宽



比较：



* 150 MHz

* 300 MHz

* 450 MHz



结果：



`bandwidth_study.png`



带宽增加时，理论距离分辨能力提高。



### LFM 时宽



保持 bandwidth 不变并改变 chirp duration。

采样率固定为 25 MHz，样本数使用 `floor(sample_rate * chirp_duration)`：
30、36、43 us 分别对应 750、900、1075 点。每个采样时刻必须位于半开区间
`[0, chirp_duration)`；帧仿真会拒绝末采样点位于 chirp 终点或之后的配置。
默认仿真仍使用 43 us、1024 点；这里只改变时宽扫描的采样窗口。

原时宽扫描固定 1024 点，30/36 us 时分别有 274/124 点超出 ramp。
修正后 FFT 频率间隔随样本数变化；这不是修改实际雷达采集参数。
帧生成器仍是解析差频教学模型，不模拟传播时延造成的 chirp 起始无效区。

独立数值回归（不需要绘图或实测数据）：

```matlab
addpath('tests');
test_simulation_window;
```

本次更新的时宽图使用修正后的 `simulate_lfmcw_frame` 在 GNU Octave 6.4.0 中
生成的单目标 FFT 谱，由 Matplotlib 3.10.9 绘制。验证覆盖帧函数与上述等价时宽
实验；未执行完整 MATLAB `run_parameter_studies.m` 或交互图形流程。现有 MATLAB
脚本可以按修正后的采样配置重新生成该图。





结果：



`chirp_duration_study.png`



时宽变化改变 chirp slope，因此同一距离目标产生的 beat frequency 发生变化。



### Chirp Period / PRF



结果：



`prf_study.png`



Chirp period 改变 slow-time 采样率和 PRF，因此影响 Doppler 采样及无模糊速度范围。



### 目标数量



模拟：



* 1 target

* 2 targets

* 3 targets



结果：



`target_number_study.png`



多个目标在不同的距离和径向速度位置形成多个 Range-Doppler 响应。



### 目标速度



比较：



* -2 m/s

* -5 m/s

* -10 m/s



结果：



`target_velocity_study.png`



目标径向速度变化使目标响应沿 velocity 维移动。



### MTI 阶数



比较：



* Before MTI

* MTI2

* MTI3



结果：



`mti_order_study.png`



高阶差分 MTI 对近零 Doppler 分量具有更强的抑制作用，同时滤波作用也更加明显。



详细说明：



`docs/LFMCW_SIMULATION.md`



---
## 16. MATLAB 仿真与 CTSAI-A100 实测对比



MATLAB 仿真使用：



* 理想化目标模型

* 已知距离

* 已知径向速度

* 可控静态杂波

* 可控噪声



真实 CTSAI-A100 数据则包含：



* 真实环境固定反射

* 多径传播

* 实际系统噪声

* 不同散射表面

* 复杂室内 / 室外传播环境



因此仿真 Range-Doppler 图通常较为理想，而实际场景中的雷达响应更加复杂。



仿真主要用于解释 LFMCW 信号处理原理和参数变化规律。



实测主要用于展示真实 CTSAI-A100 ADC 数据中的场景特征。



详细分析：



`docs/SIMULATION_VS_MEASUREMENT.md`



---
## 17. 如何运行



进入项目目录：



```matlab

cd('community/projects/a100-indoor-outdoor-radar-analysis')

```



运行统一入口：



```matlab

main

```



程序依次执行：



```text

Part 1

Measured indoor/outdoor ADC analysis



Part 2

MATLAB LFMCW simulation



Part 3

Radar parameter studies

```



完整运行成功后输出：



```text

All Issue #8 experiments completed successfully.

```



---
## 18. 单独运行



### 数据检查



```matlab

run('tests/inspect_measured_dataset.m')

```



### 实测室内 / 室外分析



```matlab

run('tests/run_measured_scene_analysis.m')

```



### LFMCW 综合仿真



```matlab

run('tests/run_lfmcw_simulation.m')

```



### 参数实验



```matlab

run('tests/run_parameter_studies.m')

```



---
## 19. 结果目录



实测：



```text

results/measured/

```



LFMCW 仿真：



```text

results/simulation/

```



参数实验：



```text

results/simulation/parameter_study/

```



所有主要结果均可通过 `main.m` 重新生成。



---
## 20. 数据和结果使用边界



本项目明确区分：



1. MATLAB 理想化仿真结果；

2. CTSAI-A100 ADC 实测结果；

3. CTSAI-A100 产品实际最终能力。



MATLAB 仿真用于研究理想化 LFMCW 雷达规律。



实测数据用于展示真实室内和室外环境中的雷达数据特征。



因此不能将 MATLAB 理想仿真结果直接解释为 CTSAI-A100 产品实际检测性能。



完整实验虽然包含 320 个 ADC TXT 文件，但由于数据总体积约 379 MB，本仓库仅提供 16 个代表性 TXT 文件用于主要处理流程复现。



---
## 21. 文档



* `docs/SCENE_AND_ACQUISITION.md`



  * 场景、数据规模和数据采集说明



* `docs/LFMCW_SIMULATION.md`



  * LFMCW、2D-FFT、MTI 和参数实验



* `docs/SIMULATION_VS_MEASUREMENT.md`



  * MATLAB 仿真与 CTSAI-A100 实测结果对比



* `data/README.md`



  * 完整数据规模及仓库代表数据说明



---
## 22. 作者



* Author / Team：HuangTao

* GitHub：@Zzht999



---
## 23. Issue



本 Community Project 基于 CTSAI-A100 Issue #8 完成。



项目代码、代表性实测数据、实验结果及文档全部集中于：



`community/projects/a100-indoor-outdoor-radar-analysis/`



