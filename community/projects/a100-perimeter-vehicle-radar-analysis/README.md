# CTSAI-A100 车辆周界感知实测与 LFMCW 雷达综合仿真



本项目基于 **CTSAI-A100**，针对车辆周界感知场景开展 ADC 实测数据分析与 MATLAB LFMCW 雷达综合仿真。



项目包含三类 CTSAI-A100 实测场景：



* 空环境

* 车辆靠近雷达

* 车辆远离雷达



在实测数据基础上完成 ADC 数据解析、Range FFT、Range-Doppler / 2D-FFT 以及 MTI 静态杂波抑制分析。



同时使用 MATLAB 建立可控的 LFMCW 雷达仿真模型，生成发射 LFM 信号、车辆目标回波、差频信号和时频图，并进一步完成 Range-Doppler、MTI 以及不同雷达参数对结果影响的实验。



本项目对应 CTSAI-A100 Community Issue #13。



---



## 1. 项目目标



本项目主要完成以下内容：



1. 使用 CTSAI-A100 采集车辆周界场景 ADC 数据；

2. 分析空环境、车辆靠近和车辆远离场景；

3. 对实测 ADC 数据进行 Range FFT 和 Range-Doppler / 2D-FFT 处理；

4. 对比 MTI 处理前后的静态/近静态杂波特征；

5. 使用 MATLAB 建立 LFMCW 雷达综合仿真；

6. 仿真发射 LFM chirp、目标回波和差频信号；

7. 生成 LFMCW 时频图；

8. 仿真运动目标和静态杂波的 Range-Doppler 响应；

9. 分析 LFM 频宽、时宽、PRF、目标数量、目标速度和 MTI 阶数的影响；

10. 对 MATLAB 仿真结果和 CTSAI-A100 实测结果进行对比，并说明两者之间的差异与实验边界。



---



## 2. 开发环境



* Radar：CTSAI-A100

* Operating System：Windows

* MATLAB：R2024b

* 数据类型：CTSAI-A100 ADC TXT

* Profile：Pf0

* Receive channels：Rx0-Rx3

* Acquisition mode：ALL channel



主要实测雷达参数：



| Parameter          |    Value |

| ------------------ | -------: |

| Start frequency    | 76.3 GHz |

| FMCW bandwidth     |  300 MHz |

| Chirp ramp-up time |    43 us |

| Chirp period       |    48 us |

| ADC frequency      |   25 MHz |

| Samples per chirp  |     1024 |

| Chirps per frame   |      256 |

| Range FFT size     |     1024 |

| Doppler FFT size   |      256 |

| TX mode            |     SISO |



根据当前配置计算：



* 距离分辨率约：`0.5245 m/bin`

* 速度分辨率约：`0.1599 m/s/bin`

* 当前处理对应的无模糊速度范围约：`±20.46 m/s`



---



## 3. 项目目录



```text

community/projects/a100-perimeter-vehicle-radar-analysis/

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

│       ├── empty/

│       ├── vehicle_approaching/

│       └── vehicle_receding/

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



## 4. 实测场景与数据



本项目包含三种车辆周界实验条件。



### 4.1 空环境



空环境采集：



* 1 个 acquisition run

* Rx0-Rx3

* 共 4 个 ADC TXT 文件



### 4.2 车辆靠近



车辆靠近场景采集：



* 5 个 acquisition runs

* 每个 run 包含 Rx0-Rx3

* 共 20 个 ADC TXT 文件



### 4.3 车辆远离



车辆远离场景采集：



* 5 个 acquisition runs

* 每个 run 包含 Rx0-Rx3

* 共 20 个 ADC TXT 文件



因此完整实测数据共包含：



```text

4 + 20 + 20 = 44 个 ADC TXT 文件

```



数据目录：



```text

data/measured/

├── empty/

│   └── run_001_Pf0_Rx0~Rx3.txt

│

├── vehicle_approaching/

│   ├── run_001_Pf0_Rx0~Rx3.txt

│   ├── ...

│   └── run_005_Pf0_Rx0~Rx3.txt

│

└── vehicle_receding/

    ├── run_001_Pf0_Rx0~Rx3.txt

    ├── ...

    └── run_005_Pf0_Rx0~Rx3.txt

```



详细采集说明见：



`docs/SCENE_AND_ACQUISITION.md`



---



## 5. 数据采集限制



本项目使用 ALL channel 方式采集 ADC 数据。



需要特别说明的是，当前数据采集和保存速度相对较慢，单个 RX 的一帧数据采集和保存大约需要十几秒。



因此：



* 车辆靠近场景中的 5 个 run 是多个离散采集样本；

* 车辆远离场景中的 5 个 run 同样是多个离散采集样本；

* `run_001` 至 `run_005` 不应被解释为严格连续、等时间间隔的车辆运动轨迹；

* Rx0-Rx3 不被假设为运动车辆完全同一时刻的严格同步阵列快照。



因此，本项目不会使用这 5 个 run 重建精确连续车辆轨迹，也不会基于这批车辆数据声明严格的四通道相干角度估计结果。



连续、可控的目标距离和目标速度变化主要通过 MATLAB LFMCW 仿真研究。



---



## 6. CTSAI-A100 ADC 数据解析



项目提供 CTSAI-A100 ADC TXT 数据读取和解包代码。



实测数据检查结果：



* 44 个 TXT 文件均可以正常读取；

* Rx index 均为 0、1、2、3；

* Samples per chirp 均为 1024；

* Chirp count 均为 256；

* ADC payload 完整。



每个数据文件在标准 ADC payload 后还存在 1 个 trailer 数值。



项目解析器将该数值作为 trailer 忽略，不影响 ADC payload 解包。



---



## 7. 实测信号处理



实测数据处理流程：



```text

CTSAI-A100 ADC TXT

        ↓

TXT parsing

        ↓

Packed uint32 ADC unpacking

        ↓

Fast-time DC removal

        ↓

Range FFT

        ↓

Range-slow-time data

        ↓

MTI / clutter suppression

        ↓

Doppler FFT

        ↓

Range-Doppler Map

```



实测数据中的第一维为 fast-time ADC samples，第二维为一帧内部的 slow-time chirps。



因此 Doppler FFT 和 MTI 均在一帧内部的 256 个 chirps 上完成，而不是在不同 acquisition run 之间进行。



---



## 8. 实测 Range FFT



项目将 ADC 数据沿 fast-time 方向进行 FFT，获得距离维结果。



物理距离轴由实际 CTSAI-A100 Pf0 配置计算。



代表性结果保存于：



```text

results/measured/empty/range_spectrum_physical.png

results/measured/vehicle_approaching/range_spectrum_physical.png

results/measured/vehicle_receding/range_spectrum_physical.png

```



---



## 9. 实测 Range-Doppler / 2D-FFT



完成 Range FFT 后，在一帧内部的 slow-time chirp 方向进行 Doppler FFT，形成 Range-Doppler Map。



代表性结果：



```text

results/measured/empty/range_doppler_before_mti_physical.png



results/measured/vehicle_approaching/

    range_doppler_before_mti_physical.png



results/measured/vehicle_receding/

    range_doppler_before_mti_physical.png

```



图中使用实际配置计算：



* Range：m

* Radial Velocity：m/s



---



## 10. 实测 MTI



项目使用二脉冲和三脉冲差分型 MTI。



MTI2：



```text

y[n] = x[n] - x[n-1]

```



MTI3：



```text

y[n] = x[n] - 2x[n-1] + x[n-2]

```



代表性实测结果：



```text

range_doppler_before_mti_physical.png

range_doppler_after_mti2_physical.png

range_doppler_after_mti3_physical.png

```



为了评价近零 Doppler 静态/近静态分量的变化，本项目选取约：



```text

-0.32 m/s ~ +0.32 m/s

```



作为近零 Doppler 分析窗口。



三个代表性 Rx0 样本得到：



| Scene               |     MTI2 |     MTI3 |

| ------------------- | -------: | -------: |

| Empty               | 42.90 dB | 68.37 dB |

| Vehicle approaching | 43.68 dB | 69.76 dB |

| Vehicle receding    | 42.23 dB | 67.70 dB |



结果表明，在当前实验定义的近零 Doppler 分析窗口内，MTI2 和 MTI3 均能够明显降低近零 Doppler 能量，其中 MTI3 的作用更强。



这些数值用于描述本项目实测样本的处理结果，不作为 CTSAI-A100 产品统一性能指标。



汇总文件：



`results/measured/mti_zero_doppler_summary.csv`



---



## 11. MATLAB LFMCW 综合仿真



除 CTSAI-A100 实测数据外，本项目还建立了独立的 MATLAB LFMCW 雷达仿真模型。



默认仿真目标：



* Initial range：30 m

* Radial velocity：-5 m/s



仿真速度约定：



```text

v < 0 : approaching

v > 0 : receding

```



仿真包含：



```text

LFMCW transmitted chirp

        ↓

Propagation delay

        ↓

Moving-target Doppler

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



## 12. LFMCW 发射信号、回波和差频信号



项目生成以下结果：



### 发射 LFM 信号



`results/simulation/lfmcw_tx_waveform.png`



### 运动目标回波



`results/simulation/lfmcw_target_echo.png`



### 差频信号



`results/simulation/lfmcw_beat_signal.png`



### 时频图



`results/simulation/lfmcw_tx_spectrogram.png`



时频图用于展示 LFM chirp 的瞬时频率随时间变化特征。



---



## 13. 仿真 Range-Doppler 和 MTI



仿真场景包含：



* 运动目标；

* 多个静态反射体；

* 复高斯噪声。



生成：



```text

results/simulation/simulation_range_doppler_before_mti.png

results/simulation/simulation_range_doppler_after_mti.png

```



未经 MTI 时，静态反射体主要集中在零 Doppler 附近。



经过 MTI 后，零 Doppler 附近的静态背景受到明显抑制，而非零 Doppler 的运动目标响应能够得到保留。



---



## 14. 参数影响实验



项目进一步分析多个关键 LFMCW 和目标参数。



结果位于：



```text

results/simulation/parameter_study/

```



### 14.1 LFM 频宽



比较：



* 150 MHz

* 300 MHz

* 450 MHz



结果：



`bandwidth_study.png`



带宽增加时，理论距离分辨能力提高，相邻距离目标更容易被区分。



### 14.2 LFM 时宽



改变 chirp duration，同时保持 bandwidth 不变。

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



时宽变化会改变 chirp slope，因此相同目标距离所对应的 beat frequency 发生变化。



### 14.3 Chirp Period / PRF



比较不同 chirp period。



结果：



`prf_study.png`



Chirp period 改变 slow-time 采样率和 PRF，因此影响 Doppler 采样和无模糊速度范围。



### 14.4 目标数量



分别模拟：



* 1 个目标

* 2 个目标

* 3 个目标



结果：



`target_number_study.png`



目标数量增加时，可以在 Range-Doppler 平面观察多个不同距离和径向速度位置的目标响应。



### 14.5 目标速度



比较：



* -2 m/s

* -5 m/s

* -10 m/s



结果：



`target_velocity_study.png`



目标径向速度变化会导致目标沿 Doppler / velocity 维移动。



### 14.6 MTI 阶数



比较：



* Before MTI

* MTI2

* MTI3



结果：



`mti_order_study.png`



差分型 MTI 能够抑制零 Doppler 附近的静态杂波。



MTI3 对近零 Doppler 的抑制更加明显，同时滤波作用也更加激进。



详细说明：



`docs/LFMCW_SIMULATION.md`



---



## 15. 仿真与实测结果对比



MATLAB 仿真与 CTSAI-A100 实测结果主要表现出以下共同特征：



### 静态背景



仿真中的静态反射体集中在零 Doppler 附近。



实测数据中同样存在明显的静态和近静态背景分量。



### MTI



仿真和实测均显示：



```text

Before MTI

    ↓

MTI2 / MTI3

    ↓

Zero / near-zero Doppler energy decreases

```



说明差分型 MTI 对静态或近静态背景具有明显抑制作用。



### 运动目标



仿真车辆目标具有已知距离和已知径向速度，因此 Range-Doppler 响应较为理想。



真实车辆属于扩展目标，可能包含：



* 车身散射；

* 车轮散射；

* 玻璃和金属结构散射；

* 地面反射；

* 环境杂波；

* 多径；

* 系统噪声。



因此实测 Range-Doppler Map 比理想仿真更加复杂。



详细分析：



`docs/SIMULATION_VS_MEASUREMENT.md`



---



## 16. 如何运行



### 16.1 准备 MATLAB



进入本项目目录：



```matlab

cd('community/projects/a100-perimeter-vehicle-radar-analysis')

```



如果 MATLAB 当前工作目录已经是项目目录，则无需再次设置路径。



### 16.2 统一入口



运行：



```matlab

main

```



程序会依次执行：



```text

Part 1

Measured CTSAI-A100 ADC analysis



Part 2

MATLAB LFMCW comprehensive simulation



Part 3

Radar parameter studies

```



全部完成后显示：



```text

All Issue #13 experiments completed successfully.

```



---



## 17. 单独运行实验



### 检查实测 ADC 数据



```matlab

run('tests/inspect_measured_dataset.m')

```



### 检查物理 Range / Velocity 坐标



```matlab

run('tests/check_measured_physical_axes.m')

```



### 实测代表性数据处理



```matlab

run('tests/run_measured_physical_analysis.m')

```



### LFMCW 综合仿真



```matlab

run('tests/run_lfmcw_simulation.m')

```



### 参数影响实验



```matlab

run('tests/run_parameter_studies.m')

```



---



## 18. 实验结果目录



实测结果：



```text

results/measured/

```



LFMCW 仿真结果：



```text

results/simulation/

```



参数实验：



```text

results/simulation/parameter_study/

```



所有结果均可通过 `main.m` 重新生成。



---



## 19. 实验边界



本项目需要明确区分：



1. MATLAB 理想化仿真结果；

2. CTSAI-A100 实际 ADC 数据分析结果；

3. CTSAI-A100 产品本身的最终检测性能。



MATLAB 仿真使用可控参数和简化目标模型，适合研究 LFMCW 雷达原理以及参数变化规律。



实测数据包含真实目标、真实环境杂波、多径、噪声以及数据采集流程带来的影响。



因此本项目不会将 MATLAB 理想仿真结果直接解释为 CTSAI-A100 产品实际性能指标。



同时，由于本次 ALL channel 车辆数据采集速度较慢，5 个车辆 run 只作为离散实验样本，不解释为完整连续车辆轨迹。



---



## 20. 相关文档



* `docs/SCENE_AND_ACQUISITION.md`



  * 实验场景、数据组织、采集方式及采集限制



* `docs/LFMCW_SIMULATION.md`



  * LFMCW 仿真、2D-FFT、MTI 和参数实验



* `docs/SIMULATION_VS_MEASUREMENT.md`



  * MATLAB 仿真和 CTSAI-A100 实测结果对比



* `data/README.md`



  * ADC 数据目录及使用说明



---



## 21. 作者



* Author / Team：DwHz

* GitHub：@DwHz



---



## 22. Issue



本 Community Project 基于 CTSAI-A100 Issue #13 完成。



项目内容包括：



* 车辆周界场景实测数据采集；

* CTSAI-A100 ADC 信号处理；

* MATLAB LFMCW 综合仿真；

* 时频分析；

* Range-Doppler / 2D-FFT；

* MTI；

* 参数影响分析；

* 仿真与实测结果比较。



项目代码、数据、文档和实验结果全部集中在本独立 Community Project 目录中。
