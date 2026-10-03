# a100-adc-requantization：ADC 有损再量化的谱保真预算

本项目基于 **CTSAI-A100** 官方示例 ADC 数据，研究一个此前社区未覆盖的问题：
**当 16 位 ADC 存储样本被有损再量化到 k 位（k = 4..14）时，官方
Range-Doppler 谱链路能保住多少保真度，"安全下限"是多少位。**

全部为 Python 离线分析（numpy + scipy + matplotlib），固定满量程
int16 [-32768, 32767]，**不做任何逐文件自适应归一化**——降低字长带来的
损失必须如实显露，而不是被增益掩盖。

---

## 1. 项目目标

* 复刻官方 MATLAB 解包与谱链路（`load_adc_data.m` / `adc_mem_2_real_fixed.m`
  / `do_fft_2d.m`）的 Python 等价实现，并把"文件比目标多 1 个值"这类
  官方代码静默截断的行为改为**显式报告**；
* 在固定满量程下实现两种再量化律：**round**（round-half-away-from-zero）
  与 **trunc**（丢低位 = 补码算术右移 = floor），k 扫 4..14，另含 k=16 恒等；
* 建立谱保真预算指标：时域 RMSE、距离谱 / RD 谱 RMSE、预定义频带能量
  保持率、峰值幅度 / 位置漂移、双音弱音可见性；
* 用合成基准锚定理论（无裁剪区舍入误差 ≤ 半步长 Δ/2、量化噪声功率
  ≈ Δ²/12、trunc 误差有偏 +Δ/2、总 MSE 比 round 大 4 倍即 RMSE 大 2 倍），
  再对 8 个实测文件逐通道给出相对 16 位基线的变化曲线与安全下限。

## 2. 数据来源

官方示例数据（本仓库自带，未做任何修改）：

* 路径：`ADC数据采集/示例adc数据和结果/`
* 8 个通道文件：`adc_远波_20260512135827_f0_{0..3}_*.txt`（init0 远波，
  4 个 RX 通道）与 `adc_近波_20260512141857_f1_{0..3}_*.txt`（init1 近波，
  4 个 RX 通道）；每个文件 131073 个逗号分隔无符号整数值、无表头，
  比解包目标 131072 多 1 个、末值为 0（本工具链显式报告该多余值后截断）。
* 配置：`ADC数据采集/matlab_signal_processing_platform_231023_for_txt_A100/cfg/CTASI-A100配置/sensor_config_init0.hxx`
  （76.3 GHz、300 MHz、43 us ramp、48 us period、256 chirp、25 MHz ADC、
  rng_nfft 1024、vel_nfft 256，adc_sample_num=1020）与 `sensor_config_init1.hxx`
  （76.119 GHz、750 MHz、104.65 us ramp、108.5 us period、128 chirp、
  rng_nfft 2048、vel_nfft 128，adc_sample_num=2048 用满）。
* sha256（完整清单见 `results/data_sha256.txt`）：

| 文件 | sha256 |
|---|---|
| adc_远波_..._f0_0_13_58_36_748.txt | bba39d6f30c7c1530e85a2bf7b2509278fbcf03c4a025cb5978a3efdefdfb3a7 |
| adc_远波_..._f0_1_13_58_45_367.txt | 33d6b42ec6458934f0a2d66d547678e6604901b7fb26c7bfee44aba3a0d7448d |
| adc_远波_..._f0_2_13_58_53_977.txt | c58910a1705586758362c348f6cd347d44ea8ee7d54fe2771193bc23671c3a70 |
| adc_远波_..._f0_3_13_59_02_596.txt | 80ebc8a7f1735612b1f91292fefd083058acabc00980c1ed7a00739e5691226f |
| adc_近波_..._f1_0_14_19_05_965.txt | 36db22359f0bc902022ec26a07a1a6cb2ed941f66f75729c42d01bf56c4d273c |
| adc_近波_..._f1_1_14_19_14_570.txt | 4fb120427111cddaf220433bf85491a700633ca28e4b41d5ea20ee01707960ab |
| adc_近波_..._f1_2_14_19_23_170.txt | 05f3a38d9297f35c7972d84d8f9e300b269457b3e6cd00e5f51d3172ac6c306e |
| adc_近波_..._f1_3_14_19_31_788.txt | 134e89fb95ae8e2d7955163e888040ecf59fc3895c21841debc07fb51127e7bb |

同目录 2 张官方 JPG（远波 / 近波 4 通道非相干积累）仅作定性参考，
不作为本研究的真值。

### adc_sample_num 口径说明

`init_function/get_radar_paramete.m` 推导
`adc_sample_num = round((adc_sample_end - adc_sample_start) * adc_freq / dec_factor)`
（dec_factor=1 时无 +1），再截顶 rng_nfft。init0：
round((42.8-2)×25) = **1020**（IEEE-754 下 (42.8-2)×25 =
1019.9999...，round 后仍为 1020；历史上"1020 还是 1021"的歧义即源于
该浮点表示，MATLAB 与 NumPy 结论一致）；init1：round((104.4-2)×25) =
2560 > 2048 → 截为 **2048**（用满）。本研究按此口径实现并在
`src/requant/config.py` 中显式记录。

## 3. 环境

* WSL / Ubuntu，Python 3.12
* 依赖：numpy、scipy、matplotlib（Agg 后端）、pytest；**不使用 pandas**
* 参考虚拟环境：`/home/cnj/openSource/.venv-a100`

## 4. 目录结构与运行步骤

```
community/projects/a100-adc-requantization/
├── run.py                 # 一键复现入口
├── src/requant/
│   ├── config.py          # init0/init1/合成帧配置（含 adc_sample_num 口径）
│   ├── pipeline.py        # 官方等价解包 + 加窗 + 2D-FFT 谱链路（显式报告多余值）
│   ├── requantize.py      # 固定满量程 k 位再量化（round / trunc）
│   ├── fidelity.py        # RMSE / 频带能量保持 / 峰值漂移 / 弱音可见性
│   ├── synthetic.py       # 合成基准（跨幅度单音、双音、带限噪声、孤立弱音）
│   ├── figures.py         # 5 张图（英文标签带单位）
│   └── cli.py             # 固定种子 20261003 的一键复现
├── tests/test_requant.py  # 22 个测试（理论锚点 + 链路端到端）
├── results/               # synthetic_sweep.csv / real_per_file.csv /
│                          # data_sha256.txt / summary.md
└── figures/               # fig1..fig5（由 cli 重新生成）
```

运行（在仓库根或项目目录均可，repo 根自动探测）：

```bash
# 测试（22 个，全部通过）
/home/cnj/openSource/.venv-a100/bin/python -m pytest community/projects/a100-adc-requantization/tests/ -q

# 一键复现全部结果与图（固定种子 20261003）
cd community/projects/a100-adc-requantization
MPLBACKEND=Agg /home/cnj/openSource/.venv-a100/bin/python run.py
```

`run.py` 等价于 `PYTHONPATH=src python -m requant.cli`。所有 results/ 与
figures/ 产物均可由该命令重跑生成（约 1 分钟）。

## 5. 结果

详细数值见 `results/summary.md`（中文摘要）与两个 CSV。要点：

* **实测安全下限**（基线 top-1% bin 频带能量）：
  单向判据 R ≥ 99% 下 round 全文件最低 k = 8 位；推荐的双侧判据
  |R−1| ≤ 1% 下 round 为 14 位、trunc 为 11 位（最保守文件）。trunc 在
  单向判据下"表面安全到 4 位"是假象：其 +Δ/2 直流偏置向频带内注入能量
  （f0_0 通道 0 在 k=4 时 R=42.3、RD 峰值跳到 (0,0) 且达基线 42 倍），
  单向 ≥99% 被注入"满足"。
* **round vs trunc**：同 k 下 trunc 时域 RMSE ≈ 2×round（Δ²/3 vs Δ²/12，
  合成单音 k=10 实测比值 1.92），且偏置能量集中零多普勒/低距离 bin。
* **弱音可见性**：合成双音（弱音 −50 dB）可见性随 k 单调丧失
  （k=16 约 68.8 dB → k=4 约 34.4 dB）。注意：骑在强载波上的弱音靠
  电平穿越调制（类自然采样 PWM / Δ-Σ 亚 LSB 传递）并不会完全消失，
  丧失的是对抬升噪声底的余量；而**无载波支撑的孤立弱音**（A=38）在
  Δ/2 ≥ A（k ≤ 8）时被整段归零。
* **峰值漂移**：k ≥ 12 时实测峰值位置漂移 0 bin、幅度偏差 ≤ 0.07 dB；
  大漂移只出现在低 k（量化噪声重排 argmax）。
* **数据侧发现**：8 个实测文件 100% 样本为 4 的倍数（低 2 位恒 0），
  即这批 dump 的存储粒度本身 ≤ 14 位，k=14 再量化对其为逐位恒等
  （k=14 行 band_retention 恰为 1）；样本绝对值最大 1972 LSB
  （≈满量程 6%），动态范围约 10 位。

图（英文标签带单位）：

| 图 | 内容 |
|---|---|
| figures/fig1_time_rmse.png | k vs 时域 RMSE（LSB，对数轴），含 Δ/√12 与 Δ/√3 理论线 |
| figures/fig2_spectrum_rmse.png | k vs 距离谱 / RD 谱 RMSE（归一化到基线峰值） |
| figures/fig3_band_retention.png | k vs 频带能量保持率，含 99% 判据线 |
| figures/fig4_two_tone_visibility.png | 双音弱音可见性示例（k=16/12/10/8/6，全图 + 弱音放大两行） |
| figures/fig5_peak_drift.png | k vs 峰值幅度变化（dB）与峰值距离 bin 漂移 |

## 6. 局限

* **16 位存储 ≠ 16 位有效 ADC 精度**：本研究只度量"对已存储样本做
  有损字长削减"的谱后果，**不推断硬件 ENOB**，也不对传感器实际有效
  位数做任何声明（数据侧 4-网格现象只是存储粒度证据，不是 ENOB 证据）；
* **再量化结果不代表更换真实低位 ADC**：真实低位 ADC 的 AGC、滤波、
  噪声色彩、满量程配置都会不同，本研究的固定满量程假设是刻意的
  实验控制，不是硬件预言；
* 离线分析，**未做硬件实测**；官方 JPG 仅作定性参考；
* 官方链路中 rng_fft_scalar=255 / vel_fft_scalar=63 等固件缩放被忠实
  复刻，它们不改变相对指标；
* 安全下限结论依赖这批数据的幅度分布（约 10 位动态、4-网格存储），
  对其他采集不外推；
* 弱音可见性使用局部环形中位数底噪口径（再量化失真稀疏，全局中位数
  在低 k 会塌缩导致虚高）。

## 7. 作者

chennaji9（健康宝 <2078634206@qq.com>）

## 8. 与已有社区项目的差异

已合并的 5 个社区项目——`a100-indoor-cart-mti`、
`a100-indoor-outdoor-radar-analysis`、
`a100-matlab-adc-pipeline-validation`、
`a100-matlab-adc-signal-processing`、
`a100-perimeter-vehicle-radar-analysis`——分别做 MTI 杂波抑制、室内外
场景分析、MATLAB 管线校验、信号处理与周界车辆分析，**均不做
"输入精度 → 谱保真"的预算研究**：没有任何一个项目回答"少存几位、
谱会坏多少、几位以内不可接受"。

与社区中"ADC 批次审计""无损 / 可逆交换"类工具的目标区别：审计类工具
回答"数据是否被改动过"，可逆交换类工具回答"能否不损失地搬移或重排
字节"；本题则**明确研究有损变换**——再量化的损失不可逆，本研究把它
当作一等公民，给出可复现的损失预算（RMSE / 频带能量 / 峰值漂移 /
弱音可见性）与显式判据下的安全下限，而非追求可逆性或完整性校验。
