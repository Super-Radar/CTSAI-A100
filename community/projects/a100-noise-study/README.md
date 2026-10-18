# a100-noise-study：官方底噪估计方法的区域敏感性与稳健替代研究（Python）

针对 CTSAI-A100 官方 MATLAB 平台的全局底噪估计函数
`calculate_global_noise.m`（固定区域 + 双速度半区取 min + 区域均值），
本项目用 Python 逐行为复现该估计量，并在仓内公开示例 ADC 数据上量化回答三个问题：

1. **区域敏感性**：同一帧数据、不同"看起来都合理"的噪声区域选择，
   官方式均值估计会差多少 dB？
2. **污染脆弱性**：若有目标/杂波/泄漏进入噪声区（单半区 vs 双半区对齐进入），
   官方均值会被抬高多少 dB？
3. **稳健替代**：中位数、分位数、截尾均值、MAD 类估计量在同等污染下
   偏差小多少？

**诚实性红线**：公开示例数据没有"空场景真值"，因此本项目不做任何绝对
底噪 / SNR 断言；所有结论均表述为"相对官方方法的偏差 / 相对无注入基线
的偏差"，一切注入实验均在图表与表格中显式标注为合成注入（synthetic）。

## 1. 项目目标

- 复现官方 `calculate_global_noise.m`（含其完整前置链路：hxx 配置解析 →
  ADC 装载/解包/reshape → `do_fft_2d.m` 加窗 2D-FFT），给出远波/近波
  各 RX 通道与官方 4 通道平均的底噪值，并附 bootstrap 置信区间；
- 量化官方估计量对噪声区域选择（range 起点/宽度 × 速度半区取法）的敏感性；
- 用可控的合成高斯峰注入，量化"固定区域 + 均值"设计的污染脆弱性，
  并与中位数 / 分位数 / 截尾均值 / MAD 类稳健估计量对比；
- 给社区一份可操作的建议：报告底噪时应声明区域与估计量；
  需要抗污染时优先中位数或 MAD 类估计量。

## 2. 数据说明

- 数据来源：本仓库官方示例数据（只读引用，**未复制入库**）
  `ADC数据采集/示例adc数据和结果/`：
  - 远波（profile init0，f0）：`adc_远波_20260512135827_f0_{0..3}_*.txt`（4 个 RX 通道）
  - 近波（profile init1，f1）：`adc_近波_20260512141857_f1_{0..3}_*.txt`
- 文件格式：无表头单行 CSV，逗号分隔 `uint32`；每文件 **131073** 个值，
  载荷期望长度 131072（= rng_nfft/2 × nchirp），恰多多出 1 个尾值，
  经核验 8 个文件尾值均为 0。装载时按官方 `load_adc_data.m` 的截断逻辑
  丢弃该尾值，丢弃动作被显式记录在 `ChannelLoadReport` 并写入
  `results/summary.md`。
- 本项目为**离线分析**，未做任何硬件实测；示例数据本身含真实目标，
  不构成空场景。
- sha256（供数据一致性核对）：

  | 文件 | sha256 |
  |---|---|
  | adc_远波_20260512135827_f0_0_13_58_36_748.txt | bba39d6f30c7c1530e85a2bf7b2509278fbcf03c4a025cb5978a3efdefdfb3a7 |
  | adc_远波_20260512135827_f0_1_13_58_45_367.txt | 33d6b42ec6458934f0a2d66d547678e6604901b7fb26c7bfee44aba3a0d7448d |
  | adc_远波_20260512135827_f0_2_13_58_53_977.txt | c58910a1705586758362c348f6cd347d44ea8ee7d54fe2771193bc23671c3a70 |
  | adc_远波_20260512135827_f0_3_13_59_02_596.txt | 80ebc8a7f1735612b1f91292fefd083058acabc00980c1ed7a00739e5691226f |
  | adc_近波_20260512141857_f1_0_14_19_05_965.txt | 36db22359f0bc902022ec26a07a1a6cb2ed941f66f75729c42d01bf56c4d273c |
  | adc_近波_20260512141857_f1_1_14_19_14_570.txt | 4fb120427111cddaf220433bf85491a700633ca28e4b41d5ea20ee01707960ab |
  | adc_近波_20260512141857_f1_2_14_19_23_170.txt | 05f3a38d9297f35c7972d84d8f9e300b269457b3e6cd00e5f51d3172ac6c306e |
  | adc_近波_20260512141857_f1_3_14_19_31_788.txt | 134e89fb95ae8e2d7955163e888040ecf59fc3895c21841debc07fb51127e7bb |

- **无空场真值**：因此不做绝对底噪断言（见"结论"与"局限"）。

## 3. 方法

### 3.1 官方算法逐条复现（引用 .m 行为）

复现对象（源文件位于
`ADC数据采集/matlab_signal_processing_platform_231023_for_txt_A100/`，只读）：

- **`FFT/calculate_global_noise.m`**（→ `src/noise_study/official_noise.py`）：
  1. `RVimage` = 4 通道 `power(abs(Radar_cube),2)` 求和后除以
     `nvirtual_array`（即通道平均功率），再 `db(..., 'power')`
     （= `10*log10`，功率量纲）转 dB；
  2. 噪声区域 range 边界：`round(0.75*rng_nfft/2) .. round(0.8*rng_nfft/2)`
     （MATLAB 1-based 含端点；`round` 为"四舍五入远离零"，与 Python
     默认银行家舍入不同，已单独实现 `matlab_round`）；
  3. 速度两个半区：`6 .. round(vel_nfft/2-6)` 与
     `round(vel_nfft/2+6) .. round(vel_nfft-6)`；
  4. 两半区子图**逐元素取 min** 得 `noiseRegion`；
  5. `noiseValue = mean(noiseRegion, 'all')`。
     - init0（rng_nfft 1024 / vel_nfft 256）：range bins 384..410，
       速度 6..122 与 134..250，min 合成后 27×117 = 3159 cells；
     - init1（2048/128）：range 768..819，速度 6..58 与 70..122，
       52×53 = 2756 cells。
- **`init_function/load_adc_data.m` + `adc_mem_2_real_fixed.m`**
  （→ `pipeline.load_adc_data` / `pipeline.adc_mem_2_real_fixed`）：
  uint32 高 16 位/低 16 位按二补码解为两个 int16 样本，前一样本 = 高 16 位；
  列优先 `reshape(col, [rng_nfft/2, nchirp])` 后转置，等价于 NumPy 的
  C-order `reshape(nchirp, rng_nfft/2)`；超长截断/过短补零到期望长度。
- **`FFT/do_fft_2d.m`**（→ `pipeline.do_fft_2d`）：
  距离维 `hann(adc_sample_num)` 加窗（只加前 `adc_sample_num` 个样本，
  其余补零到 `rng_nfft`）→ Range FFT 除以 `rng_fft_scalar`（0xff=255）→
  取前 `rng_nfft/2` 行 → 速度维 `hann(nchirp)` 加窗 → Doppler FFT 到
  `vel_nfft` 除以 `vel_fft_scalar`（0x3f=63）。两个 FFT 标量直接影响
  底噪绝对 dB 量级，复现时原样保留。
- **`init_function/get_radar_paramete.m`**（→ `pipeline.parse_sensor_config`）：
  解析 hxx（十六进制/字符串窗名→数值码/`tx_groups` 位运算求 TX 数）并按
  原公式计算派生参数。

**`adc_sample_num` 复核结论（自行核验，修正常见误记）**：
`get_radar_paramete.m` 中该值为

```matlab
if dec_factor > 1
    adc_sample_num = round((end-start)*adc_freq/dec_factor) + 1;
else
    adc_sample_num = round((end-start)*adc_freq/dec_factor);   % 无 +1
end
```

- init0（`dec_factor=1`）：`round((42.8-2)*25) = 1020`（**不是 1021**；
  `+1` 只在 `dec_factor>1` 分支存在），1020 < rng_nfft 1024，不截断，
  Range FFT 补 4 个零点；
- init1：`round((104.4-2)*25) = 2560 > rng_nfft 2048`，**按原代码钳位到
  2048**（即窗与 FFT 长度均为 2048，无补零）。

### 3.2 稳健估计量定义（对同一噪声区域 dB 值 x，n 个 cell）

| 估计量 | 公式 |
|---|---|
| official_mean（官方） | `mean(x)` |
| median | `median(x)` |
| q25 / q10 | `Q0.25(x)` / `Q0.10(x)`（线性插值，numpy 默认） |
| trim10 / trim20 | 排序后两侧各删 `floor(p·n)` 个值再取均值，p=0.10/0.20 |
| mad_floor | `median(x) + 1.4826·MAD(x)`，`MAD(x)=median(|x−median(x)|)` |

`1.4826 = 1/Φ⁻¹(3/4)` 使 `1.4826·MAD` 成为高斯情形下标准差的一致估计，
故 `mad_floor` 是"中位数 + 1σ（高斯等价）"水平。注意它估计的统计目标
与官方均值不同（中位数+1σ vs dB 均值），本项目只在"污染/换区域下的
**偏差**"这一维度上比较各估计量。

- **bootstrap 置信区间**：对区域 cell 有放回重采样（B=2000，
  种子 20260929 固定），取估计量的 2.5%/97.5% 分位。
- **合成注入**：在真实 RVimage（dB 域）上加高斯形峰
  `amp·exp(−(Δr²/2σr² + Δv²/2σv²))`（dB 相加 = 功率相乘），
  注入位置取两速度半区**对齐偏移**（both-halves，对官方 min 步骤的最坏
  情形）或仅单半区；一切输出显式标注 synthetic。

## 4. 开发环境

- Ubuntu (WSL2)，Python 3.12.3
- numpy 2.5.3 / scipy 1.18.1（未直接依赖）/ matplotlib 3.11.2（Agg 后端出 PNG）/ pytest 9.1.1
- 无需 pip 安装本包（src 布局，`PYTHONPATH=src` 直接运行）

## 5. 运行方法

```bash
# 仓库任意 checkout 内（默认自动定位仓内示例数据与 hxx 配置）
cd community/projects/a100-noise-study
PYTHONPATH=src python3 -m noise_study.cli            # 一键重生成全部图表

# 数据/配置/输出目录可参数化
PYTHONPATH=src python3 -m noise_study.cli --data-dir <adc目录> \
    --hxx-dir <hxx目录> --out-dir <输出目录> --seed 20260929

# 测试
python3 -m pytest tests/ -q
```

输出：`figures/fig1..fig4*.png` 与 `results/*.csv` + `results/summary.md`
（全部随仓库提交，可一键复现，固定种子）。

## 6. 结果展示

### 6.1 复现表（`results/reproduction_table.csv`，dB）

| dataset | channel_set | official_mean | boot95(official) | median | boot95(median) |
|---|---|---|---|---|---|
| far (init0) | ch0 | −14.18 | [−14.38, −13.99] | −13.21 | [−13.39, −13.03] |
| far | ch1 | −9.69 | [−9.88, −9.49] | −8.85 | [−9.06, −8.62] |
| far | ch2 | −9.76 | [−9.95, −9.58] | −8.96 | [−9.19, −8.74] |
| far | ch3 | −10.18 | [−10.38, −9.99] | −9.25 | [−9.46, −9.10] |
| far | **4ch 平均（官方口径）** | **−7.51** | [−7.59, −7.44] | −7.38 | [−7.50, −7.27] |
| near (init1) | ch0 | −13.88 | [−14.10, −13.67] | −12.93 | [−13.25, −12.68] |
| near | ch1 | −8.75 | [−8.97, −8.55] | −7.82 | [−8.08, −7.61] |
| near | ch2 | −8.82 | [−9.02, −8.60] | −7.97 | [−8.21, −7.80] |
| near | ch3 | −9.09 | [−9.30, −8.89] | −8.10 | [−8.34, −7.95] |
| near | **4ch 平均（官方口径）** | **−6.70** | [−6.79, −6.60] | −6.57 | [−6.67, −6.46] |

两点可复核的观察：(a) RX0 的区域均值比 RX1–3 低约 4–5 dB（通道间底噪
水平确实不同）；(b) 4 通道平均图的官方式均值（−7.51）比任何单通道都高，
这是"先对 4 通道功率取平均再取 dB"（凹函数，Jensen 上偏）叠加
"min 双半区"共同作用的统计效应，属官方口径的固有性质而非实现差异——
比较底噪时必须声明通道平均方式。

### 6.2 注入偏差对比（`figures/fig2_contamination_demo.png`，
`results/contamination_demo.csv`，远波 4ch，合成注入）

| 场景（synthetic） | 官方均值偏差 | 中位数偏差 | mad_floor 偏差 |
|---|---|---|---|
| 单半区注入 +20 dB，σ=2 bins | +0.03 dB | +0.03 dB | +0.03 dB |
| 双半区对齐 +20 dB，σ=2 | +0.16 dB | +0.09 dB | +0.14 dB |
| 双半区对齐 +40 dB，σ=2 | +0.32 dB | +0.11 dB | +0.17 dB |
| 双半区对齐 +40 dB，σ=6（宽污染） | **+2.79 dB** | **+0.82 dB** | +1.52 dB |

σ=6、+40 dB 时官方区域 3159 cells 中有 593 个被抬高 >3 dB（≈19%）。
单半区注入几乎被官方 min 步骤完全抑制（这是官方设计的一个隐藏稳健性，
值得保留）；威胁来自**双半区对齐进入或宽污染**，此时官方均值偏差约为
中位数的 3–4 倍。

### 6.3 区域扫描热力图（`figures/fig3_region_scan_heatmap.png`，
`results/region_scan.csv`）

range 起点 ∈ 0.60..0.90 × rng_nfft/2（13 档）× 宽度 ∈ {16..128} bins
（11 档）× 速度区域两种取法（官方双半区 min / 仅前半区），同一帧真实数据：

| dataset | 速度取法 | 扫描区域数 | 官方式均值 min..max | 散布 (max−min) |
|---|---|---|---|---|
| far | two_half_min（官方） | 124 | −8.65 .. −7.01 dB | **1.64 dB** |
| far | first_half | 124 | −7.16 .. −5.46 dB | 1.70 dB |
| near | two_half_min（官方） | 141 | −8.25 .. −5.68 dB | **2.57 dB** |
| near | first_half | 141 | −6.61 .. −4.19 dB | 2.42 dB |

即：不改变数据、只改变"看起来都合理"的远端区域选择，底噪估计本身就有
1.6–2.6 dB 的散布；超界组合（起点+宽度越界）在图中留白。

### 6.4 偏差曲线（`figures/fig4_bias_curves.png`，
`results/bias_curves.csv`，双半区对齐注入，合成）

注入峰幅度 0→40 dB 扫描（σ=2 紧凑峰与 σ=6 宽污染两种宽度，近/远两组数据），
40 dB 处的偏差：

| dataset | σ (bins) | official_mean | median | mad_floor |
|---|---|---|---|---|
| far | 2 | +0.32 dB | +0.11 dB | +0.17 dB |
| far | 6 | +2.79 dB | +0.82 dB | +1.52 dB |
| near | 2 | +0.36 dB | +0.07 dB | +0.16 dB |
| near | 6 | +3.28 dB | +1.06 dB | +1.82 dB |

官方均值偏差随注入幅度近似线性增长且随峰宽急剧放大；
中位数曲线在紧凑峰下几乎平坦（<0.11 dB），宽污染下也比官方低 3–4 倍。

## 7. 结论（给社区的可操作建议）

1. **报告底噪时必须声明口径**：噪声区域（range/速度边界）+ 估计量
  （均值/中位数/…）+ 通道平均方式（先平均功率再取 dB，还是逐通道）。
   同一帧数据仅换区域就有 1.6–2.6 dB 散布，口径不明则数值不可比。
2. **官方"固定区域 + 双半区 min + 均值"对单半区紧凑目标有隐藏稳健性**
   （min 步骤将其抑制到 +0.03 dB 量级），但对双半区对齐污染/宽污染
  （泄漏底座、扩展杂波）敏感：+40 dB/σ=6 bins 注入时官方均值抬高
   2.8–3.3 dB，而中位数仅 0.8–1.1 dB、约为官方的 1/3～1/4。
3. **优先中位数或 MAD 类估计量**作为底噪报告口径（或至少作为交叉校验）：
   紧凑峰下中位数偏差 <0.11 dB，宽污染下亦不超过官方的 1/3；
   `mad_floor`（median + 1.4826·MAD）同时给出稳健的离散度，便于
   CFAR 类阈值换算。
4. 官方区域较大（3159 cells）对紧凑峰有稀释作用；真正需要防范的是
   **宽污染**与**双半区对称进入**的泄漏。
5. 以上全部为"相对偏差"结论；绝对底噪/SNR 需要空场标定数据支持，
   当前公开数据不具备。

## 8. 局限

- 无空场真值：不能评估任何估计量的绝对准确度，只能比较相对偏差；
- 仅 2 帧公开示例数据（远/近各一帧 4 通道），统计功效有限，
  未做多帧/多场景外推；
- 注入为 dB 域高斯形合成峰，未建模真实目标经加窗 FFT 后的旁瓣结构
  （Hann 旁瓣约 −31.5 dB 的泄漏底座是更真实的"宽污染"来源之一，
  本项目用 σ=6 高斯峰近似之）；
- `mad_floor` 的统计目标（中位数+1σ）与官方均值不同，二者绝对值
  不可直接相减比较，只在偏差维度比较；
- 仅实现 `load_adc_data.m` 的 `nvirtual_chirp == 1`（1TX4RX）分支，
  公开 profile 恰好属于该分支；chebwin 窗（官方码 1）未复现（公开
  配置未使用）。

## 9. 与现有社区项目的关系

- 现有 5 个社区项目（`a100-indoor-cart-mti`、`a100-indoor-outdoor-radar-analysis`、
  `a100-matlab-adc-pipeline-validation`、`a100-matlab-adc-signal-processing`、
  `a100-perimeter-vehicle-radar-analysis`）均未研究底噪估计量的
  区域敏感性与稳健性问题，本项目填补该空白；
- 与 `a100-matlab-adc-pipeline-validation` 互补：该项目关注 ADC→FFT→CFAR
  链路正确性与 CFAR 检测（**使用**噪声水平做阈值），本项目关注噪声水平
  本身**估计得稳不稳**——CFAR 阈值质量受底噪估计偏差直接影响；
- 复用同一套公开示例数据与官方 .m 参考，未与任何 open PR 主题重叠。

## 10. 作者信息

- 作者：chennaji9
- 项目目录：`community/projects/a100-noise-study/`
- 许可：随仓库主许可（见仓库根 LICENSE）

## 附：目录结构

```
community/projects/a100-noise-study/
├── README.md
├── src/noise_study/
│   ├── pipeline.py        # hxx 解析 + ADC 装载/解包/reshape + 官方 do_fft_2d 链路
│   ├── official_noise.py  # calculate_global_noise.m 逐行为忠复现
│   ├── robust.py          # 稳健估计量 + bootstrap + 合成峰注入
│   ├── study.py           # 实验 1-4 与图表生成
│   └── cli.py             # 一键复现入口
├── tests/                 # pytest：官方复现回归 / 稳健估计量手算对拍 / Parseval 能量守恒
├── figures/               # fig1..fig4 PNG（英文标签，避免 matplotlib CJK 字体依赖）
└── results/               # CSV + summary.md（数值表）
```
