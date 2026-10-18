# a100-chirp-phase-sensitivity

帧内慢时间相位扰动与相干积累损失的离线分析（CTSAI-A100 固定采样数据）。

## 目标

研究**单帧内**逐 chirp（慢时间）相位扰动对毫米波 FMCW 雷达相干积累与
Doppler 谱的影响：

1. 在合成等幅复单音基准上验证相干积累增益 G(σ) 的理论公式
   （无窗与加窗两版），并给出蒙特卡洛一致性；
2. 用一阶 AR(1) 相位过程与随机游走两种相关扰动模型，考察
   **相同 σ 下相关长度对 G 的影响方向**（理论上两种参数化方向相反，
   本文同时给出并验证）；
3. 对官方示例采集（init0 / f0 × 4 RX），统计主导距离 bin 慢时间复数
   序列的相邻相位差分布，作为"实测参考"的描述统计；
4. 对实测主导 bin 注入受控相位扰动，量化 Doppler 峰幅度退化、峰邻域
   能量占比与谱二阶中心矩（扩散度）曲线。

本项目只研究选定距离 bin 的**慢时间复数序列**（Range FFT 之后跨 chirp
的复数相位），不涉及原始实数样本层面的扰动。

## 数据来源

- 目录：`ADC数据采集/示例adc数据和结果/`（8 个 txt，单行逗号分隔、
  无表头、每文件 131073 个无符号 32bit 整数值；相比
  `rng_nfft/2 × nchirp = 131072` 多 1 个末尾值 0，管道显式报告并截断）。
- 实测实验使用其中 f0（远波）4 个 RX 文件，配置 init0
  （rng_nfft=1024，nchirp=256，vel_nfft=256）；f1（近波）对应 init1
  （2048/128），仅用于哈希记录。
- 每个文件的 SHA256 见 `results/data_sha256.txt`。

### 官方链路复刻口径（MATLAB → Python）

按以下 MATLAB 源文件逐一复刻
（`ADC数据采集/matlab_signal_processing_platform_231023_for_txt_A100/`）：

- `init_function/load_adc_data.m`：截断/补零到
  `[rng_nfft/2, nchirp]`，reshape 后转置；
- `init_function/adc_mem_2_real_fixed.m`：每个 uint32 词拆为**两个
  有符号 int16 实数样本**（高 16 位在前、低 16 位在后），得到
  `[rng_nfft 样本, nchirp]` 实数矩阵（非复数 I/Q）；
- `init_function/get_radar_paramete.m`：`adc_sample_num` 口径为
  `dec_factor==1` 时 `round((adc_sample_end − adc_sample_start) ×
  adc_freq)`（无 +1），再封顶 `rng_nfft`。据此 init0 =
  round(40.8 µs × 25 MHz) = **1020**（1024 点 FFT 末 4 行补零），
  init1 = round(102.4 µs × 25 MHz) = 2560 → 封顶 **2048**（满窗）；
- `FFT/do_fft_2d.m`：距离维 `hann(adc_sample_num)`（MATLAB 周期
  hann，即 `scipy…hann(n, sym=False)`）加窗 → 1024/2048 点 Range FFT
  （÷`rng_fft_scalar`=255）→ 取前半；速度维 `hann(nchirp)` →
  Doppler FFT（÷`vel_fft_scalar`=63）。

## 环境

- WSL Ubuntu，Python 3.12；
- 依赖：numpy、scipy、matplotlib（Agg 后端）、pytest；
- 测试解释器示例：`/home/cnj/openSource/.venv-a100/bin/python`。

## 运行步骤

```bash
cd community/projects/a100-chirp-phase-sensitivity

# 单元测试（30 个用例，覆盖零扰动恒等、纯相位旋转守恒、理论与 MC 一致、
# AR/RW 参数匹配、pack/unpack 往返、多余值报告等）
/home/cnj/openSource/.venv-a100/bin/python -m pytest tests/ -q

# 复现全部实验、CSV、图与摘要（固定种子 20261003）
PYTHONPATH=src /home/cnj/openSource/.venv-a100/bin/python -m phase_sens.cli
```

产物全部落在 `results/`：`synthetic_gain.csv`、`correlation_effect.csv`、
`real_phase_stats.csv`、`doppler_degradation.csv`、`data_sha256.txt`、
`summary.md`、`figures/fig1..fig5.png`。可用 `--data-dir DIR`
`--outdir DIR` 覆盖默认路径。

## 理论锚点（简短推导）

以下均对**已对齐（DC 参考）**的等幅序列讨论：s_k = A·e^{jθ}（θ 为常数
全局相位）、扰动后 s'_k = A·e^{j(θ+φ_k)}，φ 为零均值扰动。注意
Σ_k s_k 这一"直和"只对对齐序列才是相干积累器——非零整 bin 多普勒的
纯音直和为 0，因此对非零多普勒目标（含实测数据）先按主导 Doppler bin
解调到 DC（`demodulate_peak`，实测 Δφ 统计不受影响）再计算 G。
G 对全局相位旋转不变。

**无窗**：G = |Σ_k s'_k|²/(Σ_k|s'_k|)² = |Σ_k e^{jφ_k}|²/N²。
展开 |Σ|² = Σ_{k,l} e^{j(φ_k−φ_l)}。φ 独立同分布时 E[e^{jφ}]=e^{−σ²}，
k≠l 的项各贡献 e^{−σ²}，对角项为 1，故

  **E[G] = 1/N + (1 − 1/N)·e^{−σ²}**。

**加窗（慢时间窗 w_k）**：G_w = |Σ_k w_k s'_k|²/(Σ_k w_k|s'_k|)²。
同样展开，分子期望 = Σ_k w_k² + e^{−σ²}·[(Σ_k w_k)² − Σ_k w_k²]，
分母 = A²(Σ_k w_k)²，故

  **E[G_w] = r + (1 − r)·e^{−σ²}，r = Σ_k w_k²/(Σ_k w_k)²**。

w=1 时 r=1/N，退化为无窗公式。

**相关扰动（联合高斯相位）**：E[e^{j(φ_k−φ_l)}] =
exp(−Var(φ_k−φ_l)/2)。对平稳 AR(1)（a、边缘方差 σ²、
Cov(h)=σ²a^{|h|}）：Var(φ_k−φ_l)=2σ²(1−a^{|h|})，代入成对平均得

  E[G] = 1/N + (2/N²)·Σ_{h=1}^{N−1}(N−h)·exp(−σ²(1−a^h))。

固定**边缘 σ** 时 a↑ ⇒ 指数项↑ ⇒ G↑（相关越强损失越小，极限 a→1 时
G→1，即纯共模相位旋转无损）；固定**新息 σ_ε**（边缘方差
σ_ε²/(1−a²) 随 a 增大）时方向相反：a↑ ⇒ 相位漂移越远 ⇒ G↓，
a→1 极限即随机游走（Var(φ_k−φ_l)=σ_step²|h|）：

  E[G] = 1/N + (2/N²)·Σ_h (N−h)·exp(−σ_step²·h/2)。

## 结果（种子 20261003，详见 results/summary.md）

- 合成基准：N=256/128 的 G(σ) 仿真均值与理论最大偏差在 MC 容差内
  （无窗与 hann 加窗均 <0.005 量级），见图 1；
- 相关长度：固定边缘 σ 时相关越强 G 越大（损失越小，a→1 极限 G→1），
  固定新息 σ_ε 时方向相反（G 随 a 减小），随机游走与 a→1 极限吻合，
  见图 2；
- 实测参考：init0 主导非 DC bin（约 2.6–4.7 m）的相邻相位差圆标准差
  为 1.42–1.84 rad，解调后 G 仅 0.35–0.48——该批采集的主导 bin 接近
  杂波/噪声主导，相邻相位近似去相关（数值见 `real_phase_stats.csv`
  与图 3）；
- Doppler 退化：实测 RX0 在 σ=1 rad 时峰幅度降至约 0.67×、峰邻域能量
  占比由 0.28 降至 0.14（IID 模型）；杂波主导的实测谱二阶中心矩随 σ
  略降（平滑效应），而合成纯净单音上由 0.33 bins² 大幅增大，见图 4、
  图 5。

## 局限（重要）

- **注入模型 ≠ 真实晶振/PLL 相位噪声**：IID 高斯、AR(1)、随机游走仅是
  受控实验的数学扰动模型，不代表 A100 硬件相位噪声的物理机制；
- **不能由实测谱扩散反推硬件相位噪声**：实测相邻相位差统计仅是该批
  采集自身相位稳定度的描述统计，混叠了目标散射起伏、杂波、噪声与
  平台处理链，不构成对硬件的任何标定或反演；
- **结论仅限受控扰动实验**：所有退化曲线都是"注入扰动 → 度量"的
  正向响应，不预测真实系统指标；
- **离线分析、未硬件实测**：仅处理仓内示例数据，未连接雷达硬件。

## 作者

chennaji9

## 与已有社区项目的差异

- 已合并的 5 个社区项目（a100-indoor-cart-mti、
  a100-indoor-outdoor-radar-analysis、a100-matlab-adc-pipeline-validation、
  a100-matlab-adc-signal-processing、a100-perimeter-vehicle-radar-analysis）
  覆盖了 MATLAB/Python 链路复刻、MTI、室内外场景分析与 CFAR 类主题，
  **均不做帧内慢时间相位扰动的响应研究**；
- 与"RX 通道空间积累消融"类研究的区别：本题改的是**帧内时间维**
  （单 bin 慢时间复数序列的逐 chirp 相位），不改 RX 通道组合/空间维，
  也不做通道间积累消融。
