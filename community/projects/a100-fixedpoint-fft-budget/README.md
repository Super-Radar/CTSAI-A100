# a100-fixedpoint-fft-budget：二维 FFT 定点位宽与逐级缩放预算

针对 CTSAI-A100 雷达 2D-FFT（Range FFT + Doppler FFT）链路的**定点运算位宽（wordlength）与逐级缩放（scaling）预算**离线研究：用 numpy int64 精确整数算术仿真 B 位二补码 FFT 数据通路，扫描 B、旋转因子量化位数 Q 与缩放计划，给出"达到 float64 参考 1% 误差以内"的最低预算组合、溢出发生边界与缩放建议，并用官方示例 ADC 数据（init0 配置、单 RX、按官方 hann 窗）验证。

**机器无关声明**：本研究只报告数值误差统计（相对谱误差、峰值保真、能量关系、逐级峰值/溢出计数），不报告任何运行时间或内存数据；仿真算术格式是为位宽预算设计的模型，**不代表 A100 芯片内部实现**。

## 目录结构

```
a100-fixedpoint-fft-budget/
├── README.md
├── conftest.py                 # 将 src/ 加入 sys.path（pytest 用）
├── src/fixed_fft/
│   ├── fft_fixed.py            # radix-2 迭代定点 FFT 仿真器（B 位二补码、Q 位旋转因子、
│   │                           #   none / scale-1bit-each-stage 缩放、saturate / wrap 溢出处理、
│   │                           #   逐级峰值与溢出统计）
│   ├── inputs.py               # 固定种子压力输入生成器 + 官方 ADC 数据链路加载器
│   ├── budget.py               # 与 float64 参考的误差指标 + B/Q/scaling 扫描实验
│   └── cli.py                  # 一键复现（固定种子 20261003）：实验、CSV、图、summary.md
├── tests/test_fixed_fft.py     # 17 个测试（见下）
├── figures/                    # fig1..fig5（英文标签带单位）
└── results/                    # sweep_B.csv / sweep_Q.csv / scaling_compare.csv /
                                #   real_data_error.csv / stage_waterfall.csv /
                                #   data_sha256.txt（8 个数据文件）/ summary.md（中文摘要）
```

## 数据来源与官方链路

- 数据：`ADC数据采集/示例adc数据和结果/` 下 8 个 txt（远波 f0 ×4 + 近波 f1 ×4，每文件 4 RX 之一）。每文件为单行逗号分隔的 131073 个无符号整数（uint32 存储字，末尾多 1 个 0）；sha256 见 `results/data_sha256.txt`。
- 官方链路（本研究在 Python 中逐条复刻，用于实验⑤）：
  1. `ADC数据采集/matlab_signal_processing_platform_231023_for_txt_A100/init_function/load_adc_data.m`：截断/补零到 `rng_nfft/2 × nchirp` 个字，按 MATLAB 列主序 reshape 为 `[rng_nfft/2, nchirp]` 再转置成 `[nchirp, rng_nfft/2]`；
  2. `init_function/adc_mem_2_real_fixed.m`：每个 32 位字拆 **H16 在前、L16 在后** 两个有符号 int16 **实数**样本（非复数），得到每 RX `[nchirp, rng_nfft]` 实样本矩阵；
  3. `FFT/do_fft_2d.m`：距离维先按**实际采样点数**加对称 hann 窗，FFT 到 `rng_nfft` 后取前半，慢时间维再加窗做 Doppler FFT。MATLAB 的 `fft(...)/rng_fft_scalar(=255)`、`/vel_fft_scalar(=63)` 是纯输出归一化，在相对误差中相消，本研究不重复实现。
- `adc_sample_num` 口径（由 `init_function/get_radar_paramete.m` 推导）：`round((adc_sample_end − adc_sample_start) × adc_freq / dec_factor)`，上限 `rng_nfft`：
  - init0（`cfg/CTASI-A100配置/sensor_config_init0.hxx`，rng_nfft=1024、nchirp=256、adc_freq=25、dec_factor=1、sample 2→42.8）：`round(40.8×25)=1020 ≤ 1024`，即 **1020** 点加窗 + 4 点补零；
  - init1（`sensor_config_init1.hxx`，rng_nfft=2048、nchirp=128、sample 2→104.4）：`round(102.4×25)=2560 > 2048`，被钳位到 **2048**，用满、无补零。
- 实验⑤使用 init0、远波 f0 的 RX0 单文件做 range-FFT（每 chirp 一条 1024 点 FFT，共 256 chirp）。

## 仿真器模型与误差口径

- **数据通路**：B 位二补码（B∈[8,32]），蝶形算术全部在 int64 上精确完成（位精确、机器无关）。输入为整数 LSB 样本；若输入动态范围超出 B 位（仅 B<16 时会发生），先做**最小右移**装入字内并在结果中记录 `input_shift`。
- **旋转因子**：独立系数字长 Q 位，`W = (round(cos·2^Q) − j·round(sin·2^Q))/2^Q`；乘积经算术右移 Q 位截断回整数数据通路（floor，等价简单硬件截断器；±1、±j 等特殊角精确量化）。
- **缩放计划**：`none` 或 `scale-1bit-each-stage`（每级蝶形后算术右移 1 位）。
- **溢出处理**：`saturate`（饱和到 ±满量程）或 `wrap`（二补码折返），逐级计数并记录；int64 余量保护要求 B+Q ≤ 60。
- **参考与指标**：float64 参考（`numpy.fft`）与定点核吃**同一组整数输入样本**，误差只反映 FFT 运算过程的有限字长效应。指标：单边功率谱 L2 相对误差、峰值 bin 位移与幅度相对误差、双边谱能量比、逐级峰值（LSB/bit）与溢出计数。真实数据实验中，加窗后样本取整到 int16 LSB 网格，同一向量喂参考与定点核（窗系数量化不在本题范围）。

## 环境

- WSL / Python 3.12.3；venv 内 numpy 2.5.3、scipy 1.18.1（已装，本项目代码未直接调用）、matplotlib 3.11.2（Agg 后端）、pytest 9.1.1。
- 离线运行，不需要任何硬件、网络或 MATLAB。

## 运行步骤

```bash
cd community/projects/a100-fixedpoint-fft-budget
# 测试（17 个，全绿）
/home/cnj/openSource/.venv-a100/bin/python -m pytest tests/ -q
# 复现全部实验、CSV、图与 summary.md（默认种子 20261003）
/home/cnj/openSource/.venv-a100/bin/python src/fixed_fft/cli.py --seed 20261003
```

`--data-root` 默认指向仓库内 `ADC数据采集/示例adc数据和结果/`，可用参数覆盖。

## 结果（要点）

- **预算下限**：全部合成压力输入功率谱相对误差 < 1% 需要 **B ≥ 26（Q=16，不缩放）**、**Q ≥ 8（B=28，不缩放）**；官方真实数据（init0 单 RX range-FFT）逐 chirp 平均误差 < 1% 需要 **B ≥ 17（Q=16，不缩放）**。
- **溢出边界（saturate，Q=16，不缩放）**：满幅多音/整数 bin 单音在 B ≤ 23、DC 与 ±32767/−32768 符号边界在 B ≤ 25、带限噪声在 B ≤ 20 仍发生溢出；冲激输入全程无溢出。**每级右移 1 bit 的缩放计划在 B=8..32 全范围零溢出**。
- **缩放建议**：位宽受限时用逐级缩放消除饱和失真（如 B=16 时多音误差从 0.99 降到 2.0e-3）；位宽充足（B ≥ 24~26）且输入留有余量时不缩放精度更高（B=24 时多音 2.5e-5 对 2.0e-3）。
- **真实数据误差地板**：B ≥ 17 后平均误差稳定在约 3.7e-3——该数据加窗后谱峰仅约满量程单音的 0.3%，误差由整数 LSB 数据通路逐级截断的绝对噪声主导（与 Q 基本无关），体现定点量程利用率对精度底噪的影响。
- 图：`figures/fig1_error_vs_B.png`（误差 vs B，合成输入分面）、`fig2_error_vs_Q.png`（误差 vs Q）、`fig3_scaling_compare.png`（缩放计划对比）、`fig4_stage_waterfall.png`（B=32 无饱和逐级增长轨迹 + B=16 溢出计数）、`fig5_real_data_range_fft.png`（官方数据定点 vs 浮点）。
- 完整数据见 `results/`（CSV 可由 cli 重跑再生），中文结论汇总见 `results/summary.md`。

## 测试（17 个，覆盖要求的全部断言类别）

DC 仅 bin0 非零（精确整数）；整数 bin 单音精确落位与幅度标定（含 N/4 特殊格的**精确**整数 DFT 恒等）；高 B/Q 收敛到浮点参考；通用小规模输入与 `numpy.fft` 的一致性；构造必溢出小 B 输入的溢出计数与饱和轨/二补码折返的**精确**终值；scale-1bit-each-stage 的增益不变量（DC 逐级峰值恒定、去增益后能量守恒）与无饱和；符号边界（−32768 可表示、+32768 饱和、B<16 预移位记录）；旋转因子量化往返（Q=16/24 误差 < 2^−Q）；参数越界拒绝；官方数据链路形状与补零校验。

## 局限

1. **仿真算术格式不代表 A100 芯片内部实现**——B/Q/截断/饱和等仅为位宽预算的建模载体，与芯片真实数据通路、调度和字长安排无对应关系。
2. **不报告速度基准**：全文只有数值误差统计，无任何运行时间/内存/吞吐测量（机器无关）。
3. **离线分析、未硬件实测**：结论基于合成压力输入与官方示例 ADC 数据（远波 f0 RX0、init0 range-FFT 一条链路），未做 Doppler 维定点实验与实机验证。
4. B < 16 时输入需预移位装入字内（等价输入再量化），已逐行记录在 CSV 的 `input_shift` 列；B ≥ 16 输入无损。
5. 窗系数量化不在研究范围（真实数据实验中窗在 float64 域施加后取整，参考与定点共用同一输入）。

## 作者

chennaji9

## 与已有社区工作的差异

- 已合并的 5 个社区项目（`a100-indoor-cart-mti`、`a100-indoor-outdoor-radar-analysis`、`a100-matlab-adc-pipeline-validation`、`a100-matlab-adc-signal-processing`、`a100-perimeter-vehicle-radar-analysis`）分别做 MTI/内外场分析/MATLAB 链路复刻/信号处理/周界车辆分析，**均不做 FFT 运算位宽与缩放预算**；本项目首次给出 B、Q 与缩放计划的系统扫描和 1% 预算边界。
- 与"输入样本再量化"类研究的区别：本题**保持输入不变**（合成压力输入与官方 ADC 的 int16 样本原样进入参考与定点核），只改变**运算过程**（蝶形算术字长、旋转因子量化、逐级缩放、饱和/折返）产生的有限字长效应，误差归因干净。
