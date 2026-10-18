# a100-temporal-confidence：连续点云记录的有效样本量与置信区间

对雷达目标日志这类"连续点云记录"做**离线统计推断**：当帧与帧之间强相关时，
"记录里有 N 帧"不等于"有 N 个独立样本"。本项目在两条 CTLRR-300-V2_1 示例
目标日志上估计每帧统计量的**有效样本量 N_eff** 与 **95% 置信区间**，用独立
bootstrap、移动块 bootstrap（MBB）与平稳块 bootstrap（SBB）对比区间宽度，
并用合成 AR(1) Monte Carlo 验证方法正确性。

## 目标

1. 解析多帧 RawTarget CSV 日志，构造每帧统计序列：目标数 `n_t`、固定极坐标
   网格（Δr=5 m × Δθ=10°）单元占用指示 `o_t`；
2. 估计各序列的自相关函数与截断法有效样本量
   `N_eff = N / (1 + 2·Σρ_k)`（IPS 截断，max_lag = min(100, 5%·N)）；
3. 用预定义 CUSUM 规则做平稳分段（均值变化检测），过短段（<30 帧）判
   "样本不足"而不给区间；
4. 独立帧 bootstrap 与块 bootstrap（块长扫描 b = 2..100，参照
   `b* = N^(1/3)`）的 95% CI 宽度对比；
5. 合成 Monte Carlo 覆盖率验证：iid（≈95%）、AR(1) 已知 φ
   （N_eff 匹配解析式 `N(1−φ)/(1+φ)`；块 bootstrap 覆盖率优于独立
   bootstrap）、强非平稳（分段规则触发、过短段输出"样本不足"）。

## 数据来源

仓内官方附带的上位机示例目标日志（文件名标注 CTLRR-300-V2_1；**非 A100
硬件实测**，为 RadarTools 上位机导出的示例资产）：

| 会话 | 仓库内路径 | 帧数 | 目标行 | SHA256（前 16 位） |
|---|---|---|---|---|
| A | `Python雷达可视化工具示例/负30度_Tatget_2026_03_10_15_34_39_CTLRR-300-V2_1_RawTarget.csv` | 2463 | 35803 | `a9b41f33a54312e0` |
| B | `RadarTools/RadarTools_Release/data/2026_05_20_16_24_32_Tatget_CTLRR-300-V2_1_RawTarget.csv` | 2456 | 78335 | `96fc973ab001e0b7` |

完整哈希见 `results/data_sha256.txt`。实测两文件：帧周期中位数约 53 ms，
epoch 时间戳单调，`END,Total:N` 与帧内行数全部一致（0 处不匹配）。

**日志格式**：首行表头 + 逐帧块 `START,FrameNb:N` → 时间戳行
（`YYYY/MM/DD HH:MM:SS:mmm---->`）→ 元数据行（`heightOffset:0,<epoch_ms>,...`）
→ 目标行若干 → `END,Total:N`。表头列数实测有 57/58/59/62 四种变体，数据行
可能多一个尾逗号空字段，因此解析按位置取列：0=ObjId、1=range[m]、
2=speed[m/s]、3=angle[deg]、4=snr。文件尾部的固件版本横幅行与空行被跳过
并计数。

## 方法（规则全部预先固定）

- **序列定义**：目标数序列 `n_t`；固定极坐标网格 Δr=5 m（覆盖
  [0, 5·⌈r_max/5⌉) m，r_max 取自记录本身，确定性）× Δθ=10°（覆盖
  [-180°, 180°)，36 个方位 bin）的单元占用指示 `o_t`。会话 A 网格
  24×36 单元（r_max=120 m），会话 B 6×36（30 m）。
- **代表单元规则**：在命中 ≥30 帧且占用率 ∈ (0.05, 0.95) 的单元中，取占用率
  最接近 0.5 的两个（并列按单元序号）。A：(32.5 m, −15°) frac=0.517、
  (12.5 m, +65°) frac=0.345；B：(2.5 m, −65°) frac=0.494、
  (2.5 m, +45°) frac=0.506。
- **自相关**：有偏（1/N）估计 ρ_k，max_lag = min(100, ⌊0.05N⌋)。
- **N_eff 截断（IPS）**：`N_eff = N/(1+2·Σ_{k=1..K}ρ_k)`，K 取首个
  ρ_k ≤ 0 前的正项个数（初始正序列法），上限 max_lag；N_eff 裁剪到
  [1, 10N]。iid 数据 K≈0~2，N_eff≈N；正相关数据捕获持续项。
- **平稳分段（CUSUM，预定义规则）**：段内中心化累积和
  `T = max_m |S_m|/(σ̂√n)`，σ̂ 用一阶差分稳健估计
  `σ̂² = mean(diff²)/2`；T > 1.63（布朗桥上确界 1% 临界值）判均值变化，
  在分裂点递归二分；两侧子段均须 ≥30 帧，每序列至多 20 个断点。
  <30 帧的段输出 `insufficient_samples`（样本不足），不给 CI。
- **bootstrap**：独立帧（iid）、移动块（MBB，Künsch：连续 b 帧块拼接裁剪到
  N）、平稳块（SBB，Politis–Romano：几何分布块长、均值 b、循环拼接）；
  百分位 95% CI，B=2000；块长参照 `b* = round(N^(1/3))`（≥2），扫描
  b = 2..30（逐值）∪ 35..100（步长 5）。
- **合成验证设计**：AR(1)，N=800，M=400 套，B=400，μ=3，
  φ ∈ {0, 0.5, 0.8, 0.9}； iid 与块 bootstrap 覆盖率对比；N_eff 估计与解析式
  `N(1−φ)/(1+φ)` 对比。强非平稳用已知阶跃（第 300/600 帧，段内 φ=0.3）；
  过短段用会话 A 目标数前 25 帧。
- **随机性**：单一主种子 20261003，经 `SeedSequence.spawn` 派生各阶段子种子，
  全部产物可复现。

## 环境

- WSL Ubuntu 24.04；Python 3.12.3
- 依赖：numpy 2.5.3、scipy 1.18.1、matplotlib 3.11.2（Agg 后端）、
  pytest 9.1.1；**不使用 pandas**
- 本仓验证所用解释器：`/home/cnj/openSource/.venv-a100/bin/python`

## 运行步骤

```bash
cd community/projects/a100-temporal-confidence

# 1) 测试（33 项，全部通过）
/home/cnj/openSource/.venv-a100/bin/python -m pytest tests -q

# 2) 复现全部产物（约 1~2 分钟；--repo-root 默认从本文件位置自动推断）
PYTHONPATH=src /home/cnj/openSource/.venv-a100/bin/python -m temporal_ci.cli
```

产物写入 `results/` 与 `figures/`，均为确定性输出。

## 结果

序列级摘要（`results/series_summary.csv`，节选）：

| 序列 | N | N_eff | N_eff/N | iid CI 宽 | MBB(b*) CI 宽 | 断点数 |
|---|---|---|---|---|---|---|
| A 目标数 | 2463 | 358.7 | 0.146 | 0.0751 | 0.1214 | 4 |
| A 单元1 | 2463 | 2463 | 1.000 | 0.0402 | 0.0418 | 0 |
| A 单元2 | 2463 | 2463 | 1.000 | 0.0378 | 0.0536 | 5 |
| B 目标数 | 2456 | 412.8 | 0.168 | 0.1287 | 0.2223 | 5 |
| B 单元1 | 2456 | 2456 | 1.000 | 0.0407 | 0.0163 | 0 |
| B 单元2 | 2456 | 2456 | 1.000 | 0.0395 | 0.0159 | 0 |

关键结论（详见 `results/summary.md`）：

- 目标数序列强正相关，N_eff/N ≈ 0.15~0.17（按 iid 用 N 算标准误高估信息量
  约 6~7 倍）；占用单元的检测闪烁则近似 iid（N_eff/N ≈ 1），会话 B 的两个
  单元甚至呈周期-2 交替（ρ₁≈−0.95，IPS 取 K=0 为保守方向）。
- 独立 bootstrap 对目标数序列 95% CI 宽度的低估：以 b*=N^{1/3} 计 1.6~1.7 倍，
  以扫描平台区（b≈60~70）计 2.7~2.8 倍。
- 建议块长：强相关统计量取 b∈[30, 70]（约 2~5×b*；b*=13~14 处宽度仅为平台的
  59%~62%）；近 iid / 交替型单元取 b* 即可。
- 合成验证：iid 覆盖率 94.5%~95.8%；φ=0.9 时独立 bootstrap 跌至 36.0%，
  块 bootstrap 最好 84.5%（3b* > b*，SBB > MBB）；N_eff 截断估计与解析式
  最大偏差 13.2%。已知阶跃序列的 CUSUM 检出 [295, 401, 600]（两个真实断点
  ±6 帧内命中，另有一个段内多余分裂——正相关会推高统计量，分段需人工核对）。

图（英文标签，`figures/`）：

1. `fig1_targets_per_frame.png` — 两会话每帧目标数时间序列（含 51 帧滑动均值）；
2. `fig2_autocorrelation.png` — 目标数与代表单元的 lag-k 自相关（含 iid 零假设
   95% 置信带）；
3. `fig3_neff_stability.png` — N_eff(K)/N 随截断 lag 的稳定曲线（标注 IPS 工作点）；
4. `fig4_ci_width_vs_block_len.png` — CI 宽度 / iid CI 宽度 随块长变化
   （MBB 与 SBB 对比，含 b* 参照线）；
5. `fig5_coverage_validation.png` — AR(1) 合成覆盖率与 N_eff 估计-解析值对比。

## 局限

1. **强非平稳记录不能套单一 N_eff**。CUSUM 在真实目标数序列上检出 4~5 个
   断点；存在断点的记录应分段报告，或把区间理解为"混合时段均值"的区间。
2. **结论是对"已记录统计量"的条件不确定性**——估计的是日志统计量（目标数
   均值、单元占用率）的抽样不确定性，不是雷达检测性能指标，更不能外推为
   硬件指标（检测概率、虚警率等）。
3. **离线分析、未做硬件实测**。输入是仓内 RadarTools 上位机示例日志
   （CTLRR-300-V2_1），非 A100 硬件实测数据。
4. 每会话仅 ~2.5k 帧（约 2 分钟），代表性有限；块 bootstrap 在强相关
   （N_eff/N≈0.15）下仍有欠覆盖（φ=0.9 时最好 84.5%），更长的块受 N 限制。
5. 负相关（周期-2 交替）序列的 IPS 截断给 N_eff≈N，偏保守；MBB/SBB 在此情形
   比 iid 更窄，方向与正相关情形相反。
6. 目标数序列的 MBB 宽度在 b=100 处仍未完全饱和，"平台区"按
   b≥60 子区中位数 ±5% 的预定义规则判定。

## 与仓内已有社区项目的差异

已合并的 5 个社区项目（`a100-indoor-cart-mti`、`a100-indoor-outdoor-radar-analysis`、
`a100-matlab-adc-pipeline-validation`、`a100-matlab-adc-signal-processing`、
`a100-perimeter-vehicle-radar-analysis`）关注信号处理链路、场景分析与可视化，
均不做**时间相关观测的统计推断**；与 CFAR 虚警校准类概率实验的目标也不同——
CFAR 校准回答"检测器的虚警概率是否达标"，本项目回答"这份日志里的统计量
（均值/占用率）本身的不确定度有多大"，即估计日志统计量的抽样不确定性。

## 作者

chennaji9

代码遵循仓库 Apache-2.0 许可。

## 目录结构

```
community/projects/a100-temporal-confidence/
├── README.md
├── .gitignore
├── src/temporal_ci/        # parser / series / effective_n / bootstrap / cli
├── tests/                  # 33 项断言级测试（解析变体、N_eff、分段、bootstrap）
├── results/                # series_summary.csv、block_sweep.csv、
│                           # coverage_synthetic.csv、data_sha256.txt、summary.md
└── figures/                # fig1..fig5（可由 cli.py 重跑生成）
```
