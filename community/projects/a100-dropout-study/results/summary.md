# a100-dropout-study 结果摘要（自动生成）

- 生成命令：`PYTHONPATH=src python3 -m dropout_study.cli`（固定种子，可复现）
- 总耗时：74.3 s；随机基础种子：20260929
- 合成帧：init0，单目标 R=5 m，多普勒 bin=12（v=1.9185 m/s）
- 数据：远波 4 通道 + 近波 4 通道（sha256 见 load_report.csv）

## 实验 1：零扰动回归

- 16/16 项全部通过：severity=0 时字流恒等，峰位漂移=0、峰功率误差=0 dB、总能量误差=0 dB、谱相关=1、杂散峰增量=0。

## 实验 4：可操作阈值（判定：中位数峰位漂移 <1 bin 且 |峰功率误差| <1 dB）

### 合成单目标帧 (init0)（synthetic-init0）

| 缺损模型 | 应对策略 | 最大可通过强度 | 峰功率 1 dB 插值边界 | 峰漂移 1 bin 插值边界 | 备注 |
|---|---|---|---|---|---|
| random_drop | zerofill | 0.1 | 0.10874 | n/a |  |
| random_drop | reject | n/a | n/a | n/a | no tested severity passes |
| zero_segment | zerofill | 64 | n/a | n/a | all tested severities pass |
| zero_segment | reject | 64 | n/a | n/a | all tested severities pass |
| chirp_zero | zerofill | 16 | 28.426 | n/a |  |
| chirp_zero | reject | 16 | 7.7647 | 18.98 |  |
| tail_trunc | zerofill | 0.1 | n/a | n/a | all tested severities pass |
| tail_trunc | reject | 0.1 | n/a | n/a | all tested severities pass |

### 真实远波 4 通道 (init0)（real-far-init0）

| 缺损模型 | 应对策略 | 最大可通过强度 | 峰功率 1 dB 插值边界 | 峰漂移 1 bin 插值边界 | 备注 |
|---|---|---|---|---|---|
| random_drop | zerofill | 0.1 | 0.10821 | n/a |  |
| random_drop | reject | n/a | n/a | 0.0008991 | no tested severity passes |
| zero_segment | zerofill | 64 | n/a | n/a | all tested severities pass |
| zero_segment | reject | 64 | n/a | n/a | all tested severities pass |
| chirp_zero | zerofill | 16 | 29.391 | n/a |  |
| chirp_zero | reject | 4 | 7.1588 | n/a |  |
| tail_trunc | zerofill | 0.1 | n/a | n/a | all tested severities pass |
| tail_trunc | reject | 0.1 | n/a | n/a | all tested severities pass |

### 真实近波 4 通道 (init1)（real-near-init1）

| 缺损模型 | 应对策略 | 最大可通过强度 | 峰功率 1 dB 插值边界 | 峰漂移 1 bin 插值边界 | 备注 |
|---|---|---|---|---|---|
| random_drop | zerofill | 0.05 | 0.10526 | n/a |  |
| random_drop | reject | n/a | n/a | n/a | no tested severity passes |
| zero_segment | zerofill | 64 | n/a | n/a | all tested severities pass |
| zero_segment | reject | 64 | n/a | n/a | all tested severities pass |
| chirp_zero | zerofill | 4 | 12.1 | n/a |  |
| chirp_zero | reject | 1 | 3.5894 | n/a |  |
| tail_trunc | zerofill | 0.1 | n/a | n/a | all tested severities pass |
| tail_trunc | reject | 0.1 | n/a | n/a | all tested severities pass |

## 关键数值（真实远波，4 通道平均）

- 随机丢字 0.5% / 1%（保位补零）：中位数峰功率误差 -0.04 / -0.09 dB
- 整 chirp 置零 8 个（保位补零）：中位数峰功率误差 -0.28 dB
- 尾部截断 5%：保位补零 -0.01 dB / 峰漂移 0.00 bin；截断到完整 chirp -0.45 dB / 峰漂移 0.00 bin

## 实验 5：与格式级审计的关系

| 缺损模型 | 长度审计（a100-adc-audit 式） | 内容级启发式 | 本项目结论 |
|---|---|---|---|
| 尾部截断（丢 ≥1 字） | **可捕获**：文件值数 ≠ 131073（131072 载荷 + 1 尾零惯例） | 无需 | 任何尾部丢字都会改变文件长度 |
| 随机丢字（按位补零） | 静默通过：长度不变 | 散布单字零值难以从数据侧发现 | 0.5% 丢字已带来可测谱退化 |
| 连续零段（1–64 字） | 静默通过：长度不变 | 较长全零游程可被"全零游程"规则捕获 | 位置影响大于长度（头/尾 vs 中） |
| 整 chirp 置零 | 静默通过：长度不变 | "整 chirp 全零"规则可稳定捕获 | 8 chirp 起峰功率损失明显 |

注：长度/哈希/饱和等格式级规则引自在审社区项目 `community/projects/a100-adc-audit` 的思路，本项目不重复实现；
哈希比对只能发现"同一文件两份拷贝不一致"，对首次采集的新数据没有基线可比。谱级影响请参考本项目的退化曲线与阈值表。

## 诚实性声明

人工注入的缺损模型不代表真实采集链路的故障分布；以上结论仅限于“受控缺损模型 + 官方 MATLAB 链路复现”条件下的谱退化规律。详见项目 README 的“局限”一节。
