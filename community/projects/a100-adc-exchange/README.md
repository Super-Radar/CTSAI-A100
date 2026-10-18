# CTSAI-A100 ADC 数据可逆交换包（a100-adc-exchange）

一个面向 CTSAI-A100 ADC 文本采集数据的**可逆数据交换工具**（Python）：把仓内官方示例采集 txt 打包为自描述的 `.npz` 包（原始 uint32 字 + 官方语义 int16 解包视图 + 按 chirp 组织的帧 + 来源元数据），支持导出长表 CSV / MATLAB v5 `.mat`，并能从包**逐字节还原**原始 txt（含头字段与多余尾值的原位还原）。`verify` 子命令对包做内部一致性校验 + sha256 溯源 + 逐字节重建证明，篡改包必然报错。

## 1. 项目目标

- **保真**：不丢、不改任何值。官方采集 txt 中的三字段头（B 组）与载荷末尾多出的 1 个尾值（A/B 组均有）都被单独记录，而不是静默丢弃。
- **可逆**：`pack → rebuild` 得到与源文件 sha256 一致、逐字节相同的 txt。
- **可用**：包内直接提供官方解包语义（`adc_mem_2_real_fixed.m`：uint32 → 高 16 位在前、低 16 位在后的两个有符号 int16）的 `samples` 视图与 `(nchirp, rng_nfft)` 帧视图，下游无需再写解包代码。
- **可溯源**：元数据记录源路径、sha256、大小、格式变体、头字段、载荷字数、多余尾值及其位置。

## 2. 数据说明

本工具只做**仓内官方示例数据的离线分析**，未做硬件实测。数据为本仓库自带，无需下载：

| 组 | 相对路径 | 格式 | 文件 |
|---|---|---|---|
| A（无头） | `ADC数据采集/示例adc数据和结果/` | 单行逗号分隔 uint32，共 131073 值 = 载荷 131072 + 1 个多余尾值（0） | 近波 f1 四通道 `adc_近波_20260512141857_f1_*.txt`、远波 f0 四通道 `adc_远波_20260512135827_f0_*.txt`，共 8 个 |
| B（三字段头） | `ADC数据采集/matlab_signal_processing_platform_231023_for_txt_A100/data/` | `adc_test_*_Pf0_Rx*.txt` 以 `0,1024,256,` 式三字段头开始，实测共 131076 值 = 头 3 字段 + 载荷 131072 + 1 个多余尾值（0） | `adc_test_2026052711*_Pf0_Rx{0..3}.txt`，共 4 个 |

B 组目录中另有 4 个 `adc_test_20260512141857_f1_*.txt`，为 A 组近波四通道的无头副本（sha256 与 A 组对应文件一致），故验证集取 8 + 4 = 12 个文件即可全覆盖。

传感器配置来自 `ADC数据采集/matlab_signal_processing_platform_231023_for_txt_A100/cfg/CTASI-A100配置/sensor_config_init{0,1}.hxx`：init0（远波 f0）`rng_nfft=1024, nchirp=256`；init1（近波 f1）`rng_nfft=2048, nchirp=128`；两组期望载荷同为 `rng_nfft/2 × nchirp = 131072` 字（与官方 `load_adc_data.m` 的期望长度一致）。

12 个验证文件的 sha256（完整 64 位见 `results/roundtrip_verification.txt`）：

| 文件 | sha256（前 16 位） |
|---|---|
| adc_近波_20260512141857_f1_0_14_19_05_965.txt | `36db22359f0bc902` |
| adc_近波_20260512141857_f1_1_14_19_14_570.txt | `4fb120427111cdda` |
| adc_近波_20260512141857_f1_2_14_19_23_170.txt | `05f3a38d9297f35c` |
| adc_近波_20260512141857_f1_3_14_19_31_788.txt | `134e89fb95ae8e2d` |
| adc_远波_20260512135827_f0_0_13_58_36_748.txt | `bba39d6f30c7c153` |
| adc_远波_20260512135827_f0_1_13_58_45_367.txt | `33d6b42ec6458934` |
| adc_远波_20260512135827_f0_2_13_58_53_977.txt | `c58910a170558675` |
| adc_远波_20260512135827_f0_3_13_59_02_596.txt | `80ebc8a7f1735612` |
| adc_test_20260527115033_Pf0_Rx0.txt | `c8c112139a1d72f9` |
| adc_test_20260527115041_Pf0_Rx1.txt | `ea15547d808e6714` |
| adc_test_20260527115050_Pf0_Rx2.txt | `7a5ad0f39cb829dd` |
| adc_test_20260527115059_Pf0_Rx3.txt | `e30d54914accca26` |

## 3. NPZ 包格式规范

标准包（`pack` 默认，`package_mode: "full"`）包含 5 个键：

| 键 | 类型 | 形状 | 含义 |
|---|---|---|---|
| `raw_words` | uint32 | `(131072,)` | 载荷字，与文本载荷逐值相同（不含头字段与尾值） |
| `samples` | int16 | `(262144,)` | 解包视图：每字高 16 位样本在前、低 16 位样本在后（官方 `adc_mem_2_real_fixed.m` 语义，含二进制补码） |
| `frames` | int16 | `(nchirp, rng_nfft)` | 按 chirp 组织的 ADC 帧，等价于 MATLAB 平台 `reshape(col,[rng_nfft/2,nchirp])'` 后逐字解包；`frames.ravel() == samples` |
| `trailing_words` | uint32 | `(T,)` | 载荷后多余尾值（官方数据 T=1、值为 0），专用于可逆还原 |
| `metadata` | 0-d unicode（JSON 字符串） | — | 来源与布局元数据（见下） |

`--compact` 归档模式（`package_mode: "compact"`）只存 `raw_words` / `trailing_words` / `metadata`，`samples` / `frames` 在加载时由 `raw_words` 派生，仍可完整校验与逆向重建，体积约为 txt 的 24%。

`metadata` JSON 主要字段：

| 字段 | 含义 |
|---|---|
| `source.{path, repo_relative_path, name, sha256, size_bytes}` | 源文件绝对/仓库相对路径、文件名、sha256、字节数 |
| `format.variant` | `headerless`（A 组）或 `triple_header`（B 组 Pf0） |
| `format.header_fields` | 头字段（如 `[0, 1024, 256]`；无头为 `[]`） |
| `format.text_layout` | 文本布局事实（单行、逗号分隔、无换行、无行尾换行） |
| `profile.{name, rng_nfft, nchirp}` | 使用的传感器配置（init0/init1/custom） |
| `payload_words` / `source_payload_words` | 包内载荷字数 / 源文件载荷字数（partial 时不同） |
| `frames_shape` | 帧形状，如 `[256, 1024]` |
| `trailing.{count, position, values}` | 多余尾值数量、位置（`after_payload`）与具体值 |
| `rx_index` | 通道号（B 组取自头字段，A 组取自文件名） |
| `is_complete` / `chirps_included` | 是否完整包 / partial 包实际包含的 chirp 数 |
| `package_mode` | `full` 或 `compact` |

## 4. 开发环境

- WSL Ubuntu 24.04（Windows 11 宿主）
- Python 3.12.3；numpy 2.5.3；pytest 9.1.1
- scipy 1.18.1（可选，仅 `export --mat` 需要）
- 无需雷达硬件；不联网

## 5. 运行方法

在仓库根目录（`ADC数据采集/` 所在目录）执行；`PYTHONPATH` 指向本项目的 `src`：

```bash
export PYTHONPATH=community/projects/a100-adc-exchange/src

# 打包（A 组无头；profile 按文件名 f0/f1 自动推断，也可 --profile init0|init1 或 --rng-nfft/--nchirp 显式给出）
python -m adc_exchange pack 'ADC数据采集/示例adc数据和结果/adc_远波_20260512135827_f0_0_13_58_36_748.txt' -o /tmp/f0_0.npz

# 打包 B 组三字段头文件
python -m adc_exchange pack 'ADC数据采集/matlab_signal_processing_platform_231023_for_txt_A100/data/adc_test_20260527115033_Pf0_Rx0.txt' -o /tmp/pf0_rx0.npz

# 归档模式（更小）与 partial 模式（前 16 chirp 小样本）
python -m adc_exchange pack 'ADC数据采集/示例adc数据和结果/adc_近波_20260512141857_f1_0_14_19_05_965.txt' -o /tmp/f1_0_compact.npz --compact --profile init1
python -m adc_exchange pack 'ADC数据采集/示例adc数据和结果/adc_近波_20260512141857_f1_0_14_19_05_965.txt' -o /tmp/f1_0_head.npz --profile init1 --limit-chirps 16

# 导出长表 CSV（--max-rows 截断防大文件；--mat 额外导出 MATLAB v5）
python -m adc_exchange export /tmp/f0_0.npz -o /tmp/f0_0.csv --max-rows 1000 --mat /tmp/f0_0.mat

# 逆向重建原始 txt（逐字节，含头字段与尾值原位还原；partial 包会被拒绝）
python -m adc_exchange rebuild /tmp/f0_0.npz -o /tmp/f0_0_rebuilt.txt

# 校验：包内部一致性 + 源 sha256 + 逐字节重建 + 重建文本再摄取等值（篡改包必报错，退出码 1）
python -m adc_exchange verify --npz /tmp/f0_0.npz --txt 'ADC数据采集/示例adc数据和结果/adc_远波_20260512135827_f0_0_13_58_36_748.txt'

# 单元测试（48 项，约 1-2 秒）
python -m pytest community/projects/a100-adc-exchange/tests -q

# 全量真实数据验证 + 生成 results/（12 个采集文件 roundtrip + 体积表 + 小样本包）
python community/projects/a100-adc-exchange/scripts/run_verification.py
```

CSV 长表列：`word_index, chirp_index, word_in_chirp, adc_word, sample_hi, sample_lo`（每行一个 32 位字）。

## 6. 结果展示

以下结果由 `scripts/run_verification.py` 在本仓 12 个官方采集文件上实测生成，产物在 `results/`：

**roundtrip 全通过表**（`pack → rebuild` 与源文件逐字节相同，`verify` 退出码 0；完整日志见 `results/roundtrip_verification.txt`）：

| # | 文件 | 变体 | 载荷字 | 尾值 | sha256 前 8 位 | roundtrip |
|---|---|---|---|---|---|---|
| 1 | adc_近波_..._f1_0_... | headerless | 131072 | [0] | 36db2235 | PASS |
| 2 | adc_近波_..._f1_1_... | headerless | 131072 | [0] | 4fb12042 | PASS |
| 3 | adc_近波_..._f1_2_... | headerless | 131072 | [0] | 05f3a38d | PASS |
| 4 | adc_近波_..._f1_3_... | headerless | 131072 | [0] | 134e89fb | PASS |
| 5 | adc_远波_..._f0_0_... | headerless | 131072 | [0] | bba39d6f | PASS |
| 6 | adc_远波_..._f0_1_... | headerless | 131072 | [0] | 33d6b42e | PASS |
| 7 | adc_远波_..._f0_2_... | headerless | 131072 | [0] | c58910a1 | PASS |
| 8 | adc_远波_..._f0_3_... | headerless | 131072 | [0] | 80ebc8a7 | PASS |
| 9 | adc_test_..._Pf0_Rx0 | triple_header（头 [0,1024,256]） | 131072 | [0] | c8c11213 | PASS |
| 10 | adc_test_..._Pf0_Rx1 | triple_header（头 [1,1024,256]） | 131072 | [0] | ea15547d | PASS |
| 11 | adc_test_..._Pf0_Rx2 | triple_header（头 [2,1024,256]） | 131072 | [0] | 7a5ad0f3 | PASS |
| 12 | adc_test_..._Pf0_Rx3 | triple_header（头 [3,1024,256]） | 131072 | [0] | e30d5491 | PASS |

汇总：**13 PASS / 0 FAIL**（12 个采集 + 1 个小样本包），`roundtrip: ALL BYTE-IDENTICAL`，脚本退出码 0。逐文件数据见 `results/size_comparison.csv`。

**体积对比**（12 文件合计；txt 原文 15,246,331 B）：

| 格式 | 合计字节 | 相对 txt |
|---|---|---|
| txt（十进制文本源） | 15,246,331 B | 100% |
| npz 标准包（含 3 份数组视图 + 元数据） | 10,705,201 B | **70.2%** |
| npz compact 包（仅原始字 + 元数据） | 3,614,124 B | **23.7%** |
| 载荷 uint32 二进制裸值（参考下限） | 6,291,456 B | 41.3% |
| 长表 CSV（全量导出） | 49,472,865 B | 324.5% |

标准包为了“开箱即用”同时携带 raw_words/samples/frames 三份冗余表示，故为 txt 的约 0.7 倍；若以归档为目的，`--compact` 包仅为 txt 的约 0.24 倍（4.2 倍压缩），且同样可逐字节还原。

**小样本包**：`results/sample_near_f1_rx0_first16chirps.npz`（111,149 B < 1 MiB）为近波 f1 Rx0（`adc_近波_20260512141857_f1_0_14_19_05_965.txt`）前 16 个 chirp 的 partial 包（`frames` 形状 `(16, 2048)`）。无需下载全量数据即可体验：

```python
import numpy as np, json
z = np.load('community/projects/a100-adc-exchange/results/sample_near_f1_rx0_first16chirps.npz')
print(z['frames'].shape, z['frames'].dtype)      # (16, 2048) int16
print(json.loads(str(z['metadata'].item()))['source']['sha256'])
```

注意 partial 包 `is_complete=false`，`rebuild` 会拒绝从中还原全量 txt（这是有意设计）。

## 7. 局限

- **不覆盖 USB 原始流**：只解析仓内两种已实测的文本格式（无头 / 三字段头，单行逗号分隔 uint32）；采集链路更上游的 USB 原始数据流不在范围内。
- 尾值语义基于离线观察（A/B 组末值均为 0，恰多 1 个）；工具不解释其含义，只保留任意数量尾值并保证可逆。
- **CSV 大文件**：全量长表 CSV 约为 txt 的 3.2 倍（单文件约 4.1 MB），跨工具交换请优先用 npz/compact npz，CSV 仅用于抽查（`--max-rows`）或表格软件查看。
- `verify --txt` 需要源 txt 在场；无 `--txt` 时仅做包内部一致性校验。
- 本项目为离线数据分析与工具实现，未做雷达硬件实测。

## 8. 与现有社区项目的关系

- 与 **a100-matlab-adc-pipeline-validation**（处理链验证器）互补而非重复：该校验器面向 Range/Doppler/CFAR 处理链的数值验证，其 `load_ctsai_adc.m` 在“实际值数 = 期望载荷 + 1”时按“1 个前导元数据值 + 载荷”处理，执行 `values(2:end)` **静默丢弃首个载荷字**、并把文件末尾的多余值留在载荷里；本工具按官方平台 `load_adc_data.m` 的对齐语义（载荷从头对齐，多余值记为尾部）分离头字段/载荷/尾值，**不丢任何值**，并提供逐字节可逆还原与篡改检测。定位差异：它验证“处理结果对不对”，本工具保证“数据本身完整可交换”。
- 与 a100-matlab-adc-signal-processing、a100-indoor-cart-mti、a100-indoor-outdoor-radar-analysis、a100-perimeter-vehicle-radar-analysis 等处理/应用类项目主题不重叠（它们消费 ADC 数据做信号处理，本项目解决 ADC 数据的打包、交换与还原）。
- 与 open PR 主题不重叠：本分支**只新增** `community/projects/a100-adc-exchange/` 目录，不修改任何既有文件，与任何触及既有路径的 PR 无文件冲突；亦无与其他开放 PR 相同的主题（可逆数据交换包为仓内首例）。

## 9. 作者信息

chennaji9（本仓库社区贡献者；工具 `adc_exchange` v1.0.0，测试与验证脚本随项目提交，全部结果可复现）。

## 项目结构

```text
community/projects/a100-adc-exchange/
├── README.md
├── src/adc_exchange/
│   ├── ingest.py     # 两种格式识别、头/载荷/尾值分离、官方语义解包
│   ├── pack.py       # npz 打包/加载/一致性校验（full 与 compact 两种包）
│   ├── export.py     # 长表 CSV 与 MATLAB v5 .mat 导出
│   ├── rebuild.py    # 逐字节逆向重建原始 txt
│   └── cli.py        # pack/export/rebuild/verify 子命令
├── tests/            # pytest：roundtrip 逐值相等、篡改必报错、官方解包语义对拍
├── scripts/run_verification.py  # 12 文件全量验证 + results/ 生成
└── results/          # roundtrip 记录、体积对比表、小样本 npz
```
