# A100 目标日志的无损时间分段与还原

作者：Keming Yang（[youngkermit8-coder](https://github.com/youngkermit8-coder)）。独立社区项目，遵循仓库 Apache-2.0 许可证；不修改官方工具。

## 行为

一些回放/导出工具要求帧时间严格递增。遇到重复、倒退、长间隔或缺失时间时，直接排序或补时间会改变证据。本工具保留原文件字节顺序，把可用帧分成单调时间段，并把不可判定帧隔离出来。

- 时间相等、倒退或间隔严格大于max_gap（默认1秒）开启新epoch，不改写墙钟值。
- 缺少、解析失败、同帧多条时间或没有END的帧进入quarantine.bin；下一可用帧重新开段。
- START到下一个START/EOF为原始帧块；BOM、换行、诊断行、空帧和尾部字节保留。
- 重复设备FrameNb不被当作重复帧，不据此分段。
- 每段复制原文件前缀，便于独立解析。manifest记录每帧原始偏移、导出位置、长度、SHA-256与分段原因；prefix.bin单独保留原始前缀。
- reconstruct按索引从各段/隔离文件取回字节，核对连续源区间、逐块和整体哈希；导出结束自动执行一次还原校验。

目标目录必须不存在，不覆盖旧结果。文件系统写入失败可能留下不完整目录，需另选目录重试。当前实现整文件读入内存，不是流式大文件处理器。

## 使用与复现

基线提交为 `9a41e22b22b981d1500e4a28e056407baeb1027f`。复用 [a100-indoor-cart-mti](../a100-indoor-cart-mti/README.md) 的公开数据，采集与场景说明归原贡献者 yyqdbngt；输入与原 manifest 的 SHA-256 核验，结果保留文件名/哈希。未复制原始大文件，未进行新硬件采集，原固件精确版本未知。

只依赖Python标准库，Python >=3.11，本次在Windows/Python 3.11.15验证。输入限UTF-8的START/END目标日志及形如2026/08/02 12:34:56:123---->的时间行，不是通用CSV解析器。

在本目录执行：

```sh
python -m unittest discover -s tests -v
python benchmark.py
python epochs.py ../a100-indoor-cart-mti/data/targets/2026_08_02_16_19_35_Tatget_CTLRR-300-V2_1_RawTarget.csv split-static --max-gap 1
```

还原示例：

```python
from pathlib import Path
from epochs import reconstruct

with Path("restored.csv").open("xb") as output:
    output.write(reconstruct("split-static"))
```

验证脚本在临时目录生成各段并清理，仅覆盖本目录results/summary.json，不把原始日志副本再次提交。

## 验证结果

7份公开RawTarget共**3320帧、10,571,898字节**全部逐字节往返一致；每份当前都形成1个单调epoch、0帧隔离。这批原始数据没有触发异常，不能据此声称已发现设备重启或坏时钟。

9 项测试另覆盖明确标记的人工异常：同刻、倒退、长间隔、跨午夜、重复设备ID、缺时间、多时间、不完整末帧、BOM/CRLF、空帧/尾字节、输出被改动及拒绝覆盖。测试对分段数和边界原因有独立预期。

[逐文件结果与SHA](results/summary.json)。

## 适用边界

这与RawTarget/TraTarget双流配对不同：本工具仅分割单个源日志并提供可逆字节索引；不做跨传感器时钟同步、不推断固件延迟，也不将帧号变成秒数。不同epoch之间不能假装运动连续。隔离数据始终保留，调用者可以检查后决定用途；保留字节不意味着测量值合法，后续仍需测量字段校验。

## 应用边界补充验证

新增 restore(directory, destination) 和 `python epochs.py export-directory restored.csv --restore`：按 64 KiB 字节块恢复，逐段与全文件 SHA 校验完成后才发布目标文件；拒绝覆盖，损坏输入不留下半成品。通过同目录临时文件加硬链接发布，目标文件系统须支持硬链接（本次为 Windows NTFS）。7 份公开日志、3320 帧全部逐字节恢复；最大文件 4,580,956 bytes，返回 bytes 的 API 与文件恢复的 tracemalloc 峰值分别为 9,315,983 / 1,256,279 bytes。该单次测量是 Python 分配峰值，不是进程 RSS；JSON 帧索引仍整体载入，不能声称总内存 O(1)。原 export 和 reconstruct 仍返回完整内存对象。

当前共 9 项测试通过；针对本项的独立回归见 `tests/test_followup.py`。
