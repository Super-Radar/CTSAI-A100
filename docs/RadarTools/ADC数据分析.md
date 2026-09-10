---
doc_id: "2064911668388741120"
title: "ADC数据分析"
parent_id: "2064910734866694144"
sort: 4
content_type: 1
---

本节用于说明如何基于示例工程读取和分析 CTSAI-A100 ADC 数据。

### Matlab 示例工程目录

Matlab 示例工程位于：

```text
ADC数据采集/matlab_signal_processing_platform_231023_for_txt_A100/
```

示例工程中包含 CTSAI-A100 ADC 数据处理入口脚本：

```text
ct_signal_processing_main_simple_CTASIA100.m
```

### 选择波形配置文件

打开以下文件：

```text
ADC数据采集/matlab_signal_processing_platform_231023_for_txt_A100/ct_signal_processing_main_simple_CTASIA100.m
```

根据采集时使用的波形选择对应配置文件。

远波 ADC 数据使用：

```text
sensor_config_init0.hxx
```

近波 ADC 数据使用：

```text
sensor_config_init1.hxx
```

配置示例：

```matlab
cell_cfg_file_path = {
    '.\cfg\CTASI-A100配置\sensor_config_init0.hxx'
};
```

### 拷贝 ADC 数据

将 RadarTools 采集到的 ADC 数据文件从以下目录：

```text
RadarTools/RadarTools_Release/adcData/
```

拷贝到 Matlab 示例工程的数据目录：

```text
ADC数据采集/matlab_signal_processing_platform_231023_for_txt_A100/data/
```

若数据位于会话子目录中，可以复制整个会话目录，也可以将所需文件直接复制到
示例工程的 `data/` 目录。多通道分析应选择同一会话、同一波形配置（profile）的
`Rx0`～`Rx3` 文件。

### 配置数据路径和文件名

在 `ct_signal_processing_main_simple_CTASIA100.m` 中配置 ADC 数据所在目录：

```matlab
cell_data_file_path = '.\data';
```

配置接收通道对应的数据文件名。以下示例使用仓库现有的旧版平铺文件：

```matlab
cell_data_file_name_list = {
    [cell_data_file_path, '\', 'adc_test_20260522110324_Pf0_Rx0.txt'];
    [cell_data_file_path, '\', 'adc_test_20260522110332_Pf0_Rx1.txt'];
    [cell_data_file_path, '\', 'adc_test_20260522110341_Pf0_Rx2.txt'];
    [cell_data_file_path, '\', 'adc_test_20260522110350_Pf0_Rx3.txt'];
};
```

如果保留会话子目录，应在文件名中包含该目录。例如：

```matlab
fullfile(cell_data_file_path, 'adc_test_20260707154459626', ...
    'run_001_Pf0_Rx0.txt')
```

文件名、波形配置和接收通道必须与本次采集文件保持一致。

### 运行 Matlab 处理脚本

在 Matlab 中运行：

```matlab
ct_signal_processing_main_simple_CTASIA100
```

运行后，可根据示例工程输出查看处理结果。

该流程适合用于：

* ADC 数据读取
* 波形参数加载
* 距离向处理
* 速度向处理
* 基础目标检测
* 雷达信号处理流程验证
