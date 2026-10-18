"""a100-rx-ablation: CTASI-A100 四接收通道非相干积累的通道消融研究。

子模块：
- ``pipeline``  官方 .hxx 解析 + ADC 加载 + 官方 2D-FFT 链路（功率域）复现
- ``metrics``   峰检测、峰稳定性指标、能量守恒校验
- ``ablation``  单通道画像 / 子集扫描 / 留一消融 / 平均 vs 求和对照
- ``plots``     matplotlib (Agg) 出图
- ``cli``       一键复现入口
"""

__version__ = "1.0.0"
__author__ = "chennaji9"
