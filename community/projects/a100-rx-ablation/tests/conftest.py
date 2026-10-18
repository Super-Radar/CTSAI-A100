# -*- coding: utf-8 -*-
"""pytest 配置：把 src/ 加入 sys.path，使测试与 cli 均可脱离安装运行。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
