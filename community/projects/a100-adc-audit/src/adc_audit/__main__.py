"""允许 `python -m adc_audit` 方式启动命令行入口。"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
