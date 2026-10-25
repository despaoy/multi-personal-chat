#!/usr/bin/env python3
"""MultiPersonal Chat System - 后端主服务入口（向后兼容）

此文件保留用于向后兼容。新入口请使用 run.py。
实际应用逻辑已拆分到 app/main.py。
"""

import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

if __name__ == "__main__":
    from run import main

    main()
