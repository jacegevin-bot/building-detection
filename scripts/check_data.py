#!/usr/bin/env python3
"""
数据检查脚本
运行完整的数据完整性检查
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.data_checker import run_full_check


def main():
    """主函数"""
    config_path = project_root / "configs" / "config.yaml"

    if not config_path.exists():
        print(f"错误: 配置文件不存在: {config_path}")
        sys.exit(1)

    result = run_full_check(str(config_path))

    if not result.all_passed:
        print("\n请修复以上问题后重试。")
        sys.exit(1)
    else:
        print("\n数据检查通过，可以继续后续流程。")
        sys.exit(0)


if __name__ == "__main__":
    main()
