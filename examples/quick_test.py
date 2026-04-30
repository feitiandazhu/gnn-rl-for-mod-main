#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速训练示例
运行5个episodes的快速测试
"""

import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from train_wuhan_simple import main
import argparse

if __name__ == "__main__":
    # 设置快速测试参数
    sys.argv = [
        'quick_test.py',
        '--max_episodes', '5',
        '--max_steps', '24',
        '--beta', '0.5',
        '--device', 'cpu'
    ]
    
    print("运行快速训练测试...")
    main()
