#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的环境测试脚本
"""

import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.envs.simple_wuhan_env import SimpleWuhanAMoDEnv

def test_environment():
    print("测试武汉环境适配器...")
    
    # 创建环境
    data_path = os.path.join(project_root, 'data', 'scenario_wuhan_20k.json')
    env = SimpleWuhanAMoDEnv(data_path)
    
    print(f"环境创建成功")
    print(f"区域数量: {env.nregion}")
    print(f"时间步数: {env.tf}")
    print(f"区域列表: {env.region}")
    
    # 测试重置
    obs = env.reset()
    print(f"环境重置成功")
    print(f"观察类型: {type(obs)}")
    
    # 测试一步
    rebAction = {}
    new_obs, reward, done, info = env.reb_step(rebAction)
    print(f"执行一步成功")
    print(f"奖励: {reward}")
    print(f"完成: {done}")
    print(f"信息: {info}")

if __name__ == "__main__":
    test_environment()
