#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
武汉AMoD环境简化训练脚本
使用A2C-GNN算法训练重平衡策略
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
from tqdm import tqdm

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.algos.a2c_gnn import A2C
from src.envs.simple_wuhan_env import SimpleWuhanAMoDEnv

def simple_rebalancing_action(env, target_distribution):
    """
    基于目标分布计算简单的重平衡动作
    """
    # 获取当前车辆分布
    current_distribution = {}
    for i in env.region:
        current_distribution[i] = env.acc[i][env.time]
    
    rebalancing_flow = {}
    
    # target_distribution 是一个列表，需要转换为字典
    target_dict = {}
    for i, region in enumerate(env.region):
        target_dict[region] = target_distribution[i] if i < len(target_distribution) else 0
    
    for i in env.region:
        for j in env.region:
            if i != j:
                # 计算需要重平衡的车辆数量
                current_vehicles = current_distribution.get(i, 0)
                target_vehicles = target_dict.get(i, 0)
                
                if current_vehicles > target_vehicles:
                    excess = current_vehicles - target_vehicles
                    # 简单策略：将多余车辆重平衡到需求最高的区域
                    max_demand_region = max(env.region, 
                                          key=lambda k: sum(env.demand.get((i, k), {}).get(env.time, 0) for i in env.region))
                    if max_demand_region != i:
                        rebalancing_flow[(i, max_demand_region)] = min(excess, 5)  # 限制每次最多重平衡5辆车
    
    return rebalancing_flow

def main():
    parser = argparse.ArgumentParser(description='武汉AMoD环境简化训练')
    parser.add_argument('--max_episodes', type=int, default=100, help='最大训练回合数')
    parser.add_argument('--max_steps', type=int, default=48, help='每回合最大时间步数')
    parser.add_argument('--beta', type=float, default=0.5, help='重平衡成本系数')
    parser.add_argument('--device', type=str, default='cpu', help='计算设备 (cpu/cuda)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("武汉AMoD GNN-RL训练脚本（简化版）")
    print("=" * 60)
    print(f"设备: {args.device.upper()}")
    print(f"最大训练回合: {args.max_episodes}")
    print(f"每回合时间步数: {args.max_steps}")
    print(f"重平衡成本系数: {args.beta}")
    print("=" * 60)
    
    # 设置设备
    device = torch.device(args.device)
    
    # 创建环境
    data_path = os.path.join(project_root, 'data', 'scenario_wuhan_20k.json')
    env = SimpleWuhanAMoDEnv(data_path, beta=args.beta)
    
    print(f"环境初始化完成")
    print(f"区域数量: {env.nregion}, 时间步数: {env.tf}")
    print(f"区域列表: {env.region}")
    
    # 创建A2C模型
    model = A2C(
        env=env,
        input_size=21,  # 修正后的输入维度
        device=device
    )
    
    print(f"\n开始训练...")
    print(f"训练回合: {args.max_episodes}")
    print(f"每回合时间步: {args.max_steps}")
    
    # 训练统计
    episode_rewards = []
    episode_served_demands = []
    episode_rebalancing_costs = []
    
    for episode in tqdm(range(args.max_episodes), desc="训练进度"):
        episode_time_start = time.time()
        
        # 重置环境
        obs = env.reset()
        episode_reward = 0
        episode_served_demand = 0
        episode_rebalancing_cost = 0
        
        for step in range(args.max_steps):
            # 创建简化的观察向量
            obs_vector = []
            
            # 添加当前车辆分布
            for i in env.region:
                obs_vector.append(env.acc[i][env.time])
            
            # 添加时间信息
            obs_vector.append(env.time / env.tf)  # 归一化时间
            
            # 添加需求信息
            for i in env.region:
                total_demand = sum(env.demand.get((i, j), {}).get(env.time, 0) for j in env.region)
                obs_vector.append(total_demand)
            
            # 选择动作 - 传递原始观察数据而不是张量
            target_distribution = model.select_action(obs)
            
            # 基于目标分布计算重平衡动作
            rebAction = simple_rebalancing_action(env, target_distribution)
            
            # 在环境中执行动作
            new_obs, rebreward, done, info = env.reb_step(rebAction)
            episode_reward += rebreward
            
            # 将奖励添加到模型中
            model.rewards.append(rebreward)
            
            # 跟踪回合性能
            episode_served_demand += info.get('served_demand', 0)
            episode_rebalancing_cost += info.get('rebalancing_cost', 0)
            
            # 更新观察
            obs = new_obs
            
            # 如果满足终止条件则停止回合
            if done:
                break
        
        # 执行策略反向传播
        model.training_step()
        
        # 清空奖励和动作列表
        model.rewards = []
        model.saved_actions = []
        
        # 计算总时间
        episode_time_total = time.time() - episode_time_start
        
        # 记录统计
        episode_rewards.append(episode_reward)
        episode_served_demands.append(episode_served_demand)
        episode_rebalancing_costs.append(episode_rebalancing_cost)
        
        # 打印进度
        if (episode + 1) % 10 == 0 or episode == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_served = np.mean(episode_served_demands[-10:])
            avg_reb_cost = np.mean(episode_rebalancing_costs[-10:])
            
            print(f"Episode {episode + 1} | Reward: {episode_reward:.2f} | "
                  f"ServedDemand: {episode_served_demand:.2f} | RebCost: {episode_rebalancing_cost:.2f}")
    
    print(f"\n训练完成!")
    
    # 保存模型
    model_save_path = os.path.join(project_root, 'saved_files', 'ckpt', 'wuhan', 'a2c_gnn_final.pth')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"模型已保存到: {model_save_path}")
    
    # 保存训练日志
    log_save_path = os.path.join(project_root, 'saved_files', 'rl_logs', 'wuhan', 'a2c_gnn_train.pth')
    os.makedirs(os.path.dirname(log_save_path), exist_ok=True)
    training_log = {
        'episode_rewards': episode_rewards,
        'episode_served_demands': episode_served_demands,
        'episode_rebalancing_costs': episode_rebalancing_costs,
        'args': vars(args)
    }
    torch.save(training_log, log_save_path)
    print(f"训练日志已保存到: {log_save_path}")
    
    # 打印最终统计
    print(f"\n训练统计:")
    print(f"平均奖励: {np.mean(episode_rewards):.2f}")
    print(f"平均服务需求: {np.mean(episode_served_demands):.2f}")
    print(f"平均重平衡成本: {np.mean(episode_rebalancing_costs):.2f}")

if __name__ == "__main__":
    main()