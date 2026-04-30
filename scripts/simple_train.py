#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的武汉AMoD训练脚本
不使用复杂的GNN，使用简单的神经网络
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.envs.simple_wuhan_env import SimpleWuhanAMoDEnv

class SimpleActorCritic(nn.Module):
    """简单的Actor-Critic网络"""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        
        # 共享特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor网络
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic网络
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value

class SimpleA2C:
    """简化的A2C算法"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, device='cpu'):
        self.device = device
        self.model = SimpleActorCritic(input_dim, hidden_dim, output_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        self.saved_actions = []
        self.rewards = []
        self.gamma = 0.99
        
    def select_action(self, obs):
        """选择动作"""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, value = self.model(obs_tensor)
            
        # 采样动作
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        
        # 保存动作和值
        self.saved_actions.append((action, action_probs, value))
        
        return action.item()
    
    def training_step(self):
        """执行训练步骤"""
        if len(self.rewards) == 0:
            return
            
        # 计算回报
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # 计算损失
        policy_losses = []
        value_losses = []
        
        for (action, action_probs, value), R in zip(self.saved_actions, returns):
            # 策略损失
            dist = torch.distributions.Categorical(action_probs)
            log_prob = dist.log_prob(action)
            policy_losses.append(-log_prob * (R - value.item()))
            
            # 价值损失
            value_losses.append(F.mse_loss(value, R.unsqueeze(0)))
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        total_loss.backward()
        self.optimizer.step()
        
        # 清空缓冲区
        self.saved_actions = []
        self.rewards = []

def simple_rebalancing_action(env, action):
    """基于动作计算重平衡"""
    # 简单的重平衡策略：将车辆从车辆多的区域移动到车辆少的区域
    current_dist = env.get_current_distribution()
    
    # 按车辆数量排序区域
    sorted_regions = sorted(current_dist.items(), key=lambda x: x[1], reverse=True)
    
    rebalancing_flow = {}
    
    # 从车辆最多的区域重平衡到车辆最少的区域
    if len(sorted_regions) >= 2:
        source_region = sorted_regions[0][0]
        target_region = sorted_regions[-1][0]
        
        if source_region != target_region and current_dist[source_region] > 5:
            rebalancing_flow[(source_region, target_region)] = min(5, current_dist[source_region] - 5)
    
    return rebalancing_flow

def main():
    parser = argparse.ArgumentParser(description='简化武汉AMoD训练')
    parser.add_argument('--max_episodes', type=int, default=100, help='最大训练回合数')
    parser.add_argument('--max_steps', type=int, default=48, help='每回合最大时间步数')
    parser.add_argument('--beta', type=float, default=0.5, help='重平衡成本系数')
    parser.add_argument('--device', type=str, default='cpu', help='计算设备 (cpu/cuda)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("简化武汉AMoD训练脚本")
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
    
    # 创建A2C模型
    input_dim = env.nregion + 1  # 车辆分布 + 时间
    hidden_dim = 64
    output_dim = env.nregion
    
    model = SimpleA2C(input_dim, hidden_dim, output_dim, device)
    
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
            # 创建观察向量
            obs_vector = []
            
            # 添加当前车辆分布
            current_dist = env.get_current_distribution()
            for region in env.region:
                obs_vector.append(current_dist[region])
            
            # 添加时间信息
            obs_vector.append(env.time / env.tf)
            
            # 选择动作
            action = model.select_action(obs_vector)
            
            # 基于动作计算重平衡
            rebAction = simple_rebalancing_action(env, action)
            
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
    model_save_path = os.path.join(project_root, 'saved_files', 'ckpt', 'wuhan', 'simple_a2c_final.pth')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.model.state_dict(), model_save_path)
    print(f"模型已保存到: {model_save_path}")
    
    # 保存训练日志
    log_save_path = os.path.join(project_root, 'saved_files', 'rl_logs', 'wuhan', 'simple_a2c_train.pth')
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
