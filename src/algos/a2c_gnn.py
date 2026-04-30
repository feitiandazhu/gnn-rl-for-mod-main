"""
A2C-GNN
-------
This file contains the A2C-GNN specifications. In particular, we implement:
(1) GNNParser
    Converts raw environment observations to agent inputs (s_t).
(2) GNNActor:
    Policy parametrized by Graph Convolution Networks (Section III-C in the paper)
(3) GNNCritic:
    Critic parametrized by Graph Convolution Networks (Section III-C in the paper)
(4) A2C:
    Advantage Actor Critic algorithm using a GNN parametrization for both Actor and Critic.
"""

import numpy as np 
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.utils import grid
from collections import namedtuple

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
args = namedtuple('args', ('render', 'gamma', 'log_interval'))
args.render= True
args.gamma = 0.97
args.log_interval = 10

#########################################
############## PARSER ###################
#########################################

class GNNParser():
    """
    Parser converting raw environment observations to agent inputs (s_t).
    """
    def __init__(self, env, T=10, grid_h=8, grid_w=8, scale_factor=0.01):
        super().__init__()
        self.env = env
        self.T = T
        self.s = scale_factor
        self.grid_h = grid_h
        self.grid_w = grid_w
        
    def parse_obs(self, obs):
        # 获取区域数量
        nregion = len(self.env.region)
        
        # 构建特征矩阵
        features = []
        
        # 当前车辆分布特征
        # obs[0] 是 acc 字典，obs[1] 是当前时间
        current_time = obs[1]
        current_acc = torch.tensor([obs[0][n][current_time]*self.s for n in self.env.region]).view(1, 1, nregion).float()
        features.append(current_acc)
        
        # 未来车辆到达特征
        future_acc_list = []
        for t in range(current_time+1, min(current_time+self.T+1, self.env.tf)):
            future_acc_t = []
            for n in self.env.region:
                # 当前车辆 + 未来到达车辆
                future_val = obs[0][n][current_time] * self.s
                if n in obs[2] and t in obs[2][n]:  # obs[2] 是 dacc
                    future_val += obs[2][n][t] * self.s
                future_acc_t.append(future_val)
            future_acc_list.append(future_acc_t)
        
        if future_acc_list:
            future_acc = torch.tensor(future_acc_list).view(1, -1, nregion).float()
            features.append(future_acc)
        
        # 需求特征（如果可用）
        try:
            demand_features_list = []
            for t in range(current_time+1, min(current_time+self.T+1, self.env.tf)):
                demand_t = []
                for i in self.env.region:
                    total_demand = 0
                    for j in self.env.region:
                        if (i, j) in obs[3] and t in obs[3][i, j]:  # obs[3] 是 demand
                            total_demand += obs[3][i, j][t] * self.s
                    demand_t.append(total_demand)
                demand_features_list.append(demand_t)
            
            if demand_features_list:
                demand_features = torch.tensor(demand_features_list).view(1, -1, nregion).float()
                features.append(demand_features)
        except:
            # 如果需求数据不可用，使用零矩阵
            time_horizon = min(self.T, self.env.tf-current_time)
            if time_horizon > 0:
                demand_features = torch.zeros(1, time_horizon, nregion).float()
                features.append(demand_features)
        
        # 拼接所有特征
        if len(features) > 1:
            x = torch.cat(features, dim=1).squeeze(0).view(-1, nregion).T
        else:
            x = features[0].squeeze(0).view(-1, nregion).T
        
        # 确保特征维度正确
        if x.shape[0] != nregion:
            x = x.T
        
        # 创建图结构 - 使用全连接图或基于距离的图
        edge_index = self._create_edge_index(nregion)
        data = Data(x, edge_index)
        return data
    
    def _create_edge_index(self, nregion):
        """创建边索引，可以是全连接图或基于距离的图"""
        # 简单全连接图
        edges = []
        for i in range(nregion):
            for j in range(nregion):
                if i != j:
                    edges.append([i, j])
        
        if len(edges) == 0:
            # 如果没有边，创建自环
            edges = [[i, i] for i in range(nregion)]
        
        return torch.tensor(edges).T
    
#########################################
############## ACTOR ####################
#########################################
class GNNActor(nn.Module):
    """
    Actor \pi(a_t | s_t) 输出目标车辆分布
    输入：当前状态（车辆分布、需求等）
    输出：每个区域的目标车辆数量
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = GCNConv(in_channels, in_channels)
        self.conv2 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, 64)
        self.lin2 = nn.Linear(64, 32)
        self.lin3 = nn.Linear(32, 1)
        
        # 添加dropout防止过拟合
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, data):
        # 第一层GCN
        out1 = F.relu(self.conv1(data.x, data.edge_index))
        x = out1 + data.x
        
        # 第二层GCN
        out2 = F.relu(self.conv2(x, data.edge_index))
        x = out2 + x
        
        # 全连接层
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = F.relu(self.lin2(x))
        x = self.dropout(x)
        
        # 输出目标车辆数量（使用softplus确保非负）
        x = F.softplus(self.lin3(x)) + 1e-6  # 避免零值
        
        return x

#########################################
############## CRITIC ###################
#########################################

class GNNCritic(nn.Module):
    """
    Critic parametrizing the value function estimator V(s_t).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, 32)
        self.lin2 = nn.Linear(32, 32)
        self.lin3 = nn.Linear(32, 1)
    
    def forward(self, data):
        out = F.relu(self.conv1(data.x, data.edge_index))
        x = out + data.x 
        x = torch.sum(x, dim=0)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x

#########################################
############## A2C AGENT ################
#########################################

class A2C(nn.Module):
    """
    Advantage Actor Critic algorithm for the AMoD control problem. 
    """
    def __init__(self, env, input_size, eps=np.finfo(np.float32).eps.item(), device=torch.device("cpu")):
        super(A2C, self).__init__()
        self.env = env
        self.eps = eps
        self.input_size = input_size
        self.hidden_size = input_size
        self.device = device
        
        self.actor = GNNActor(self.input_size, self.hidden_size)
        self.critic = GNNCritic(self.input_size, self.hidden_size)
        self.obs_parser = GNNParser(self.env)
        
        self.optimizers = self.configure_optimizers()
        
        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.to(self.device)
        
    def forward(self, obs, jitter=1e-20):
        """
        forward of both actor and critic
        """
        # parse raw environment data in model format
        x = self.parse_obs(obs).to(self.device)
        
        # actor: 输出目标车辆分布
        target_vehicles = self.actor(x).reshape(-1)
        
        # critic: estimates V(s_t)
        value = self.critic(x)
        return target_vehicles, value
    
    def parse_obs(self, obs):
        state = self.obs_parser.parse_obs(obs)
        return state
    
    def select_action(self, obs):
        """
        选择动作：输出目标车辆分布
        返回：每个区域的目标车辆数量列表
        """
        target_vehicles, value = self.forward(obs)
        
        # 添加噪声进行探索（训练时）
        if self.training:
            noise = torch.randn_like(target_vehicles) * 0.1
            target_vehicles = target_vehicles + noise
            target_vehicles = torch.clamp(target_vehicles, min=0.1)  # 确保非负
        
        # 保存动作和值用于训练
        self.saved_actions.append(SavedAction(target_vehicles.sum(), value))
        
        return list(target_vehicles.cpu().detach().numpy())

    def training_step(self):
        R = 0
        saved_actions = self.saved_actions
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (value) loss
        returns = [] # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in self.rewards[::-1]:
            # calculate the discounted value
            R = r + args.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        for (action_sum, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # 简化的策略损失：基于动作总和的损失
            # 这里我们使用MSE损失来训练actor输出合理的目标车辆分布
            target_action_sum = torch.tensor([R.item()]).to(self.device)
            policy_losses.append(F.mse_loss(action_sum, target_action_sum))

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).to(self.device)))

        # take gradient steps
        self.optimizers['a_optimizer'].zero_grad()
        a_loss = torch.stack(policy_losses).sum()
        a_loss.backward()
        self.optimizers['a_optimizer'].step()
        
        self.optimizers['c_optimizer'].zero_grad()
        v_loss = torch.stack(value_losses).sum()
        v_loss.backward()
        self.optimizers['c_optimizer'].step()
        
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]
    
    def predict_target_distribution(self, obs):
        """
        预测目标车辆分布（用于实时预测）
        输入：当前观测状态
        输出：每个区域的目标车辆数量
        """
        self.eval()  # 设置为评估模式
        with torch.no_grad():
            target_vehicles, _ = self.forward(obs)
            return list(target_vehicles.cpu().numpy())
    
    def get_current_distribution(self, obs):
        """
        获取当前车辆分布
        """
        acc, time, dacc, demand = obs
        current_dist = []
        for region in self.env.region:
            current_dist.append(acc[region][time+1])
        return current_dist
    
    def configure_optimizers(self):
        optimizers = dict()
        actor_params = list(self.actor.parameters())
        critic_params = list(self.critic.parameters())
        optimizers['a_optimizer'] = torch.optim.Adam(actor_params, lr=1e-3)
        optimizers['c_optimizer'] = torch.optim.Adam(critic_params, lr=1e-3)
        return optimizers
    
    def save_checkpoint(self, path='ckpt.pth'):
        checkpoint = dict()
        checkpoint['model'] = self.state_dict()
        for key, value in self.optimizers.items():
            checkpoint[key] = value.state_dict()
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path='ckpt.pth'):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model'])
        for key, value in self.optimizers.items():
            self.optimizers[key].load_state_dict(checkpoint[key])
    
    def log(self, log_dict, path='log.pth'):
        torch.save(log_dict, path)