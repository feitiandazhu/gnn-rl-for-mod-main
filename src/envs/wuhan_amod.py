"""
武汉配送AMoD环境
基于真实武汉配送数据的自动驾驶出行服务环境
"""
import numpy as np
import pandas as pd
import json
from collections import defaultdict


class WuhanAMoDEnv:
    """
    武汉配送AMoD环境
    
    参数:
    - num_regions: 区域数量（16个）
    - num_vehicles: 车辆总数（44辆）
    - num_timesteps: 时间步总数（60步）
    - data_dir: 数据目录路径
    """
    
    def __init__(self, data_dir='../../../配送数据/processed'):
        # 加载配置
        with open(f'{data_dir}/config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.num_regions = config['num_regions']
        self.num_vehicles = config['num_vehicles']
        self.num_timesteps = config['num_timesteps']
        self.minutes_per_step = config['minutes_per_step']
        
        # 加载数据
        self.region_centers = pd.read_csv(f'{data_dir}/region_centers.csv')
        self.distance_matrix = pd.read_csv(f'{data_dir}/region_distance_matrix.csv').values
        self.demand_matrix = pd.read_csv(f'{data_dir}/demand_matrix.csv').values
        self.vehicle_allocation = pd.read_csv(f'{data_dir}/vehicle_allocation.csv')
        
        # 环境状态
        self.current_timestep = 0
        self.vehicle_positions = None  # 每辆车当前位置
        self.vehicle_status = None  # 每辆车状态（空闲/服务中）
        self.served_orders = 0
        self.total_wait_time = 0
        self.total_travel_distance = 0
        
        # 统计信息
        self.episode_stats = {
            'served_orders': 0,
            'unserved_orders': 0,
            'total_wait_time': 0,
            'total_travel_distance': 0,
            'avg_wait_time': 0,
            'service_rate': 0
        }
        
        print(f"武汉AMoD环境初始化:")
        print(f"  - 区域数: {self.num_regions}")
        print(f"  - 车辆数: {self.num_vehicles}")
        print(f"  - 时间步: {self.num_timesteps}")
        print(f"  - 总需求: {self.demand_matrix.sum()}")
    
    def reset(self):
        """重置环境"""
        self.current_timestep = 0
        
        # 初始化车辆位置（按分配方案）
        self.vehicle_positions = []
        for region_id in range(self.num_regions):
            num_vehicles = int(self.vehicle_allocation.iloc[region_id]['num_vehicles'])
            self.vehicle_positions.extend([region_id] * num_vehicles)
        self.vehicle_positions = np.array(self.vehicle_positions)
        
        # 初始化车辆状态（0=空闲, 1=服务中）
        self.vehicle_status = np.zeros(self.num_vehicles, dtype=int)
        
        # 重置统计
        self.served_orders = 0
        self.total_wait_time = 0
        self.total_travel_distance = 0
        
        self.episode_stats = {
            'served_orders': 0,
            'unserved_orders': 0,
            'total_wait_time': 0,
            'total_travel_distance': 0,
            'avg_wait_time': 0,
            'service_rate': 0
        }
        
        return self._get_state()
    
    def _get_state(self):
        """
        获取当前状态
        返回: dict包含各种状态信息
        """
        # 计算每个区域的车辆数
        vehicle_distribution = np.zeros(self.num_regions)
        for pos in self.vehicle_positions:
            vehicle_distribution[pos] += 1
        
        # 当前时间步的需求
        current_demand = self.demand_matrix[self.current_timestep] if self.current_timestep < self.num_timesteps else np.zeros(self.num_regions)
        
        # 未来几个时间步的需求（用于预测）
        future_demand = np.zeros(self.num_regions)
        for t in range(self.current_timestep + 1, min(self.current_timestep + 5, self.num_timesteps)):
            future_demand += self.demand_matrix[t]
        
        state = {
            'vehicle_distribution': vehicle_distribution,
            'current_demand': current_demand,
            'future_demand': future_demand,
            'timestep': self.current_timestep,
            'vehicle_positions': self.vehicle_positions.copy(),
            'vehicle_status': self.vehicle_status.copy()
        }
        
        return state
    
    def step(self, actions):
        """
        执行一步
        
        参数:
        - actions: 车辆调度动作，shape=(num_vehicles,)，每个元素是目标区域ID
        
        返回:
        - next_state: 下一个状态
        - reward: 奖励
        - done: 是否结束
        - info: 额外信息
        """
        # 获取当前需求
        current_demand = self.demand_matrix[self.current_timestep].copy()
        
        # 执行车辆调度
        rebalancing_cost = 0
        for vehicle_id, target_region in enumerate(actions):
            if self.vehicle_status[vehicle_id] == 0:  # 只调度空闲车辆
                current_pos = self.vehicle_positions[vehicle_id]
                if target_region != current_pos:
                    # 计算调度成本（距离）
                    distance = self.distance_matrix[current_pos, target_region]
                    rebalancing_cost += distance
                    # 移动车辆
                    self.vehicle_positions[vehicle_id] = target_region
        
        # 匹配订单和车辆
        served_this_step = 0
        unserved_this_step = 0
        
        for region_id in range(self.num_regions):
            demand = int(current_demand[region_id])
            # 计算该区域可用车辆数
            available_vehicles = np.sum(
                (self.vehicle_positions == region_id) & (self.vehicle_status == 0)
            )
            
            # 服务订单
            served = min(demand, available_vehicles)
            served_this_step += served
            unserved_this_step += (demand - served)
            
            # 更新统计
            self.served_orders += served
            self.total_wait_time += (demand - served) * self.minutes_per_step
        
        # 计算奖励
        service_reward = served_this_step * 100  # 每服务一个订单奖励100
        unserved_penalty = unserved_this_step * -50  # 每未服务订单惩罚50
        rebalancing_penalty = -rebalancing_cost / 1000  # 调度成本惩罚（归一化）
        
        reward = service_reward + unserved_penalty + rebalancing_penalty
        
        # 更新统计
        self.episode_stats['served_orders'] += served_this_step
        self.episode_stats['unserved_orders'] += unserved_this_step
        self.episode_stats['total_travel_distance'] += rebalancing_cost
        
        # 前进到下一时间步
        self.current_timestep += 1
        
        # 检查是否结束
        done = self.current_timestep >= self.num_timesteps
        
        if done:
            # 计算最终统计
            total_demand = self.demand_matrix.sum()
            self.episode_stats['service_rate'] = self.episode_stats['served_orders'] / total_demand if total_demand > 0 else 0
            self.episode_stats['avg_wait_time'] = self.episode_stats['total_wait_time'] / total_demand if total_demand > 0 else 0
        
        # 获取下一状态
        next_state = self._get_state()
        
        info = {
            'served': served_this_step,
            'unserved': unserved_this_step,
            'rebalancing_cost': rebalancing_cost,
            'timestep': self.current_timestep - 1
        }
        
        return next_state, reward, done, info
    
    def get_adjacency_matrix(self):
        """
        获取区域邻接矩阵（用于GNN）
        基于距离构建邻接关系
        """
        # 使用距离的倒数作为邻接权重
        adj_matrix = 1.0 / (self.distance_matrix + 1e-6)
        # 归一化
        adj_matrix = adj_matrix / adj_matrix.max()
        # 对角线设为1
        np.fill_diagonal(adj_matrix, 1.0)
        return adj_matrix
    
    def get_episode_stats(self):
        """获取episode统计信息"""
        return self.episode_stats.copy()
    
    def render(self):
        """可视化当前状态"""
        state = self._get_state()
        print(f"\n时间步 {self.current_timestep}/{self.num_timesteps}")
        print(f"车辆分布: {state['vehicle_distribution']}")
        print(f"当前需求: {state['current_demand']}")
        print(f"已服务订单: {self.served_orders}")
        print(f"服务率: {self.episode_stats['service_rate']:.2%}")


def test_wuhan_env():
    """测试武汉AMoD环境"""
    print("=" * 80)
    print("测试武汉AMoD环境")
    print("=" * 80)
    
    # 创建环境
    env = WuhanAMoDEnv()
    
    # 重置环境
    state = env.reset()
    print(f"\n初始状态:")
    print(f"  - 车辆分布: {state['vehicle_distribution']}")
    print(f"  - 当前需求: {state['current_demand']}")
    
    # 运行几个时间步
    total_reward = 0
    for t in range(5):
        # 简单策略：车辆留在原地
        actions = env.vehicle_positions.copy()
        
        next_state, reward, done, info = env.step(actions)
        total_reward += reward
        
        print(f"\n时间步 {t}:")
        print(f"  - 服务订单: {info['served']}")
        print(f"  - 未服务订单: {info['unserved']}")
        print(f"  - 奖励: {reward:.2f}")
        print(f"  - 累计奖励: {total_reward:.2f}")
        
        if done:
            break
    
    print(f"\n最终统计:")
    stats = env.get_episode_stats()
    for key, value in stats.items():
        print(f"  - {key}: {value}")


if __name__ == '__main__':
    test_wuhan_env()

