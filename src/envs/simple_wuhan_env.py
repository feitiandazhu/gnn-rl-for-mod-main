#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的武汉AMoD环境适配器
基于原始AMoD环境，适配武汉数据
"""

import os
import sys
import json
import numpy as np
import torch
from collections import defaultdict
import networkx as nx

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

class SimpleWuhanAMoDEnv:
    """
    简化的武汉AMoD环境
    """
    
    def __init__(self, data_path, beta=0.5):
        self.data_path = data_path
        self.beta = beta
        self.load_data()
        self.initialize_environment()
        
    def load_data(self):
        """加载武汉场景数据"""
        with open(self.data_path, 'r') as f:
            self.data = json.load(f)
        
        # 提取基本信息
        self.tf = self.data['tf']  # 时间步数
        self.tstep = self.data['tstep']  # 时间步长（分钟）
        self.edges = self.data['edges']  # 边列表
        self.acc_init = self.data['acc_init']  # 初始车辆分布
        self.tripAttr = self.data['tripAttr']  # 订单属性
        
        # 创建区域列表
        regions = set()
        for edge in self.edges:
            regions.add(edge[0])
            regions.add(edge[1])
        self.region = sorted(list(regions))
        self.nregion = len(self.region)
        
        print(f"加载武汉场景: {self.nregion}个区域, {self.tf}个时间步")
        
    def initialize_environment(self):
        """初始化环境状态"""
        # 创建网络图
        self.G = nx.DiGraph()
        
        # 添加节点
        for region in self.region:
            self.G.add_node(region, accInit=self.acc_init.get(str(region), 0))
        
        # 添加边
        for edge in self.edges:
            i, j = edge[0], edge[1]
            if i != j:  # 避免自环
                self.G.add_edge(i, j, time=1)  # 默认旅行时间为1
        
        # 初始化状态变量
        self.acc = defaultdict(dict)  # 车辆分布
        self.dacc = defaultdict(dict)  # 到达车辆
        self.demand = defaultdict(dict)  # 需求
        self.price = defaultdict(dict)  # 价格
        self.rebFlow = defaultdict(dict)  # 重平衡流
        self.paxFlow = defaultdict(dict)  # 乘客流
        
        # 初始化需求数据
        for trip_key, time_data in self.tripAttr.items():
            # 解析trip_key "(i,j)" 格式
            trip_key = trip_key.strip('()')
            i, j = map(int, trip_key.split(','))
            
            for t_str, demand_value in time_data.items():
                t = int(t_str)
                self.demand[i, j][t] = demand_value
                self.price[i, j][t] = 1.0  # 默认价格
        
        # 初始化流
        for i, j in self.G.edges:
            self.rebFlow[i, j] = defaultdict(float)
            self.paxFlow[i, j] = defaultdict(float)
        
        print(f"环境初始化完成: {self.G.number_of_nodes()}个节点, {self.G.number_of_edges()}条边")
        
    def reset(self):
        """重置环境"""
        self.time = 0
        
        # 重置车辆分布
        self.acc = defaultdict(dict)
        self.dacc = defaultdict(dict)
        
        # 初始化所有时间步的车辆分布
        for region in self.region:
            for t in range(self.tf + 1):
                self.acc[region][t] = self.G.nodes[region]['accInit']
            self.dacc[region] = defaultdict(float)
        
        # 重置流
        self.rebFlow = defaultdict(dict)
        self.paxFlow = defaultdict(dict)
        for i, j in self.G.edges:
            self.rebFlow[i, j] = defaultdict(float)
            self.paxFlow[i, j] = defaultdict(float)
        
        # 重置信息
        self.info = {
            'revenue': 0,
            'served_demand': 0,
            'rebalancing_cost': 0,
            'operating_cost': 0
        }
        
        self.reward = 0
        self.obs = (self.acc, self.time, self.dacc, self.demand)
        
        return self.obs
    
    def reb_step(self, rebAction):
        """重平衡步骤"""
        t = self.time
        self.reward = 0
        
        # 处理乘客需求
        served_demand = 0
        for i, j in self.demand:
            if t in self.demand[i, j]:
                demand = self.demand[i, j][t]
                available_vehicles = self.acc[i][t]
                
                # 服务需求
                served = min(demand, available_vehicles)
                self.paxFlow[i, j][t] = served
                
                # 更新车辆分布
                self.acc[i][t] -= served
                
                # 计算奖励
                self.reward += served * self.price[i, j][t]
                served_demand += served
        
        # 执行重平衡
        rebalancing_cost = 0
        for (i, j), flow in rebAction.items():
            if (i, j) in self.G.edges:
                # 限制重平衡车辆数量
                max_reb = min(self.acc[i][t], flow)
                self.rebFlow[i, j][t] = max_reb
                self.acc[i][t] -= max_reb
                
                # 计算重平衡成本
                reb_time = self.G.edges[i, j]['time']
                cost = reb_time * self.beta * max_reb
                rebalancing_cost += cost
                self.reward -= cost
        
        # 处理车辆到达
        for i, j in self.G.edges:
            if t in self.rebFlow[i, j]:
                self.acc[j][t+1] += self.rebFlow[i, j][t]
            if t in self.paxFlow[i, j]:
                self.acc[j][t+1] += self.paxFlow[i, j][t]
        
        self.time += 1
        
        # 更新信息
        self.info['served_demand'] = served_demand
        self.info['rebalancing_cost'] = rebalancing_cost
        self.info['revenue'] = self.reward
        
        done = (self.time >= self.tf)
        self.obs = (self.acc, self.time, self.dacc, self.demand)
        
        return self.obs, self.reward, done, self.info
    
    def get_current_distribution(self):
        """获取当前车辆分布"""
        current_dist = {}
        for region in self.region:
            current_dist[region] = self.acc[region][self.time]
        return current_dist
    
    def get_demand_at_region(self, region):
        """获取指定区域的需求"""
        total_demand = 0
        for j in self.region:
            if (region, j) in self.demand and self.time in self.demand[region, j]:
                total_demand += self.demand[region, j][self.time]
        return total_demand
