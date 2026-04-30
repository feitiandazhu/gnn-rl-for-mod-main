"""
武汉配送环境适配器
将武汉配送数据适配到AMoD环境框架
"""
import json
import numpy as np
from collections import defaultdict
import networkx as nx

class WuhanScenarioAdapter:
    """
    将武汉配送数据适配到AMoD环境
    """
    
    def __init__(self, json_file="data/scenario_wuhan_20k.json"):
        self.json_file = json_file
        self.load_scenario()
        self.create_graph()
        
    def load_scenario(self):
        """加载场景数据"""
        with open(self.json_file, 'r') as f:
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
        
        print(f"加载场景: {self.nregion}个区域, {self.tf}个时间步")
        
    def create_graph(self):
        """创建网络图"""
        self.G = nx.DiGraph()
        
        # 添加节点
        for i, region in enumerate(self.region):
            self.G.add_node(region, accInit=self.acc_init.get(str(region), 0))
        
        # 添加边
        for edge in self.edges:
            i, j = edge[0], edge[1]
            if i != j:  # 避免自环
                self.G.add_edge(i, j, time=1)  # 默认旅行时间为1
        
        print(f"创建图: {self.G.number_of_nodes()}个节点, {self.G.number_of_edges()}条边")
        
    def get_random_demand(self, reset=False):
        """获取随机需求（用于训练）"""
        # 返回tripAttr格式的需求数据
        return self.tripAttr
    
    def get_demand_time_matrix(self):
        """获取需求时间矩阵"""
        # 创建需求时间矩阵
        demand_time = defaultdict(dict)
        
        # tripAttr格式: {"(i,j)": {"t": demand_value}}
        for trip_key, time_data in self.tripAttr.items():
            # 解析trip_key "(i,j)" 格式
            trip_key = trip_key.strip('()')
            i, j = map(int, trip_key.split(','))
            
            # 初始化字典
            if i not in demand_time:
                demand_time[i] = {}
            if j not in demand_time[i]:
                demand_time[i][j] = defaultdict(int)
            
            # 添加时间步需求
            for t_str, demand_value in time_data.items():
                t = int(t_str)
                demand_time[i][j][t] = demand_value
        
        return demand_time
    
    def get_reb_time_matrix(self):
        """获取重平衡时间矩阵"""
        # 创建重平衡时间矩阵
        reb_time = defaultdict(dict)
        
        # tripAttr格式: {"(i,j)": {"t": demand_value}}
        for trip_key, time_data in self.tripAttr.items():
            # 解析trip_key "(i,j)" 格式
            trip_key = trip_key.strip('()')
            i, j = map(int, trip_key.split(','))
            
            # 初始化字典
            if i not in reb_time:
                reb_time[i] = {}
            if j not in reb_time[i]:
                reb_time[i][j] = defaultdict(int)
            
            # 添加时间步重平衡时间（默认1个时间步）
            for t_str in time_data.keys():
                t = int(t_str)
                reb_time[i][j][t] = 1  # 默认重平衡时间为1个时间步
        
        return reb_time


class WuhanAMoDAdapter:
    """
    武汉AMoD环境适配器
    适配到原始AMoD环境接口
    """
    
    def __init__(self, scenario, beta=0.2):
        self.scenario = scenario
        self.beta = beta
        
        # 从scenario获取数据
        self.G = scenario.G
        self.tf = scenario.tf
        self.tstep = scenario.tstep
        self.region = scenario.region
        self.nregion = scenario.nregion
        
        # 初始化需求和时间矩阵
        self.demandTime = scenario.get_demand_time_matrix()
        self.rebTime = scenario.get_reb_time_matrix()
        
        # 初始化状态
        self.time = 0
        self.demand = defaultdict(dict)
        self.price = defaultdict(dict)
        self.acc = defaultdict(dict)
        self.dacc = defaultdict(dict)
        
        # 初始化信息
        self.info = {
            'served_demand': 0,
            'rebalancing_cost': 0,
            'revenue': 0,
            'operating_cost': 0
        }
        
        # 初始化边列表
        self.edges = []
        for i in self.G:
            self.edges.append((i, i))
            for e in self.G.out_edges(i):
                self.edges.append(e)
        self.edges = list(set(self.edges))
        
        print(f"武汉AMoD适配器初始化完成")
        print(f"区域数: {self.nregion}, 时间步: {self.tf}")
        
    def reset(self):
        """重置环境"""
        # 重置状态
        self.acc = defaultdict(dict)
        self.dacc = defaultdict(dict)
        self.rebFlow = defaultdict(dict)
        self.paxFlow = defaultdict(dict)
        self.demand = defaultdict(dict)
        self.price = defaultdict(dict)
        
        # 初始化车辆分布
        for n in self.G:
            # 初始化所有时间步的车辆分布
            for t in range(self.tf + 1):
                self.acc[n][t] = self.G.nodes[n]['accInit']
            self.dacc[n] = defaultdict(float)
        
        # 加载需求数据
        tripAttr = self.scenario.get_random_demand(reset=True)
        self.regionDemand = defaultdict(dict)
        
        # tripAttr格式: {"(i,j)": {"t": demand_value}}
        for trip_key, time_data in tripAttr.items():
            # 解析trip_key "(i,j)" 格式
            trip_key = trip_key.strip('()')
            i, j = map(int, trip_key.split(','))
            
            for t_str, d in time_data.items():
                t = int(t_str)
                self.demand[i, j][t] = d
                self.price[i, j][t] = 1.0  # 默认价格
                if t not in self.regionDemand[i]:
                    self.regionDemand[i][t] = 0
                self.regionDemand[i][t] += d
        
        # 初始化流
        for i, j in self.G.edges:
            self.rebFlow[i, j] = defaultdict(float)
            self.paxFlow[i, j] = defaultdict(float)
        
        # 初始化服务需求
        self.servedDemand = defaultdict(dict)
        for i, j in self.demand:
            self.servedDemand[i, j] = defaultdict(float)
        
        self.time = 0
        self.obs = (self.acc, self.time, self.dacc, self.demand)
        self.reward = 0
        
        return self.obs
    
    def pax_step(self, paxAction=None, CPLEXPATH=None, PATH='', platform='linux'):
        """乘客匹配步骤"""
        t = self.time
        
        # 简化的匹配逻辑
        self.paxAction = defaultdict(float)
        self.reward = 0
        
        # 处理当前时间步的需求
        for i, j in self.demand:
            if t in self.demand[i, j]:
                demand = self.demand[i, j][t]
                available_vehicles = self.acc[i][t]
                
                # 服务需求
                served = min(demand, available_vehicles)
                self.paxAction[i, j] = served
                
                # 更新车辆分布
                self.acc[i][t] -= served
                self.paxFlow[i, j][t] = served
                
                # 计算奖励
                self.reward += served * self.price[i, j][t]
                self.info['served_demand'] += served
                self.info['revenue'] += served * self.price[i, j][t]
        
        self.obs = (self.acc, self.time, self.dacc, self.demand)
        done = False
        
        return self.obs, max(0, self.reward), done, self.info
    
    def reb_step(self, rebAction):
        """重平衡步骤"""
        t = self.time
        self.reward = 0
        self.rebAction = rebAction
        
        # 执行重平衡
        for k in range(len(self.edges)):
            i, j = self.edges[k]
            if (i, j) not in self.G.edges:
                continue
            
            # 更新车辆数量
            self.rebAction[k] = min(self.acc[i][t+1], rebAction[k])
            # 获取重平衡时间，如果不存在则默认为1
            if i in self.rebTime and j in self.rebTime[i]:
                reb_time = self.rebTime[i][j].get(t, 1)
            else:
                reb_time = 1
            self.rebFlow[i, j][t+reb_time] = self.rebAction[k]
            self.acc[i][t+1] -= self.rebAction[k]
            self.dacc[j][t+reb_time] += self.rebFlow[i, j][t+reb_time]
            
            # 计算重平衡成本
            cost = reb_time * self.beta * self.rebAction[k]
            self.info['rebalancing_cost'] += cost
            self.info['operating_cost'] += cost
            self.reward -= cost
        
        # 处理车辆到达
        for k in range(len(self.edges)):
            i, j = self.edges[k]
            if (i, j) in self.rebFlow and t in self.rebFlow[i, j]:
                self.acc[j][t+1] += self.rebFlow[i, j][t]
            if (i, j) in self.paxFlow and t in self.paxFlow[i, j]:
                self.acc[j][t+1] += self.paxFlow[i, j][t]
        
        self.time += 1
        
        # 更新边权重
        for i, j in self.G.edges:
            if i in self.rebTime and j in self.rebTime[i]:
                self.G.edges[i, j]['time'] = self.rebTime[i][j].get(self.time, 1)
            else:
                self.G.edges[i, j]['time'] = 1
        
        done = (self.tf == t+1)
        self.obs = (self.acc, self.time, self.dacc, self.demand)
        
        return self.obs, self.reward, done, self.info
