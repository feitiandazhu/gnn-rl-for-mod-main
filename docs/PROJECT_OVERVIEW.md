# GNN-RL for AMoD (Autonomous Mobility on Demand)

基于图神经网络和强化学习的自主移动服务优化系统

## 项目结构

```
gnn-rl-for-amod-main/
├── src/                    # 源代码
│   ├── algos/             # 算法实现
│   │   ├── a2c_gnn.py     # A2C-GNN算法
│   │   └── reb_flow_solver.py  # 重平衡流求解器
│   ├── envs/              # 环境实现
│   │   ├── amod_env.py    # AMoD环境基类
│   │   └── wuhan_env_adapter.py  # 武汉环境适配器
│   ├── cplex_mod/         # CPLEX模型文件
│   └── misc/              # 工具函数
├── scripts/               # 训练脚本
│   └── train_wuhan_simple.py  # 武汉环境训练脚本
├── examples/              # 示例代码
│   └── quick_test.py      # 快速测试示例
├── configs/               # 配置文件
│   └── training_config.py # 训练配置
├── data/                  # 数据文件
│   ├── scenario_wuhan_20k.json  # 武汉场景数据
│   ├── raw/               # 原始数据
│   └── region_visualization.png  # 区域可视化
├── saved_files/           # 保存文件
│   ├── ckpt/              # 模型检查点
│   └── rl_logs/           # 训练日志
├── docs/                  # 文档
├── requirements.txt       # 依赖包
└── README.md             # 项目说明
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 快速测试

```bash
cd gnn-rl-for-amod-main
python examples/quick_test.py
```

### 3. 完整训练

```bash
cd gnn-rl-for-amod-main
python scripts/train_wuhan_simple.py --max_episodes 1000 --max_steps 48
```

## 主要特性

- **A2C-GNN算法**: 基于Actor-Critic和图神经网络的强化学习算法
- **武汉AMoD环境**: 针对武汉地区的自主移动服务环境模拟
- **重平衡策略**: 智能车辆重平衡优化
- **模块化设计**: 清晰的代码结构，易于扩展

## 配置说明

训练参数可在 `configs/training_config.py` 中修改：

- `max_episodes`: 最大训练回合数
- `max_steps_per_episode`: 每回合最大时间步数
- `beta`: 重平衡成本系数
- `learning_rate`: 学习率
- `device`: 计算设备 (cpu/cuda)

## 数据说明

- `scenario_wuhan_20k.json`: 包含20,000个订单的武汉场景数据
- 数据格式: 16个区域，48个时间步
- 包含需求矩阵、重平衡时间矩阵等信息

## 训练输出

训练完成后会生成：
- `saved_files/ckpt/wuhan/a2c_gnn_final.pth`: 训练好的模型
- `saved_files/rl_logs/wuhan/a2c_gnn_train.pth`: 训练日志

## 性能指标

- **平均奖励**: 每回合的平均奖励值
- **服务需求**: 满足的乘客需求数量
- **重平衡成本**: 车辆重平衡的成本

## 扩展开发

1. **新环境**: 在 `src/envs/` 中添加新的环境适配器
2. **新算法**: 在 `src/algos/` 中实现新的强化学习算法
3. **新配置**: 在 `configs/` 中添加新的配置文件
4. **新示例**: 在 `examples/` 中添加新的使用示例
