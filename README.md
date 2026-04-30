# Graph Neural Network Reinforcement Learning for AMoD Systems

基于图神经网络和强化学习的自主移动服务优化系统 - 武汉版本

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 快速测试（5个episodes）
```bash
python examples/quick_test.py
```

### 3. 完整训练
```bash
python scripts/train_wuhan_simple.py --max_episodes 1000 --max_steps 48
```

## 📁 项目结构

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
├── examples/              # 示例代码
├── configs/               # 配置文件
├── data/                  # 数据文件
│   ├── scenario_wuhan_20k.json  # 武汉场景数据
│   └── raw/               # 原始数据
├── saved_files/           # 保存文件
│   ├── ckpt/              # 模型检查点
│   └── rl_logs/           # 训练日志
└── docs/                  # 文档
```

## 🎯 主要特性

- **A2C-GNN算法**: 基于Actor-Critic和图神经网络的强化学习算法
- **武汉AMoD环境**: 针对武汉地区的自主移动服务环境模拟
- **重平衡策略**: 智能车辆重平衡优化
- **模块化设计**: 清晰的代码结构，易于扩展

## ⚙️ 配置说明

训练参数可在 `configs/training_config.py` 中修改：

- `max_episodes`: 最大训练回合数
- `max_steps_per_episode`: 每回合最大时间步数  
- `beta`: 重平衡成本系数
- `learning_rate`: 学习率
- `device`: 计算设备 (cpu/cuda)

## 📊 数据说明

- `scenario_wuhan_20k.json`: 包含20,000个订单的武汉场景数据
- 数据格式: 16个区域，48个时间步
- 包含需求矩阵、重平衡时间矩阵等信息

## 📈 训练输出

训练完成后会生成：
- `saved_files/ckpt/wuhan/a2c_gnn_final.pth`: 训练好的模型
- `saved_files/rl_logs/wuhan/a2c_gnn_train.pth`: 训练日志

## 🔧 原始实现

本项目基于原始论文 [Graph Neural Network Reinforcement Learning for Autonomous Mobility-on-Demand Systems](https://arxiv.org/abs/2104.11434) 的实现，并针对武汉地区进行了适配和优化。

## Examples

To train an agent, `main.py` accepts the following arguments:
```bash
cplex arguments:
    --cplexpath     defines directory of the CPLEX installation
    
model arguments:
    --test          activates agent evaluation mode (default: False)
    --max_episodes  number of episodes to train agent (default: 16k)
    --max_steps     number of steps per episode (default: T=60)
    --no-cuda       disables CUDA training (default: True, i.e. run on CPU)
    --directory     defines directory where to log files (default: saved_files)
    
simulator arguments: (unless necessary, we recommend using the provided ones)
    --seed          random seed (default: 10)
    --demand_ratio  (default: 0.5)
    --json_hr       (default: 7)
    --json_tsetp    (default: 3)
    --no-beta       (default: 0.5)
```

**Important**: Take care of specifying the correct path for your local CPLEX installation. Typical default paths based on different operating systems could be the following
```bash
Windows: "C:/Program Files/ibm/ILOG/CPLEX_Studio128/opl/bin/x64_win64/"
OSX: "/Applications/CPLEX_Studio128/opl/bin/x86-64_osx/"
Linux: "/opt/ibm/ILOG/CPLEX_Studio128/opl/bin/x86-64_linux/"
```
### Training and simulating an agent

1. To train an agent (with the default parameters) run the following:
```
python main.py
```

2. To evaluate a pretrained agent run the following:
```
python main.py --test=True
```

## Credits
This work was conducted as a joint effort with [Kaidi Yang*](https://sites.google.com/site/kdyang1990/), [James Harrison*](https://stanford.edu/~jh2/), [Filipe Rodrigues'](http://fprodrigues.com/), [Francisco C. Pereira'](http://camara.scripts.mit.edu/home/) and [Marco Pavone*](https://web.stanford.edu/~pavone/), at Technical University of Denmark' and Stanford University*. 

## Reference
```
@inproceedings{GammelliYangEtAl2021,
  author = {Gammelli, D. and Yang, K. and Harrison, J. and Rodrigues, F. and Pereira, F. C. and Pavone, M.},
  title = {Graph Neural Network Reinforcement Learning for Autonomous Mobility-on-Demand Systems},
  year = {2021},
  note = {Submitted},
}
```

----------
In case of any questions, bugs, suggestions or improvements, please feel free to contact me at daga@dtu.dk.
