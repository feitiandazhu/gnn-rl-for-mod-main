# 训练配置
TRAINING_CONFIG = {
    # 环境参数
    "num_regions": 16,
    "num_timesteps": 48,
    "num_vehicles": 200,
    "beta": 0.5,  # 重平衡成本系数
    
    # 训练参数
    "max_episodes": 1000,
    "max_steps_per_episode": 48,
    "learning_rate": 0.001,
    "gamma": 0.99,  # 折扣因子
    
    # 模型参数
    "input_dim": 21,
    "hidden_dim": 64,
    "output_dim": 16,
    
    # 设备
    "device": "cpu",  # 或 "cuda"
    
    # 数据路径
    "data_path": "data/scenario_wuhan_20k.json",
    
    # 保存路径
    "model_save_path": "saved_files/ckpt/wuhan/a2c_gnn_final.pth",
    "log_save_path": "saved_files/rl_logs/wuhan/a2c_gnn_train.pth"
}

# 环境配置
ENV_CONFIG = {
    "demand_matrix_shape": (16, 16, 48),
    "reb_time_matrix_shape": (16, 16),
    "reward_scale": 1.0,
    "penalty_scale": 0.1
}

# 模型配置
MODEL_CONFIG = {
    "activation": "relu",
    "dropout": 0.1,
    "batch_norm": True,
    "optimizer": "adam",
    "weight_decay": 1e-4
}
