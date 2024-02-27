# RL
Reinforcement learning

- 记录强化学习自学的相关东西
  - github pag [https://scchy.github.io/RL](https://scchy.github.io/RL)
- 强化学习实践的项目
- 强化学习的框架 在 [src 目录](./src/)下


# 框架简介
目录
```text
src
├── requestment.txt
├── RLAlgo
│   ├── _base_net.py
│   ├── DDPG.py
│   ├── DQN.py
│   ├── PPO2.py
│   ├── PPO.py
│   ├── __pycache__
│   ├── SAC.py
│   └── TD3.py
├── RLUtils
│   ├── config.py
│   ├── env_wrapper.py
│   ├── __init__.py
│   ├── memory.py
│   ├── __pycache__
│   ├── state_util.py
│   └── trainer.py
├── test
│   ├── border_detector.py
│   ├── __dqn.log
│   ├── test_ddpg.py
│   ├── test_dqn.py
│   ├── test_env_explore.ipynb
│   ├── test_env.md
│   ├── test_models
│   ├── test_ppo.py
│   ├── test_sac.py
│   └── test_TD3.py
└── TODO.md
```

## 环境要求

核心包

| package | version |
|--|--|
|python版本 | `Python 3.10`|
|torch | 2.1.1|
|torchvision | 0.16.1|
|gymnasium | 0.29.0|

## 运行示例
```python
import gymnasium as gym
import torch
from RLAlgo.PPO2 import PPO2
from RLUtils import train_on_policy, random_play, play, Config, gym_env_desc

env_name = 'Hopper-v4'
gym_env_desc(env_name)
path_ = os.path.dirname(__file__) 
env = gym.make(
    env_name, 
    exclude_current_positions_from_observation=True,
    # healthy_reward=0
)
cfg = Config(
    env, 
    # 环境参数
    save_path=os.path.join(path_, "test_models" ,'PPO_Hopper-v4_test2'), 
    seed=42,
    # 网络参数
    actor_hidden_layers_dim=[256, 256, 256],
    critic_hidden_layers_dim=[256, 256, 256],
    # agent参数
    actor_lr=1.5e-4,
    critic_lr=5.5e-4,
    gamma=0.99,
    # 训练参数
    num_episode=12500,
    off_buffer_size=512,
    off_minimal_size=510,
    max_episode_steps=500,
    PPO_kwargs={
        'lmbda': 0.9,
        'eps': 0.25,
        'k_epochs': 4, 
        'sgd_batch_size': 128,
        'minibatch_size': 12, 
        'actor_bound': 1,
        'dist_type': 'beta'
    }
)
agent = PPO2(
    state_dim=cfg.state_dim,
    actor_hidden_layers_dim=cfg.actor_hidden_layers_dim,
    critic_hidden_layers_dim=cfg.critic_hidden_layers_dim,
    action_dim=cfg.action_dim,
    actor_lr=cfg.actor_lr,
    critic_lr=cfg.critic_lr,
    gamma=cfg.gamma,
    PPO_kwargs=cfg.PPO_kwargs,
    device=cfg.device,
    reward_func=None
)

agent.train()
train_on_policy(env, agent, cfg, wandb_flag=False, train_without_seed=True, test_ep_freq=1000, 
                online_collect_nums=cfg.off_buffer_size,
                test_episode_count=5)
agent.load_model(cfg.save_path)
agent.eval()
env_ = gym.make(env_name, 
                exclude_current_positions_from_observation=True,
                # render_mode='human'
                )
play(env_, agent, cfg, episode_count=3, play_without_seed=True, render=False)
```