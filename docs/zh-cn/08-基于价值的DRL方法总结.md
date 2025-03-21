
# 基于价值的DRL方法总结

DQN出现高估`action-value`, 有一以下2个情形
1. 最大Q值 TD target: 
   1. $y_t = r_t + \gamma argmax_a Q(s_{t+1}, a;w)$
2. Bootstrapping:  
   1. SGD: $w = w - \alpha(Q(s_t, a_t;w)-y_t)\frac{\partial Q(s_t, a_t;w)}{\partial w}$

solve problem:
1. doubleDQN: 
   1. $a^* = argmax_a Q(s_{t+1}, a;w)$
   2. <b>$y_t = r_t + \gamma  Q(s_{t+1}, a^*;\hat{w})$</b>
   3. $w = w - \alpha(Q(s_t, a_t;w)-y_t)\frac{\partial Q(s_t, a_t;w)}{\partial w}$
   4. Periodically update $\hat{w} $
      1. $\hat{w} = w$
      2. $\hat{w} = \tau w + (1 -\tau) \hat{w}$

|方法| action描述 | state描述 | QNet | TagetQNet | action选取 | QTarget | loss | 备注 |
|-|-|-|-|-|-|-|-|-|
| DQN | 仅离散动作 | 支持连续状态 | $Q(s_t)$ -> q | - | $a_{mx}=max_a(Q(s_{t+1}))$| $q_{t+1}=Q(s_{t+1})[a_{mx}];\\  Tar=r_t + \gamma * q_{t+1}$ | MSE($Q(s_t)$, Tar)| 对传统Qtable状态空间有限的拓展|
| DQN' | 仅离散动作 | 支持连续状态 | $Q(s_t)$ -> q | $Q_{tar}$[simple method to avoid bootstrapping] | $a_{mx}=max_a(Q_{tar}(s_{t+1}))$| $q_{t+1}=Q_{tar}(s_{t+1})[a_{mx}];\\  Tar=r_t + \gamma * q_{t+1}$ | MSE($Q(s_t)$, Tar)| 对传统Qtable状态空间有限的拓展|
| doubleDQN | 仅离散动作 | 支持连续状态 | $Q(s_t)$ -> q | $Q_{tar}$ | <font color=darkred>$a^*=max_a(Q(s_{t+1}))$</font>| $q_{t+1}=Q_{tar}(s_{t+1})[a^*];\\  Tar=r_t + \gamma * q_{t+1}$ | MSE($Q(s_t)$, Tar)| 对DQN Qtarget高估的修正|
| DuelingDQN | 仅离散动作 | 支持连续状态 | $V(s_t)$ -> <font color=darkred>V + A - mean(A) </font>-> q | deepcopy(V) | $a=max_a(V_{tar}(s_{t+1}))$| $q_{t+1}=V_{tar}(s_{t+1})[a];\\ Tar=r_t + \gamma * q_{t+1}$ | MSE($Q(s_t)$, Tar)| 拆分成价值函数和优势函数计算q,另一种修正QTagret高估方法 |


## 8.1 环境实验与调参经验

|环境与描述 | 使用方法 | 网络结构 | 环境参数配置 | agent参数配置 | 训练参数配置 | 经验 |
|-|-|-|-|-|-|-|
|[ CartPole-v1 ](state: (4,),action: 2(离散))| DuelingDQN | [10, 10] | `dict(action_contiguous_=False, seed=42)` | `dict(epsilon=0.01, target_update_freq=3, gamma=0.95, learning_rate=2e-3)` | `dict(num_episode=500, off_buffer_size=2048, off_minimal_size=1024, sample_size=256, max_episode_steps=240)`| 相对简单环境buffer可以相对小一些，训练效果不佳时`num_episode`可以调大  |
|[ MountainCar-v0 ](state: (2,),action: 3(离散 ))| DuelingDQN | [32, 32] | `dict(action_contiguous_=False, seed=42)` | `dict(epsilon=0.05, target_update_freq=3, gamma=0.95, learning_rate=2e-3)` | `dict(num_episode=200, off_buffer_size=3076, off_minimal_size=1024, sample_size=256, max_episode_steps=500)`| 相对复杂环境: 1.增加网络宽度与深度 2.需要增加探索率`epsilon`; 3.将回合步数调大`max_episode_steps`;需要多次训练尝试，同时也可以适量的增大`off_buffer_size`  |
|[ Acrobot-v1 ](state: (6,),action: 3(离散 ))| DuelingDQN | [128, 64] | `dict(action_contiguous_=False, seed=42)` | `dict(epsilon=0.05, target_update_freq=3, gamma=0.95, learning_rate=2e-3)` | `dict(num_episode=300, off_buffer_size=20480, off_minimal_size=1024, sample_size=256, max_episode_steps=400)`| 相对复杂环境: 1.增加网络宽度与深度 2.需要增加探索率`epsilon`; 3.将回合步数调大`max_episode_steps`;4. 适量的增大`off_buffer_size`;5. 增加迭代次数  |
|[ LunarLander-v2 ](state: (8,),action: 4(离散 ))| DuelingDQN | [128, 64] | `dict(action_contiguous_=False, seed=42)` | `dict(epsilon=0.05, target_update_freq=3, gamma=0.99, learning_rate=2e-3)` | `dict(num_episode=800, off_buffer_size=20480, off_minimal_size=2048, sample_size=128, max_episode_steps=200)`| 由于环境需要正确降落, 所以在调参的时候需要调整`max_episode_steps`不能过大也不过小，过大可能会导致智能体悬空，同时需要增加迭代次数`num_episode` ,适当减小每次的`sample_size` |
|[ ALE/DemonAttack-v5 ](state: (210, 160, 3),action: 6(离散 ))| doubleDQN | CNN+[200, 200]| `dict(action_contiguous_=False, seed=random)` |`dict(epsilon=0.05, target_update_freq=16, gamma=0.95, learning_rate=1.0e-4, epsilon_start=0.95, epsilon_decay_steps=15000)`| `dict(num_episode=1200, off_buffer_size=12000, off_minimal_size=1024, sample_size=32, max_episode_steps=280)`| 主要取决于CNN网络对特征的提取能力, 由于奖励只有打到敌机才有，所以需要适当增加跳帧的数量，强转成"连续奖励"，同时需要增加迭代次数`num_episode` ,适当减小每次的`sample_size` |


## 8.2 实验效果

|环境与描述 | 使用方法 | 配置| 效果|
|-|-|-|-|
|[ CartPole-v1 ](state: (4,),action: 2(离散))| DuelingDQN | 上述对应配置| ![duelingDQN_CartPole](../pic/duelingDQN_CartPole-v1.gif) |
|[ MountainCar-v0 ](state: (2,),action: 3(离散 ))| DuelingDQN | 上述对应配置| ![duelingDQN_MountainCar](../pic/duelingDQN_MountainCar-v0.gif) |
|[ Acrobot-v1 ](state: (6,),action: 3(离散 ))| DuelingDQN | 上述对应配置| ![duelingDQN_Acrobot](../pic/DQN_Acrobot-v1.gif) |
|[ LunarLander-v2 ](state: (8,),action: 4(离散 ))| DuelingDQN | 上述对应配置| ![duelingDQN_LunarLander](../pic/duelingDQN_LunarLander-v2.gif) |
|[ ALE/DemonAttack-v5 ](state: (210, 160, 3),action: 6(离散 ))| doubleDQN | 上述对应配置| ![doubleDQN-DemonAc](../pic/DQN_DemonAttack-v5.gif) |

## 8.3 环境实验脚本简述

详细看[github test_dqn脚本](https://github.com/scchy/RL/blob/main/src/test/test_dqn.py)

**CartPole-v1**

```python
import gym
import torch
from RLAlgo.DQN import DQN
from RLUtils import train_off_policy, play, Config, gym_env_desc

env_name = 'CartPole-v1' 
gym_env_desc(env_name)
env = gym.make(env_name)
action_contiguous_ = False

cfg = Config(
    env, 
    # 环境参数
    split_action_flag=True,
    save_path=r'D:\TMP\dqn_target_q.ckpt',
    # 网络参数
    hidden_layers_dim=[10, 10],
    # agent参数
    learning_rate=2e-3,
    target_update_freq=3,
    gamma=0.95,
    epsilon=0.01,
    # 训练参数
    num_episode=500,
    off_buffer_size=2048,
    off_minimal_size=1024,
    sample_size=256,
    max_episode_steps=240,
    # agent 其他参数
    dqn_type = 'duelingDQN'
)
dqn = DQN(
    state_dim=cfg.state_dim,
    hidden_layers_dim=cfg.hidden_layers_dim,
    action_dim=cfg.action_dim,
    learning_rate=cfg.learning_rate,
    gamma=cfg.gamma,
    epsilon=cfg.epsilon,
    target_update_freq=cfg.target_update_freq,
    device=cfg.device,
    dqn_type=cfg.dqn_type
)
train_off_policy(env, dqn, cfg, action_contiguous=action_contiguous_)

```

**MountainCar-v0**

```python
env_name = 'MountainCar-v0' 
gym_env_desc(env_name)
env = gym.make(env_name)
action_contiguous_ = False # 是否将连续动作离散化

cfg = Config(
    env, 
    # 环境参数
    split_action_flag=True,
    save_path=r'D:\TMP\dqn_target_q.ckpt',
    seed=42,
    # 网络参数
    hidden_layers_dim=[32, 32],
    # agent参数
    learning_rate=2e-3,
    target_update_freq=3,
    gamma=0.95,
    epsilon=0.05,
    # 训练参数
    num_episode=200,
    off_buffer_size=2048+1024,
    off_minimal_size=1024,
    sample_size=256,
    max_episode_steps=500,
    # agent 其他参数
    dqn_type = 'duelingDQN'
)

```


**Acrobot-v1**

```python
env_name = 'Acrobot-v1' 
gym_env_desc(env_name)
env = gym.make(env_name)
action_contiguous_ = False # 是否将连续动作离散化

cfg = Config(
    env, 
    # 环境参数
    split_action_flag=True,
    save_path=r'D:\TMP\Acrobot_dqn_target_q.ckpt',
    seed=42,
    # 网络参数
    hidden_layers_dim=[128, 64],
    # agent参数
    learning_rate=2e-3,
    target_update_freq=3,
    gamma=0.95,
    epsilon=0.05,
    # 训练参数
    num_episode=300,
    off_buffer_size=20480,
    off_minimal_size=1024,
    sample_size=256,
    max_episode_steps=400,
    # agent 其他参数
    dqn_type = 'duelingDQN'
)
```


**LunarLander-v2**

```python
env_name = 'LunarLander-v2' 
gym_env_desc(env_name)
env = gym.make(env_name)
action_contiguous_ = False # 是否将连续动作离散化

cfg = Config(
    env, 
    # 环境参数
    split_action_flag=True,
    save_path=r'D:\TMP\LunarLander_dqn_target_q.ckpt',
    seed=42,
    # 网络参数
    hidden_layers_dim=[128, 64],
    # agent参数
    learning_rate=2e-3,
    target_update_freq=3,
    gamma=0.99,
    epsilon=0.05,
    # 训练参数
    num_episode=800,
    off_buffer_size=20480,
    off_minimal_size=2048,
    sample_size=128,
    max_episode_steps=200,
    # agent 其他参数
    dqn_type = 'duelingDQN'
)
```


**ALE/DemonAttack-v5**
详细看[github test_dqn.py : DemonAttack_v5_dqn_new_test()](https://github.com/scchy/RL/blob/main/src/test/test_dqn.py)

一些技巧(tricks):

1. 环境观察与调整:  
   1. 跳帧：一个action执行5个step(5桢) 强转成“连续奖励空间”
   2. 图像裁剪->图像转灰度->图像归一化
   3. 对多个输出进行通道叠加`FrameStack`
2. CNN网络
   1. 解决梯度消失问题(`Vanishing gradient problem`)
   2. 池化技巧：MaxPool2d + AvgPool2d
   3. 增加LayerNorm
3. DoubleDQN算法调整
   1. 增加epsilon decay
4. 小结
   1. 这种环境用onpolicy的算法会更加合适
   2. 稀疏奖励用DQN难以收敛
   3. 可以将动作空间缩小：减少作用不大的动作，比如 向左 <= 向左&发射自动；向右 <= 向右&发射自动

```python
# Episode [ 10061 / 15000|(seed=8409) ]:  67%|██████▋   | 10060/15000 [9:03:17<2:44:32,  2.00s/it, steps=23, lastMeanRewards=140.40, BEST=545.37, bestTestReward=810.89]

def DemonAttack_v5_dqn_new_test():
    # [ seed=7270 ] Get reward 1500.0. Last 596 times
    env_name = 'ALE/DemonAttack-v5' 
    gym_env_desc(env_name)
    env = gym.make(env_name, obs_type="rgb")
    print("gym.__version__ = ", gym.__version__ )
    env = FrameStack(
        ResizeObservation(
            GrayScaleObservation(baseSkipFrame(
                env, 
                skip=5, 
                cut_slices=[[15, 188], [0, 160]],
                start_skip=14)), 
            shape=84
        ), 
        num_stack=4
    )
    path_ = os.path.dirname(__file__)
    cfg = Config(
        env, 
        # 环境参数
        split_action_flag=False,
        save_path=os.path.join(path_, "test_models", f'dqn_DemonAttack-v5-new_1'),
        seed=42,
        # 网络参数
        hidden_layers_dim=[200, 200],
        # agent参数
        learning_rate=1.0e-4,
        target_update_freq=16,
        gamma=0.99,
        epsilon=0.05,
        # 训练参数
        num_episode=1200,
        off_buffer_size=12000,
        off_minimal_size=1024, 
        sample_size=32,
        max_episode_steps=280,
        # agent 其他参数
        dqn_type = 'DoubleDQN-CNN',
        epsilon_start=0.95,
        epsilon_decay_steps=15000
    )
    dqn = DQN(
        state_dim=cfg.state_dim,
        hidden_layers_dim=cfg.hidden_layers_dim,
        action_dim=cfg.action_dim,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        epsilon=cfg.epsilon,
        target_update_freq=cfg.target_update_freq,
        device=cfg.device,
        dqn_type=cfg.dqn_type,
        epsilon_start=cfg.epsilon_start,
        epsilon_decay_steps=cfg.epsilon_decay_steps
    )
    # dqn.train()
    # train_off_policy(env, dqn, cfg, done_add=False, 
    #                  train_without_seed=True, 
    #                  wandb_flag=False, 
    #                  test_ep_freq=50, test_episode_count=10)
    dqn.load_model(cfg.save_path)
    dqn.eval()
    env = gym.make(env_name, obs_type="rgb")#, render_mode='human')
    env = FrameStack(
        ResizeObservation(
            GrayScaleObservation(baseSkipFrame(
                env, 
                skip=5, 
                cut_slices=[[15, 188], [0, 160]],
                start_skip=14)), 
            shape=84
        ), 
        num_stack=4
    )
    play(env, dqn, cfg, episode_count=1, 
         play_without_seed=True, render=False)
```
