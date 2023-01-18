# 基于价值的深度强化学习方法
## 7.1 doubleDQN实践(Pendulum-v1)

> 完整代码 [Github: Doubledqn_lr](https://github.com/scchy/RL/blob/main/projects/DDQN_Pendulum-v0/Doubledqn_lr.py)  
> 完整代码 [Github: Duelingdqn_lr](https://github.com/scchy/RL/blob/main/projects/DDQN_Pendulum-v0/Duelingdqn_lr.py)

### 7.1.1 环境描述

环境是倒立摆（Inverted Pendulum），该环境下有一个处于随机位置的倒立摆。环境的状态包括倒立摆角度的**正弦值，余弦值，角速度**；动作为对倒立摆施加的力矩(`action = Box(-2.0, 2.0, (1,), float32)`)。每一步都会根据当前倒立摆的状态的好坏给予智能体不同的奖励，该环境的奖励函数为，倒立摆向上保持直立不动时奖励为 0，倒立摆在其他位置时奖励为负数。环境本身没有终止状态，所以训练的时候需要设置终止条件(笔者在本文设置了`260`)。

### 7.1.2 构建智能体

构建智能体：
`policy`是和之前一样的。探索和利用， 就是利用的时候基于nn模型的预测
主要核心：
- QNet:
  -  就是一个多层的NN
  -  update就是用MSELoss进行梯度下降
- `DQN`: `支持DQN和doubleDQN`
  - update:
    - 经验回放池R中的数据足够, 从R中采样N个数据 `{ (si, ai, ri, si+1) }+i=1,...,N `
    - <font color=darkred>对于DQN: 对每个数据计算 $y_i = r_i + \gamma Q_{w^-}(s',arg max(Q_{w^-}(s', a'))$ ,**动作的选取依靠目标网络**($Q_{w-}$)</font>
    - <font color=darkred>对于doubleDQN: 对每个数据计算 $y_i = r_i + \gamma  Q_{w^-}(s',arg max(Q_{w}(s', a'))$ ,**动作的选取依靠训练网络**($Q_w$) </font>
    	- $f_{max}$就是用$QNet(s_{i+1})$计算出每个action对应的值，取最大值的index, 然后根据这个index去$TagetQNet(s_{i+1})$中取最大值
下面的代码可以看出差异
```python
        # 下个状态的最大Q值
        if self.dqn_type == 'DoubleDQN': # DQN与Double DQN的区别
            max_action = self.q(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = n_actions_q.gather(1, max_action)
        else: # DQN的情况
            max_next_q_values = n_actions_q.max(1)[0].view(-1, 1)

        q_targets = reward + self.gamma * max_next_q_values * (1 - done) 
```
- 最小化目标函数$L=\frac{1}{N}\sum (yi - QNet(s_i, a_i))^2$
	- 用 $y_i$ 和 `q(states).gather(1, action)` 计算损失并更新参数

**代码实现**
```python
class QNet(nn.Module):
    def __init__(self, state_dim: int, hidden_layers_dim: typ.List, action_dim: int):
        super(QNet, self).__init__()
        self.features = nn.ModuleList()
        for idx, h in enumerate(hidden_layers_dim):
            self.features.append(
                nn.ModuleDict({
                    'linear': nn.Linear(state_dim if not idx else hidden_layers_dim[idx-1], h),
                    'linear_active': nn.ReLU(inplace=True)
                    
                })
            )
        self.header = nn.Linear(hidden_layers_dim[-1], action_dim)
    
    def forward(self, x):
        for layer in self.features:
            x = layer['linear_active'](layer['linear'](x))
        return self.header(x)

    def model_compelet(self, learning_rate):
        self.cost_func = nn.MSELoss()
        self.opt = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def update(self, pred, target):
        self.opt.zero_grad()
        loss = self.cost_func(pred, target)
        loss.backward()
        self.opt.step()


class DQN:
    def __init__(self, 
                state_dim: int, 
                hidden_layers_dim, 
                action_dim: int, 
                learning_rate: float,
                gamma: float,
                epsilon: float=0.05,
                traget_update_freq: int=1,
                device: typ.AnyStr='cpu',
                dqn_type: typ.AnyStr='DQN'
                ):
        self.action_dim = action_dim
        # QNet & targetQNet
        self.q = QNet(state_dim, hidden_layers_dim, action_dim)
        self.target_q = copy.deepcopy(self.q)
        self.q.to(device)
        self.q.model_compelet(learning_rate)
        self.target_q.to(device)

        # iteration params
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        
        # target update freq
        self.traget_update_freq = traget_update_freq
        self.count = 0
        self.device = device
        
        # dqn类型
        self.dqn_type = dqn_type
        
    def policy(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        action = self.target_q(torch.FloatTensor(state))
        return np.argmax(action.detach().numpy())
    
    def update(self, samples: deque):
        """
        Q<s, a, t> = R<s, a, t> + gamma * Q<s+1, a_max, t+1>
        """
        self.count += 1
        state, action, reward, next_state, done = zip(*samples)
        
        states = torch.FloatTensor(state).to(self.device)
        action = torch.tensor(action).view(-1, 1).to(self.device)
        reward = torch.tensor(reward).view(-1, 1).to(self.device)
        next_states = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done).view(-1, 1).to(self.device)
        
        actions_q = self.q(states)
        n_actions_q = self.target_q(next_states)
        q_values = actions_q.gather(1, action)
        # 下个状态的最大Q值
        if self.dqn_type == 'DoubleDQN': # DQN与Double DQN的区别
            max_action = self.q(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = n_actions_q.gather(1, max_action)
        else: # DQN的情况
            max_next_q_values = n_actions_q.max(1)[0].view(-1, 1)

        q_targets = reward + self.gamma * max_next_q_values * (1 - done) 
        # MSELoss update
        self.q.update(q_values.float(), q_targets.float())
        if self.count % self.traget_update_freq == 0:
            self.target_q.load_state_dict(
                self.q.state_dict()
            )
```
### 7.1.3 智能体训练
#### 注意点
在训练的时候：
- 所有的参数都在`Config`中进行配置，便于调参

```python
class Config:
    num_episode  = 300
    state_dim = None
    hidden_layers_dim = [10, 10]
    action_dim = 20
    learning_rate = 2e-3
    gamma = 0.95
    epsilon = 0.01
    traget_update_freq = 3
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    buffer_size = 2048
    minimal_size = 1024
    batch_size = 128
    render = False
    save_path =  r'D:\TMP\model.ckpt' 
    dqn_type = 'DoubleDQN'
    # 回合停止控制
    max_episode_rewards = 260
    max_episode_steps = 260

    def __init__(self, env):
        self.state_dim = env.observation_space.shape[0]
        try:
            self.action_dim = env.action_space.n
        except Exception as e:
            pass
        print(f'device = {self.device} | env={str(env)}')
```

- 由于这次环境的action空间是连续的，**我们需要有一个函数进行action的离散和连续的转换**

```python
def Pendulum_dis_to_con(discrete_action, env, action_dim):  # 离散动作转回连续的函数
    action_lowbound = env.action_space.low[0]  # 连续动作的最小值
    action_upbound = env.action_space.high[0]  # 连续动作的最大值
    action_range = action_upbound - action_lowbound
    return action_lowbound + (discrete_action / (action_dim - 1)) * action_range

```

#### 训练

需要注意的是笔者的`gym版本是0.26.2`

```python
def train_dqn(env, cfg, action_contiguous=False):
    buffer = replayBuffer(cfg.buffer_size)
    dqn = DQN(
        state_dim=cfg.state_dim,
        hidden_layers_dim=cfg.hidden_layers_dim, 
        action_dim=cfg.action_dim,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        epsilon=cfg.epsilon,
        traget_update_freq=cfg.traget_update_freq,
        device=cfg.device,
        dqn_type=cfg.dqn_type
    )
    tq_bar = tqdm(range(cfg.num_episode))
    rewards_list = []
    now_reward = 0
    bf_reward = -np.inf
    for i in tq_bar:
        tq_bar.set_description(f'Episode [ {i+1} / {cfg.num_episode} ]')
        s, _ = env.reset()
        done = False
        episode_rewards = 0
        steps = 0
        while not done:
            a = dqn.policy(s)
            # [Any, float, bool, bool, dict]
            if action_contiguous:
                c_a = Pendulum_dis_to_con(a, env, cfg.action_dim)
                n_s, r, done, _, _ = env.step([c_a])
            else:
                n_s, r, done, _, _ = env.step(a)
            buffer.add(s, a, r, n_s, done)
            s = n_s
            episode_rewards += r
            steps += 1
            # buffer update
            if len(buffer) > cfg.minimal_size:
                samples = buffer.sample(cfg.batch_size)
                dqn.update(samples)
            if (episode_rewards >= cfg.max_episode_rewards) or (steps >= cfg.max_episode_steps):
                break

        rewards_list.append(episode_rewards)
        now_reward = np.mean(rewards_list[-10:])
        if bf_reward < now_reward:
            torch.save(dqn.target_q.state_dict(), cfg.save_path)

        bf_reward = max(bf_reward, now_reward)
        tq_bar.set_postfix({'lastMeanRewards': f'{now_reward:.2f}', 'BEST': f'{bf_reward:.2f}'})
    env.close()
    return dqn


if __name__ == '__main__':
    print('=='*35)
    print('Training Pendulum-v1')
    p_env = gym.make('Pendulum-v1')
    p_cfg = Config(p_env)
    p_dqn = train_dqn(p_env, p_cfg, True)
```

### 7.1.4 训练出的智能体观测

最后将训练的最好的网络拿出来进行观察

```python
p_dqn.target_q.load_state_dict(torch.load(p_cfg.save_path))
play(gym.make('Pendulum-v1', render_mode="human"), p_dqn, p_cfg, episode_count=2, action_contiguous=True)
```

从下图中我们可以看出，本次的训练成功还是可以的。
![在这里插入图片描述](https://img-blog.csdnimg.cn/8fbff172f5054ad2b9b68ff48627fbb5.gif#pic_center)
> 完整脚本查看笔者github: [Doubledqn_lr.py](https://github.com/scchy/RL/blob/main/projects/DDQN_Pendulum-v0/Doubledqn_lr.py) 记得点`Star`哦


笔者后续会更深入的学习强化学习并对`gym`各个环境逐一进行训练


## 7.2 DataWhale习题回答

### 为什么传统的深度Q 网络的效果并不好？

传统的表格方法Q-learning按$Q_t(s, a) = Q_t(s, a) + \alpha(R_t + \gamma max_a Q_t(s_{t+1}) - Q_t(s, a))$来迭代QTable, 最终是需要让`$Q_t(s, a)$`和`$R_t + \gamma max_a Q_t(s_{t+1})$`趋于一致。

在深度Q网络的损失函数同样基于此设计$L = MSE(Q_t(s, a), R_t + \gamma max_a Q_t(s_{t+1}))$
从式子中我们可以看出目标值很容易一不小心被设得太高，因为在计算目标的时候，实际采用的是哪个动作获得最大价值，就把它加上去变成我们的目标。即，每次我们都会选择哪个Q值被高估的动作，总是会选哪个奖励被高估的动作这个最大的结果去加上rt当目标，所以目标总是太大。

### 接着上个思考题，我们应该怎么解决目标值总是太大的问题呢？

采用DoubleQ网络解决该问题。在DoubleQ网络中，存在目标QNet和训练QNet:

- 训练QNet：决定选用什么动作
- 目标QNet: 计算所选动作的价值
- 这样在一定程度上降低了对Q值的高估

### 如何理解Dueling DQN的模型变化带来的好处？

Dueling DQN 能更高效学习状态价值函数。  
Dueling DQN 不直接计算$Q(s, a)$,而是拆分成

- 优势函数`A` 
- 状态价值函数`V`
- 状态价值 + 优势来计算Q值:$Q(s, a)=V + A - E(A)$(以用平均代替最大化操作)
- V和A共享了一些网络层，头部分成V和A。在更新时不一定会将V和Q都更新

将其分成两个部分后，我们就不需要将所有的状态-动作都采样一遍，这样就可以更加高效的估计Q值

直接看网络结构我们就可以看出两者的主要差异

```python
class VANet(nn.Module):
    # 可以让智能体开始关注不同动作优势值的差异
    # Dueling DQN 能够更加频繁、准确地学习状态价值函数
    def __init__(self, state_dim, hidden_layers_dim, action_dim):
        super(VANet, self).__init__()
        self.features = nn.ModuleList()
        for idx, h in enumerate(hidden_layers_dim):
            self.features.append(nn.ModuleDict({
                'linear': nn.Linear(hidden_layers_dim[idx-1] if idx else state_dim, h),
                'linear_active': nn.ReLU(inplace=True)
            })) 
        
        # 表示采取不同动作的差异性
        self.adv_header = nn.Linear(hidden_layers_dim[-1], action_dim)
        # 状态价值函数
        self.v_header = nn.Linear(hidden_layers_dim[-1], 1)
    
    def forward(self, x):
        for layer in self.features:
            x = layer['linear_active'](layer['linear'](x))
        
        adv = self.adv_header(x)
        v = self.v_header(x)
        # Q值由V值和A值计算得到
        Q = v + adv - adv.mean().view(-1, 1)  
        return Q


class QNet(nn.Module):
    def __init__(self, state_dim, hidden_layers_dim, action_dim):
        super(QNet, self).__init__()
        self.featires = nn.ModuleList()
        for idx, h in enumerate(hidden_layers_dim):
            self.features.append(nn.ModuleDict({
                'linear': nn.Linear(hidden_layers_dim[idx-1] if idx else state_dim, h),
                'linear_active': nn.ReLU(inplace=True)
            }))
        self.head = nn.Linear(hidden_layers_dim[-1], action_dim)
    
    def forward(self, x):
        for layer in self.features:
            x = layer['linear_active'](layer['linear'](x))
        return self.head(x)


if dqn_type == 'DuelingDQN':
    self.q = VANet(state_dim, hidden_layers_dim, action_dim).to(device)
else:
    self.q = QNet(state_dim, hidden_layers_dim, action_dim).to(device)

```

### 使用蒙特卡罗和时序差分平衡方法的优劣分别有哪些

优势：  

- 因为我们现在采样了比较多的步骤，之前是只采样一步，所以某一步得到的数据是真实值，接下来都是Q值估计出来的。现在比较多步骤，采样N步才估测价值，所以估测的部分所造成的影响就会比较小。

劣势：  

- N步相加，会增加对应的方差。但是可以通过调整N值，在方差与不精确的Q值之间做衡量

### DQN相比于基于策略梯度方法为什么训练起来效果更好，更平稳？

DQN比较容易训练的一个理由是: 在DQN里面，我们只要能够估计出q函数，就保证一定可以找到一个比较好的策略。也就是我们只要能够估计出Q函数，就保证可以改进策略。而估计Q函数是比较容易的，因为它就是一个回归问题。


### DQN在处理连续型动作时存在什么样的问题呢？对应的解决方法有哪些呢？

因为要选取收益最大的action，对于连续动作不易处理。

从环境的action角度出发：
- 将动作离散化，进行DQN训练
- 
- 同时将离散动作转化为连续值在环境中执行

从模型设计角度出发：

- 最大化目标函数，将action作为参数，要找一组action去最大化Q函数， 梯度上升去更新action值。但是 等于是每次要决定采取哪一个动作的时候，都还要训练一次网络，显然运算量是很大的
- 设计网络，输入`state`, Q函数输出3个东西: 向量u(s) 矩阵$\sum(s)$ 标量 V(s)网络输出后才引入action$Q(s, a) = -(a - u(s))^T \sum(s)(a - u(s)) + V(s)$
- 不用深度Q网络, 将基于策略的方法PPO和基于价值的方法DQN结合在一起，也就可以得到Actor-Criticor的方法
