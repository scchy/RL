## Neuronlike adaptive elements that can solve difficult learning control problems

paper Link: [sci-hub: Neuronlike adaptive elements that can solve difficult learning control problems](https://sci-hub.se/10.1109/tsmc.1983.6313077)


### 摘要
    通过两个类似神经元的自适应元素组成的系统解决一个复杂的控制学习问题。
- 研究环境: Cart-pole (和gym的[classic_control/cart_pole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) 类似)
- 算法:  ASE + ACE
  - associative search element (ASE) : 强化输入与输出之间的关联
  - adaptive critic element (ACE)：构建一个比单独的强化反馈更有信息量的评估函数
- 主要贡献：
  - **自适应元素的能力**：ASE 和 ACE 的结合能够解决复杂的控制学习问题，即使在反馈信号质量较低的情况下
  - **对神经科学的启示**：论文提出，如果生物网络中的组件也具有类似的自适应能力，那么可能存在与本文描述的自适应元素相似的神经元
  - **对人工智能的启示**：这种自适应元素的设计为构建能够解决复杂问题的网络提供了一种新的方法


### 核心算法

 
```python 
class ASE:
    """associative search element"""
    def __init__(self, input_dim, output_dim, learning_rate=0.1):
        self.w = np.random.randn(input_dim, output_dim)
        self.lr = learning_rate
    
    def forward(self, state):
        return np.dot(state, self.w)
    
    def __call__(self, state):
        return self.forward(state)
    
    def update(self, s, a, r, s_n):
        delta = r + np.dot(n_s, self.w) - np.dot(s, self.w)
        self.w += self.lr * np.outer(state, delta)

class ACE:
    """adaptive critic element"""
    def __init__(self, input_dim, learning_rate=0.1, gamma=0.95):
        self.w = np.random.randn(input_dim)
        self.lr = learning_rate
        self.gamma = gamma
    
    def forward(self, state):
        return np.dot(state, self.w)
    
    def __call__(self, state):
        return self.forward(state)

    def update(self, s, r, s_n):
        delta = r + self.gamma * np.dot(s_n, self.w) - np.dot(s, self.w)
        self.w += self.lr * delta * state
```



```python


import gymnasium as gym 
from tqdm.auto import tqdm


env = gym.make('CartPole-v1')
# 示例：小车-杆平衡问题
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n  

ase = ASE(input_dim, output_dim)
ace = ACE(input_dim)

r_l = []
for episode in tqdm(range(10)):
    r_t = 0
    done = False
    s, _ = env.reset()
    while not done:
        action = ase.forward(s)  # ASE 选择动作
        n_s, r, terminated, truncated, infos = env.step(np.argmax(action))  # 模拟环境的下一个状态
        s = n_s
        r_t += r
        done = terminated or truncated

    r_l.append(r_t)

print(r_l)


# 模拟训练过程
num_episodes = 10000
for episode in tqdm(range(num_episodes)):
    s, _ = env.reset()
    done = False
    while not done:
        action = ase.forward(s)  # ASE 选择动作
        n_s, r, terminated, truncated, infos = env.step(np.argmax(action))  # 模拟环境的下一个状态

        # 更新 ACE
        internal_reward = ace.forward(s)
        ace.update(s, r, n_s)

        # 更新 ASE
        ase.update(s, action, internal_reward, n_s)

        s = n_s
        done = terminated or truncated

print("训练完成！")




r_l = []
for episode in tqdm(range(10)):
    r_t = 0
    done = False
    s, _ = env.reset()
    while not done:
        action = ase.forward(s)  # ASE 选择动作
        n_s, r, terminated, truncated, infos = env.step(np.argmax(action))  # 模拟环境的下一个状态
        s = n_s
        r_t += r
        done = terminated or truncated

    r_l.append(r_t)

print(r_l)

```
