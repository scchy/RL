# python3
# Create Date:2021-01-13
# Author: Scc_hy
# Func:  DQN
# Tip: 
#      CartpoleObserver 环境重构
#      CartpoleActor    进行环境探索和模型更新
#      QNeuralEstimzter: Qnet
#      Scaler          : 对state进行标准化
#      CartpoleActorTrainer: 进行Agent训练
# =================================================================================

import random
import numpy as np
import torch as t
from torch import nn
import gym
from collections import namedtuple, deque
import os

Experience = namedtuple('Experience', 's a r n_s d')

class CartPoleObserver:
    def __init__(self, env):
        self._env = env
    
    @property
    def action_space(self):
        return self._env.action_space
    
    @property
    def observation_space(self):
        return self._env.observation_space

    def render(self):
        self._env.render(mode="human")

    def transform(self, state):
        # 便于vstack
        return np.array(state)

    def reset(self):
        return self.transform(self._env.reset())

    def step(self, action):
        n_state, reward, done, info = self._env.step(action)
        return self.transform(n_state), reward, done, info

    def close(self):
        self._env.close()


class Scaler:
    """标准化
    ta = t.tensor([range(100), range(100, 200)]).float()
    t.tensor([np.array([11, 11]), np.array([11, 11])])
    sc = Scaler()
    sc.fit(ta)
    sc.predict(ta)
    """
    def __init__(self, method='standard'):
        """torch 标准化
        """
        self.method = method

    def standard_scaler(self, tensor_in):
        self.a = t.mean(tensor_in.float(), dim=0)
        self.b = t.std(tensor_in.float(), dim=0)
    
    def fit(self, tensor_in):
        if self.method == 'standard':
            self.standard_scaler(tensor_in)

    def predict(self, tensor_in):
        if type(tensor_in) == np.ndarray:
            tensor_in = t.tensor(tensor_in)
        if self.method == 'standard':
            return (tensor_in.float() - self.a) / self.b

    def save(self, file):
        a = self.a.numpy()
        b = self.b.numpy()
        with open(file, 'w') as f:
            f.write(f'{a},{b}')

    def load(self, file):
        with open(file, 'w') as f:
            cc = f.read()
        a, b = cc.split(',')
        self.a = t.tensor(float(a)).float()
        self.b = t.tensor(float(b)).float()


class QNeuralEstimator(nn.Module):
    """QNet
    简单的nn回归器
    sample
        n_model = QNeuralEstimator(input_dim=3, hidden_layer_size=[16, 16], output_dim=2)
        x1 = t.tensor([[1, 2, 3], [1, 2, 3]]).float()
        y1 = t.tensor([[1, 1], [1, 1]]).float()
        n_model(x1)
        n_model.update(x1, y1)
        n_model(x1)
    """
    def __init__(self, input_dim, hidden_layer_size, output_dim):
        super(QNeuralEstimator, self).__init__()
        self.features = nn.ModuleList()
        for i, h in enumerate(hidden_layer_size):
            self.features.append(nn.ModuleDict({
                'linear': nn.Linear(input_dim, h) if not i \
                    else nn.Linear(hidden_layer_size[i-1], h),
                'linear_active': nn.ReLU(inplace=True)
            }))
        
        self.out = nn.Linear(hidden_layer_size[-1], output_dim)
        self.model_compelet()
    
    def forward(self, x):
        for layer in self.features:
            x = layer['linear'](x)
            x = layer['linear_active'](x)
        return self.out(x)
    
    def model_compelet(self):
        self.cost_func = nn.MSELoss()
        self.opt = t.optim.Adam(self.parameters(), lr=0.001)
    
    def update(self, batch_x, batch_y):
        self.opt.zero_grad()
        pred_ = self(batch_x)
        loss = self.cost_func(batch_y, pred_)
        loss.backward()
        self.opt.step()
    
    def fit(self, x, y):
        self.update(x, y)


class CartPoleActor:
    def __init__(self, epsilon, actions):
        self.actions = actions
        self.epsilon = epsilon
        self.scaler_model = None
        self.model = None
        self.initialized = False
    
    def save(self, model_path):
        scaler_file = 'scaler_param.txt'
        self.scaler_model.save(os.path.join(model_path, scaler_file))

        pass

    def load_model(self, model_path):
        scaler_file = 'scaler_param.txt'
        self.scaler_model = Scaler()
        self.scaler_model.load(os.path.join(model_path, scaler_file))
        pass
    
    @classmethod
    def load(cls, env, model_path, epsilon=0.001):
        actions = list(range(env.action_space.n))
        agent = cls(epsilon, actions)
        agent.load_model(model_path)
        agent.initialized = True
        print('Load Model Done!')
    
    def initialize(self, experiences):
        self.scaler_model = Scaler()
        self.model = QNeuralEstimator(
            input_dim=experiences[0].s.shape[0],
            hidden_layer_size=[10, 10],
            output_dim=len(self.actions)
        )
        states = t.tensor([e.s for e in experiences])
        self.scaler_model.fit(states)
        self.update([experiences[0]], gamma=0)
        self.initialized = True
        print('Done initialization. From now, begin training!')
    
    def estimate(self, s):
        s = self.scaler_model.predict(s)
        return self.model(s)
    
    def _predict(self, states):
        if self.initialized:
            return self.estimate(states)

        dim_ = len(self.actions) 
        size = dim_ * len(states)
        return np.random.uniform(size=size).reshape((-1, dim_))
    
    def update(self, batch_experiences, gamma):
        """
        更新状态的预估Q值:
        Q<s, a, t> = R<s, a, t> + gamma * Q<s+1, a_max, t+1>
        """
        states = t.tensor([e.s for e in batch_experiences])
        n_states = t.tensor([e.n_s for e in batch_experiences])

        estimateds = self.estimate(states)
        futures = self.estimate(n_states)
        for i, e in enumerate(batch_experiences):
            R = e.r
            if not e.d:
                R += gamma * t.max(futures[i])
            
            estimateds[i][e.a] = R

        estimateds = t.tensor(estimateds).float()
        self.model.fit(self.scaler_model.predict(states), estimateds)
    
    def policy(self, s):
        if random.random() < self.epsilon or not self.initialized:
            return np.random.randint(len(self.actions))
        
        estimates = self.estimate(s)
        return np.argmax(estimates.detach().numpy())

    def play(self, env, episode_count=5, render=True):
        """
        对训练完成的QNet进行策略游戏
        """
        for e in range(episode_count):
            s = env.reset()
            done = False
            episode_reward = 0
            episode_cnt = 0
            while not done:
                if render:
                    env.render()
                a = self.policy(s)
                n_state, reward, done, info = env.step(a)
                episode_reward += reward
                episode_cnt += 1
                s = n_state
            else:
                print(f'Get reward {episode_reward}. Last {episode_cnt} times')
        
            env.close()


class CartPoleActorTrainer:
    def __init__(self, buffer_size=1024, batch_size=32, gamma=0.9):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.experiences = deque(maxlen=buffer_size)
        self.training = False
        self.training_count = 0

    def train_loop(self, env, agent, episode=200, render=False):
        # 严格early_stop: 连续达到最优值8次停止迭代
        th_ = 8
        not_improve_cnt = 0
        episode_reward = deque(maxlen=2)
        episode_reward.append(0)
        for e in range(episode):
            s = env.reset()
            done = False
            step_count = 0
            episode_r = 0
            while not done:
                if render:
                    env.render()
                
                a = agent.policy(s)
                n_state, reward, done, info = env.step(a)
                episode_r += reward
                # 收集样本
                self.experiences.append(
                    Experience(s, a, reward, n_state, done)
                )
                if not self.training and \
                    len(self.experiences) == self.buffer_size:
                    agent.initialize(self.experiences)
                    self.training = True
                
                # 样本采用并进行参数更新
                self.step(agent)
                s = n_state
                step_count += 1
            else:
                self.episode_end(e, step_count)
                episode_reward.append(episode_r)
                if self.training:
                    self.training_count += 1
            
            if episode_reward[1] == episode_reward[0]:
                not_improve_cnt += 1
                print(f'keep times {not_improve_cnt}')
            else:
                not_improve_cnt = 0
            
            if not_improve_cnt == th_:
                break
    
    def train(self, env, episode_count=200, epsilon=0.1, render=False):
        actions = list(range(env.action_space.n))
        agent = CartPoleActor(epsilon, actions)
        self.train_loop(env, agent, episode_count, render)
        return agent
    
    def step(self, agent):
        if self.training:
            batch = random.sample(self.experiences, self.batch_size)
            agent.update(batch, self.gamma)
    
    def episode_end(self, episode, step_count):
        recent_idx = range(len(self.experiences) - step_count, len(self.experiences))
        recent = [self.experiences[i] for i in recent_idx]
        rewards = sum([e.r for e in recent])
        print(f'[{episode}]-Trained({self.training_count}) Reward- {rewards}')


if __name__ == '__main__':
    env = CartPoleObserver(gym.make('CartPole-v0'))
    trainer = CartPoleActorTrainer(buffer_size=1024, batch_size=128, gamma=0.9)
    trained_agent = trainer.train(env, episode_count=300)
    trained_agent.play(env)

