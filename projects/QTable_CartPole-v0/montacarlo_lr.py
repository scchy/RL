# python3
# Create date: 2021-12-18
# Author: Scc_hy
# Func: monta-carlo lr 解决 CartPole-v0问题
# ===============================================================================

import gym
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

np_round = partial(
            np.round, decimals=2
        )

np_round(np.array([1.111111,2.2213131]))

# ---------------------------------------------------------------
# 环境探索
# ---------------------------------------------------------------
env = gym.make('CartPole-v0')
s = env.reset()
print('High: ', env.observation_space.high,
      '\nLow: ',env.observation_space.low)


ob_list = []
for _ in range(10000):
    # env.render()
    a = env.action_space.sample()
    n_state, reward, done, info = env.step(a)
    ob_list.append(np.take(n_state, [1, 3]))

env.close()

ob_arr = np.stack(ob_list)
fig, axes = plt.subplots(2, 1, figsize=(16, 4))
for i in range(2):
    axes[i].plot(ob_arr[:, i])
    max_ = ob_arr[:, i].max()
    min_ = ob_arr[:, i].min()
    x = 1 if i == 0 else 3
    axes[i].set_title(f'observation-[{x}] max: ${max_:.2f}$   min: ${min_:.2f}$')
    axes[i].set_xticks([])

plt.show()


# ---------------------------------------------------------------
# 构建智能体
## 波尔兹曼探索 
# ---------------------------------------------------------------
class CartPoleActor:
    def __init__(self, env, round_num=1):
        self.env = env
        self.round_num = round_num
        self.np_round = partial(
            np.round, decimals=round_num
        )
        self.actions = list(range(env.action_space.n))
        self.a_low, self.b_low = self.np_round(np.take(env.observation_space.low, [0, 2]))
        self.a_high, self.b_high = self.np_round(np.take(env.observation_space.high, [0, 2]))
        self.Q = self.__make_Q_table()

    def get_distribution_arr(self, a_low, a_high):
        a_cnt = int((a_high - a_low) * (10 ** self.round_num) +1)
        a = self.np_round(np.linspace(a_low, a_high, a_cnt))
        if not np.sum(0 == a):
            a = np.concatenate([a, np.array([-0., 0.])])
        else:
            a = np.concatenate([a, np.array([-0.])])
        return a

    def __make_Q_table(self):
        a = self.get_distribution_arr(self.a_low, self.a_high)
        b = self.get_distribution_arr(-3., 3.)
        c = self.get_distribution_arr(self.b_low, self.b_high)
        d = self.get_distribution_arr(-3., 3.)
        Q_dict = dict()
        for s1 in a:
            for s2 in b:
                for s3 in c:
                    for s4 in d:
                        Q_dict[str(self.np_round(np.array([s1, s2, s3, s4])))
                            ] = np.random.uniform(0, 1, len(self.actions))
        print('len(Q_dict) = ', len(Q_dict))
        return Q_dict

    @staticmethod
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    
    def policy(self, s):
        return np.random.choice(
            self.actions, 1, p=self.softmax(self.Q[s])
        )[0]
    

# ---------------------------------------------------------------
# 训练智能体
# ---------------------------------------------------------------

class CartPoleMontoCarlo:
    def __init__(self, actor, env):
        """
        actor 可以为已经训练好的agent
        """
        self.actor = actor
        self._env = env
        self.np_round = actor.np_round

    def take_state(self, state):
        if type(state) == str:
            return state
        s1, s2, s3, s4 = self.np_round(state)
        s1 = np.clip(s1, -4.8, 4.8)
        s2 = np.clip(s3, -3., 3.)
        s3 = np.clip(s2, -0.4, 0.4)
        s4 = np.clip(s4, -3., 3.)
        return str(self.np_round(np.array([s1, s2, s3, s4])))

    def train(self, epoches, gamma, learning_rate):
        env = self._env
        loop_cnt_list = []
        # 进行epochs训练
        for e in range(epoches):
            # 进行一次训练
            s = env.reset()
            done = False
            loop_cnt = 0
            state_list, action_list, reward_list = [], [], []
            while not done:
                s = self.take_state(s)
                a = self.actor.policy(s)
                n_state, reward, done, info = env.step(a)
                n_state = self.take_state(n_state)
                
                state_list.append(s)
                action_list.append(a)
                reward_list.append(reward)
                loop_cnt += 1
                s = n_state
            else:
                loop_cnt_list.append(loop_cnt)
            
            # monta-carlo
            n_steps = len(state_list)
            for i in range(n_steps):
                s, a = state_list[i], action_list[i]
                G, t = 0, 0
                for j in range(i, n_steps):
                    G += gamma ** t * reward_list[j]
                    t += 1

                # future - now
                self.actor.Q[s][a] += learning_rate * (G - self.actor.Q[s][a])

            if e % 100 == 0:
                m_ = np.mean(loop_cnt_list[-50:])
                std_ = np.std(loop_cnt_list[-50:])
                print(f'Epoch [{e}]: balance last {m_:.2f} (+/- {std_:.3f}) times')
        return self.actor, loop_cnt_list
    
    def actor_play(self, env, epoches):
        for i in range(epoches):
            env.render()
            s = self.take_state(s)
            a = self.actor.policy(s)
            n_state, reward, done, info = env.step(a)
            s = n_state
    
    def actor_done(self, env):
        env.close()


env = gym.make('CartPole-v0')
actor = CartPoleActor(env, round_num=1)
trainer = CartPoleMontoCarlo(actor, env)
actor, loop_cnt_list = trainer.train(
    epoches=2000,
    gamma=0.9,
    learning_rate=0.1
)


# look rewards
smooth_cnt = []
for idx in range(len(loop_cnt_list)-1000):
    smooth_cnt.append(np.mean(loop_cnt_list[idx:idx+1000]))

plt.plot(smooth_cnt)
plt.show()
