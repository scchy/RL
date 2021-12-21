# python3
# Create date: 2021-12-16
# Author: Scc_hy
# Func: Qlearning
# ============================================

from _agent import Agent, env

class Qlearning(Agent):
    """Qlearning
    q_lr = Qlearning(env)
    q_lr.play(epoches=3000, gamma=0.9, learning_rate=0.1, interval=20, render=False)
    q_lr.plot_reward(interval=50)
    """
    def __init__(self, env, epsilon=0.1):
        super(Qlearning, self).__init__(env=env, epsilon=epsilon)
    
    def play(self, epoches=100, gamma=0.9, learning_rate=0.1, interval=20, render=False):
        self.record_reset()
        env_ = self.env
        print(env_)
        for e in range(epoches):
            s = env_.reset()
            done = False
            while not done:
                if render:
                    env_.render()
                a = self.policy(s)
                n_state, reward, done, info = env_.step(a)
                gain = reward + gamma * max(self.Q[n_state])
                self.Q[s][a] += learning_rate * (gain - self.Q[s][a])
                s = n_state
            else:
                self.record_reward(reward)
            if e % interval == 0:
                self.print_info(e, interval)
                


q_lr = Qlearning(env)
q_lr.play(epoches=3000, gamma=0.9, learning_rate=0.1, interval=20, render=False)
q_lr.plot_reward(interval=50)