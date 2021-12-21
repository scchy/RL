# python3
# Create date: 2021-12-21
# Author: Scc_hy
# Func: SARS
# ============================================

from _agent import Agent, env

class SARS(Agent):
    """
    sars = SARS(env)
    sars.play(epoches=3000, gamma=0.9, learning_rate=0.1, interval=20, render=False)
    sars.plot_reward(interval=50)
    """
    def __init__(self, env, epsilon=0.1):
        super(SARS, self).__init__(env=env, epsilon=epsilon)
    
    def play(self, epoches=100, gamma=0.9, learning_rate=0.1, interval=20, render=False):
        env_ = self.env
        self.record_reset()
        for e in range(epoches):
            s = env_.reset()
            a = self.policy(s)
            done = False
            while not done:
                if render:
                    env_.render()
                n_state, reward, done, info = env_.step(a)
                n_a = self.policy(n_state)
                
                gain = reward + gamma * self.Q[n_state][n_a]
                self.Q[s][a] += learning_rate * (gain - self.Q[s][a])
                s = n_state
                a = n_a
            else:
                self.record_reward(reward)
            if e % interval == 0:
                self.print_info(e, interval)
                

sars = SARS(env)
sars.play(epoches=1000, gamma=0.9, learning_rate=0.1, interval=20, render=False)
sars.plot_reward(interval=50)