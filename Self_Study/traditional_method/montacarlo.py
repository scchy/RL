# python3
# Create date: 2021-12-21
# Author: Scc_hy
# Func: montacarlo
# ============================================

from _agent import Agent, env


class MontaCarlo(Agent):
    """
    mc = MontaCarlo(env)
    mc.play(epoches=3000, gamma=0.9, learning_rate=0.1, interval=20, render=False)
    mc.plot_reward(interval=50)
    """
    def __init__(self, env, epsilon=0.1):
        super(MontaCarlo, self).__init__(env=env, epsilon=epsilon)
    
    def play(self, epoches=100, gamma=0.9, learning_rate=0.1, interval=20, render=False):
        env_ = self.env
        self.record_reset()
        for e in range(epoches):
            s = env_.reset()
            done = False
            state_list = []
            action_list = []
            reward_list = []
            while not done:
                if render:
                    env_.render()
                a = self.policy(s)
                n_state, reward, done, info = env_.step(a)
                state_list.append(s)
                action_list.append(a)
                reward_list.append(reward)
                s = n_state
            else:
                self.record_reward(reward)
            if e % interval == 0:
                self.print_info(e, interval)
            # update
            steps = len(state_list)
            for i in range(steps):
                s, a = state_list[i], action_list[i]
                G , t = 0, 0
                for j in range(i, steps):
                    G += ( gamma ** t *  reward_list[j])
                    t += 1
            
                self.Q[s][a] += learning_rate * (G - self.Q[s][a])



mc = MontaCarlo(env)
mc.play(epoches=1000, gamma=0.9, learning_rate=0.1, interval=20, render=False)
mc.plot_reward(interval=50)