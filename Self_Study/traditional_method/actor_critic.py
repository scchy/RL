# python3
# Create date: 2021-12-21
# Author: Scc_hy
# Func: actor criticor
# ============================================

from _agent import Agent, env


class Critic:
    def __init__(self, env):
        self.env = env
        self.V = [0.0] * env.observation_space.n


class ActorCritic:
    """
    critic = Critic(env)
    actor = Agent(env)
    ac = ActorCritic(env, actor, critic)
    actor, criticor = ac.play(epoches=3000, gamma=0.9, learning_rate=0.1, interval=20, render=False)
    ac._actor.plot_reward(interval=50)
    """
    def __init__(self, env, actor, criticor):
        self.env = env
        self._actor = actor
        self._criticor = criticor
    
    def play(self, epoches=100, gamma=0.9, learning_rate=0.1, interval=20, render=False):
        actor = self._actor
        criticor = self._criticor
        env_ = self.env
        actor.record_reset()
        for e  in range(epoches):
            s = env.reset()
            done = False
            while not done:
                if render:
                    env_.render()
                a = actor.policy(s)
                n_state, reward, done, info = env_.step(a)
                
                gain = reward + gamma * criticor.V[n_state]
                td = gain - criticor.V[s]
                criticor.V[s] += learning_rate * td
                actor.Q[s][a] += learning_rate * td
                s = n_state
            else:
                actor.record_reward(reward)
            if e % interval == 0:
                actor.print_info(e, interval)
        return actor, criticor


critic = Critic(env)
actor = Agent(env)
ac = ActorCritic(env, actor, critic)
actor, criticor = ac.play(epoches=1000, gamma=0.9, learning_rate=0.1, interval=20, render=False)
ac._actor.plot_reward(interval=50)