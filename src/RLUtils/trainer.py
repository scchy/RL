
from .memory import replayBuffer
from .state_util import Pendulum_dis_to_con
import torch
from tqdm.auto import tqdm
import numpy as np



def train_off_policy(env, agent ,cfg, action_contiguous=False):
    buffer = replayBuffer(cfg.off_buffer_size)
    tq_bar = tqdm(range(cfg.num_episode))
    rewards_list = []
    now_reward = -np.inf
    bf_reward = -np.inf
    for i in tq_bar:
        tq_bar.set_description(f'Episode [ {i+1} / {cfg.num_episode} ]')
        s, _ = env.reset()
        done = False
        episode_rewards = 0
        steps = 0
        while not done:
            a = agent.policy(s)
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
            if len(buffer) > cfg.off_minimal_size:
                samples = buffer.sample(cfg.sample_size)
                agent.update(samples)
            if (episode_rewards >= cfg.max_episode_rewards) or (steps >= cfg.max_episode_steps):
                break
        
        rewards_list.append(episode_rewards)
        now_reward = np.mean(rewards_list[-10:])
        if (bf_reward < now_reward) and (i >= 10):
            try:
                torch.save(agent.target_q.state_dict(), cfg.save_path)
            except Exception as e:
                torch.save(agent.actor.state_dict(), cfg.save_path)
            bf_reward = now_reward

        tq_bar.set_postfix({
            'lastMeanRewards': f'{now_reward:.2f}',
            'BEST': f'{bf_reward:.2f}'
        })
    env.close()
    return agent


def play(env, env_agent, cfg, episode_count=2, action_contiguous=False):
    """
    对训练完成的QNet进行策略游戏
    """
    for e in range(episode_count):
        s, _ = env.reset()
        done = False
        episode_reward = 0
        episode_cnt = 0
        while not done:
            env.render()
            a = env_agent.policy(s)
            if action_contiguous:
                c_a = Pendulum_dis_to_con(a, env, cfg.action_dim)
                n_state, reward, done, _, _ = env.step([c_a])
            else:
                n_state, reward, done, _, _ = env.step(a)
            episode_reward += reward
            episode_cnt += 1
            s = n_state
            if (episode_reward >= 3 * cfg.max_episode_rewards) or (episode_cnt >= 3 * cfg.max_episode_steps):
                break

        print(f'Get reward {episode_reward}. Last {episode_cnt} times')

    env.close()



def train_on_policy(env, agent, cfg):
    mini_b = cfg.PPO_kwargs['minibatch_size']
    tq_bar = tqdm(range(cfg.num_episode))
    rewards_list = []
    now_reward = 0
    bf_reward = -np.inf
    update_flag = False
    buffer_ = replayBuffer(cfg.off_buffer_size)
    for i in tq_bar:
        if update_flag:
            buffer_ = replayBuffer(cfg.off_buffer_size)

        tq_bar.set_description(f'Episode [ {i+1} / {cfg.num_episode} ](minibatch={mini_b})')    
        s, _ = env.reset()
        done = False
        episode_rewards = 0
        steps = 0
        while not done:
            a = agent.policy(s)
            n_s, r, done, _, _ = env.step(a)
            buffer_.add(s, a, r, n_s, done)
            s = n_s
            episode_rewards += r
            steps += 1
            if (episode_rewards >= cfg.max_episode_rewards) or (steps >= cfg.max_episode_steps):
                break

        update_flag = agent.update(buffer_.buffer)
        rewards_list.append(episode_rewards)
        now_reward = np.mean(rewards_list[-10:])
        if (bf_reward < now_reward) and (i >= 10):
            torch.save(agent.actor.state_dict(), cfg.save_path)
            bf_reward = now_reward
        
        tq_bar.set_postfix({'lastMeanRewards': f'{now_reward:.2f}', 'BEST': f'{bf_reward:.2f}'})
    env.close()
    return agent