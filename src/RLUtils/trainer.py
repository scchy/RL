
from .memory import replayBuffer
from .state_util import Pendulum_dis_to_con
import torch
from tqdm.auto import tqdm
import numpy as np
import wandb
from torch.optim.lr_scheduler import StepLR


def random_action(env):
    asp = env.action_space
    try:
        return np.random.uniform(low=asp.low, high=asp.high)
    except Exception as e:
        return np.random.choice(asp.n)


def train_off_policy(env, agent ,cfg, 
                     action_contiguous=False, done_add=False, reward_func=None, train_without_seed=False,
                     wandb_flag=False,
                     step_lr_flag=False,
                     step_lr_kwargs=None):
    if wandb_flag:
        wandb.login()
        cfg_dict = cfg.__dict__
        if step_lr_flag:
            cfg_dict['step_lr_flag'] = step_lr_flag
            cfg_dict['step_lr_kwargs'] = step_lr_kwargs
        wandb.init(
            project="RL-train_off_policy",
            config=cfg_dict
        )
    if step_lr_flag:
        opt = agent.actor_opt if hasattr(agent, "actor_opt") else agent.opt
        schedule = StepLR(opt, step_size=step_lr_kwargs['step_size'], gamma=step_lr_kwargs['gamma'])
    buffer = replayBuffer(cfg.off_buffer_size)
    tq_bar = tqdm(range(cfg.num_episode))
    rewards_list = []
    now_reward = -np.inf
    bf_reward = -np.inf
    for i in tq_bar:
        rand_seed = np.random.randint(0, 9999)
        final_seed = rand_seed if train_without_seed else cfg.seed
        s, _ = env.reset(seed=final_seed)
        tq_bar.set_description(f'Episode [ {i+1} / {cfg.num_episode}|(seed={final_seed}) ]')
        done = False
        episode_rewards = 0
        steps = 0
        drop_flag = False
        while not done:
            if len(buffer) < cfg.off_minimal_size:
                a = random_action(env)
            else:
                a = agent.policy(s)
            if action_contiguous:
                c_a = Pendulum_dis_to_con(a, env, cfg.action_dim)
                n_s, r, done, _, _ = env.step([c_a])
            else:
                n_s, r, done, _, _ = env.step(a)
            
            mem_done = done
            ep_r = r
            if reward_func is not None:
                try:
                    r = reward_func(r)
                except Exception as e:
                    r, mem_done = reward_func(r, mem_done)
            buffer.add(s, a, r, n_s, mem_done)
            s = n_s
            episode_rewards += ep_r
            steps += 1
            # buffer update
            if len(buffer) >= cfg.off_minimal_size:
                samples = buffer.sample(cfg.sample_size)
                agent.update(samples)
                # print('Start Update')
            if (episode_rewards >= cfg.max_episode_rewards) or (steps >= cfg.max_episode_steps):
                drop_flag = True
                break
        
        if done_add and drop_flag:
            for _ in range(steps):
                buffer.buffer.pop()
            # print(f'\ndrop not done experience-{steps}')
        
        rewards_list.append(episode_rewards)
        now_reward = np.mean(rewards_list[-10:])
        if (bf_reward < now_reward) and (i >= 10):
            if hasattr(agent, "save_model"):
                agent.save_model(cfg.save_path)
            else:
                try:
                    torch.save(agent.target_q.state_dict(), cfg.save_path)
                except Exception as e:
                    torch.save(agent.actor.state_dict(), cfg.save_path)
            bf_reward = now_reward

        tq_bar.set_postfix({
            "steps": steps,
            'lastMeanRewards': f'{now_reward:.2f}',
            'BEST': f'{bf_reward:.2f}'
        })
        if wandb_flag:
            log_dict = {
                "steps": steps,
                'lastMeanRewards': now_reward,
                'BEST': bf_reward,
                "episodeRewards": episode_rewards
            }
            if step_lr_flag:
                log_dict['actor_lr'] = opt.param_groups[0]['lr']
            wandb.log(log_dict)
        if step_lr_flag:
            schedule.step()
    env.close()
    if wandb_flag:
        wandb.finish()
    return agent


def play(env, env_agent, cfg, episode_count=2, action_contiguous=False, play_without_seed=False):
    """
    对训练完成的QNet进行策略游戏
    """
    for e in range(episode_count):
        rand_seed = np.random.randint(0, 9999)
        s, _ = env.reset(seed=rand_seed if play_without_seed else cfg.seed)
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



def train_on_policy(env, agent, cfg, wandb_flag=False):
    if wandb_flag:
        wandb.login()
        wandb.init(
            project="RL-train_on_policy",
            config=cfg.__dict__
        )
    try:
        mini_b = cfg.PPO_kwargs.get('minibatch_size', 12)
    except Exception as e:
        mini_b = 12
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
        s, _ = env.reset(seed=cfg.seed)
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
            if hasattr(agent, "save_model"):
                agent.save_model(cfg.save_path)
            else:
                try:
                    torch.save(agent.target_q.state_dict(), cfg.save_path)
                except Exception as e:
                    torch.save(agent.actor.state_dict(), cfg.save_path)
            bf_reward = now_reward
        
        tq_bar.set_postfix({"steps": steps,'lastMeanRewards': f'{now_reward:.2f}', 'BEST': f'{bf_reward:.2f}'})
        if wandb_flag:
            wandb.log({
                "steps": steps,
                'lastMeanRewards': now_reward,
                'BEST': bf_reward,
                "episodeRewards": episode_rewards
            })
        
    env.close()
    if wandb_flag:
        wandb.finish()
    return agent