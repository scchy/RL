
from .memory import replayBuffer
from .state_util import Pendulum_dis_to_con
import torch
from tqdm.auto import tqdm
import numpy as np
import wandb
from torch.optim.lr_scheduler import StepLR


def done_fix(done, episode_steps, max_episode_steps):
    if done and episode_steps != max_episode_steps:
        dw = True
    else:
        dw = False
    return dw


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
                     step_lr_kwargs=None,
                     update_every=1,
                     test_ep_freq=100
                     ):
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
    best_ep_reward = -np.inf
    policy_idx = 0
    tt_steps = 0
    for i in tq_bar:
        if (1 + i) % test_ep_freq == 0:
            if hasattr(agent, "eval"):
                agent.eval()
            ep_reward = play(env, agent, cfg, episode_count=3, play_without_seed=train_without_seed, render=False)
            if hasattr(agent, "train"):
                agent.train()
            
            if ep_reward > best_ep_reward:
                best_ep_reward = ep_reward
                # 模型保存
                if hasattr(agent, "save_model"):
                    agent.save_model(cfg.save_path)
                else:
                    try:
                        torch.save(agent.target_q.state_dict(), cfg.save_path)
                    except Exception as e:
                        torch.save(agent.actor.state_dict(), cfg.save_path)

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
                # print('random_action a=', np.round(a, 3))
            else:
                a = agent.policy(s)
                if policy_idx == 0:
                    rewards_list = []
                    print('Finished collect orginal data')
                # if a[0] > -0.09:
                #     print('a=', np.round(a, 3))
                policy_idx += 1
            if action_contiguous:
                c_a = Pendulum_dis_to_con(a, env, cfg.action_dim)
                n_s, r, done, _, _ = env.step([c_a])
            else:
                n_s, r, done, _, _ = env.step(a)
            
            steps += 1
            tt_steps += 1
            mem_done = done_fix(done, steps, cfg.max_episode_steps)
            ep_r = r
            if reward_func is not None:
                try:
                    r = reward_func(r)
                except Exception as e:
                    r, mem_done = reward_func(r, mem_done)
                    
            # state, action, reward, next_state, done
            buffer.add(s, a, r, n_s, mem_done)
            s = n_s
            episode_rewards += ep_r
            # buffer update
            if (len(buffer) >= cfg.off_minimal_size) and (tt_steps % update_every == 0):
                samples = buffer.sample(cfg.sample_size)
                agent.update(samples)

            if (episode_rewards >= cfg.max_episode_rewards) or (steps >= cfg.max_episode_steps):
                drop_flag = True
                break
        
        if done_add and drop_flag:
            for _ in range(steps):
                buffer.buffer.pop()
            # print(f'\ndrop not done experience-{steps}')
        
        rewards_list.append(episode_rewards)
        now_reward = np.mean(rewards_list[-10:])
        if (bf_reward < now_reward) and (i >= 10) and (len(buffer) >= cfg.off_minimal_size):
            if hasattr(agent, "eval"):
                agent.eval()
            # best 时也进行测试
            ep_reward = play(env, agent, cfg, episode_count=3)
            if hasattr(agent, "train"):
                agent.train()
            
            if ep_reward > best_ep_reward:
                best_ep_reward = ep_reward
                # 模型保存
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
            'BEST': f'{bf_reward:.2f}',
            "bestTestReward": f'{best_ep_reward:.2f}'
        })
        if wandb_flag:
            log_dict = {
                "steps": steps,
                'lastMeanRewards': now_reward,
                'BEST': bf_reward,
                "episodeRewards": episode_rewards,
                "bestTestReward": best_ep_reward
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


def play(env, env_agent, cfg, episode_count=2, action_contiguous=False, play_without_seed=False, render=True):
    """
    对训练完成的QNet进行策略游戏
    """
    ep_reward_record = []
    for e in range(episode_count):
        rand_seed = np.random.randint(0, 9999)
        s, _ = env.reset(seed=rand_seed if play_without_seed else cfg.seed)
        done = False
        episode_reward = 0
        episode_cnt = 0
        while not done:
            if render:
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
        ep_reward_record.append(episode_reward)

        print(f'Get reward {episode_reward}. Last {episode_cnt} times')
    
    if render:
        env.close()
    return np.mean(ep_reward_record)



def train_on_policy(env, agent, cfg, wandb_flag=False, step_lr_flag=False, step_lr_kwargs=None):
    if wandb_flag:
        wandb.login()
        cfg_dict = cfg.__dict__
        if step_lr_flag:
            cfg_dict['step_lr_flag'] = step_lr_flag
            cfg_dict['step_lr_kwargs'] = step_lr_kwargs
        wandb.init(
            project="RL-train_on_policy",
            config=cfg_dict
        )
    try:
        mini_b = cfg.PPO_kwargs.get('minibatch_size', 12)
    except Exception as e:
        mini_b = 12
    if step_lr_flag:
        opt = agent.actor_opt if hasattr(agent, "actor_opt") else agent.opt
        schedule = StepLR(opt, step_size=step_lr_kwargs['step_size'], gamma=step_lr_kwargs['gamma'])

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
            steps += 1
            mem_done = done_fix(done, steps, cfg.max_episode_steps)
            buffer_.add(s, a, r, n_s, mem_done)
            s = n_s
            episode_rewards += r
            if (episode_rewards >= cfg.max_episode_rewards) or (steps >= cfg.max_episode_steps):
                break

        update_flag = agent.update(buffer_.buffer)
        if step_lr_flag:
            schedule.step()
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
            log_dict = {
                "steps": steps,
                'lastMeanRewards': now_reward,
                'BEST': bf_reward,
                "episodeRewards": episode_rewards
            }
            if step_lr_flag:
                log_dict['actor_lr'] = opt.param_groups[0]['lr']
            wandb.log(log_dict)

        
    env.close()
    if wandb_flag:
        wandb.finish()
    return agent
