
from .memory import replayBuffer
from .state_util import Pendulum_dis_to_con
import torch
from tqdm.auto import tqdm
import numpy as np
import wandb
import copy
from torch.optim.lr_scheduler import StepLR
from datetime import datetime 
from pynvml import (
    nvmlDeviceGetHandleByIndex, nvmlInit, nvmlDeviceGetMemoryInfo, 
    nvmlDeviceGetName,  nvmlShutdown, nvmlDeviceGetCount
)


def cuda_mem():
    # 21385MiB / 81920MiB
    fill = 0
    n = datetime.now()
    nvmlInit()
    # 创建句柄
    for i in range(nvmlDeviceGetCount()):
        handle = nvmlDeviceGetHandleByIndex(i)
        # 获取信息
        info = nvmlDeviceGetMemoryInfo(handle)
        # 获取gpu名称
        gpu_name = nvmlDeviceGetName(handle)
        # 查看型号、显存、温度、电源
        print("[ {} ]-[ GPU{}: {}".format(n, 0, gpu_name), end="    ")
        print("总共显存: {:.3}G".format((info.total // 1048576) / 1024), end="    ")
        print("空余显存: {:.3}G".format((info.free // 1048576) / 1024), end="    ")
        model_use = (info.used  // 1048576) - fill
        print("模型使用显存: {:.3}G({}MiB)".format( model_use / 1024, model_use))
    nvmlShutdown()


def save_agent_model(agent, cfg, info=None):
    if info is None:
        info = ""
    if hasattr(agent, "save_model"):
        agent.save_model(cfg.save_path)
    else:
        try:
            torch.save(agent.target_q.state_dict(), cfg.save_path)
        except Exception as e:
            torch.save(agent.actor.state_dict(), cfg.save_path)
    
    print(f'Save model -> {cfg.save_path} {info}')


def done_fix(done, episode_steps, max_episode_steps):
    if done and episode_steps != max_episode_steps:
        dw = True
    else:
        dw = False
    return dw


def random_action(env):
    asp = env.action_space
    try:
        return asp.sample()
    except Exception as e:
        return np.random.choice(asp.n)


def train_off_policy(env, agent ,cfg, 
                     action_contiguous=False, done_add=False, 
                     reward_func=None, train_without_seed=False,
                     wandb_flag=False,
                     step_lr_flag=False,
                     step_lr_kwargs=None,
                     update_every=1,
                     test_ep_freq=100,
                     test_episode_count=3,
                     done_fix_flag=False
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
            ep_reward = play(env, agent, cfg, episode_count=test_episode_count, play_without_seed=train_without_seed, render=False)
            if ep_reward > best_ep_reward:
                best_ep_reward = ep_reward
                # 模型保存
                save_agent_model(agent, cfg, info='[ test_ep_freq-SAVE ]')

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
                if policy_idx == 0:
                    rewards_list = []
                    print('Finished collect orginal data')

                policy_idx += 1
            if action_contiguous:
                c_a = Pendulum_dis_to_con(a, env, cfg.action_dim)
                n_s, r, terminated, truncated, _ = env.step([c_a])
            else:
                n_s, r, terminated, truncated, _ = env.step(a)
            
            done = terminated or truncated
            steps += 1
            tt_steps += 1

            mem_done = done_fix(done, steps, cfg.max_episode_steps) if done_fix_flag else done
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
        
        if (len(buffer) >= cfg.off_minimal_size) and (i >= 10):
            rewards_list.append(episode_rewards)
            now_reward = np.mean(rewards_list[-10:])
        if (bf_reward < now_reward) and (i >= 10) and (len(buffer) >= cfg.off_minimal_size):
            # best 时也进行测试
            ep_reward = play(env, agent, cfg, episode_count=test_episode_count, play_without_seed=train_without_seed, render=False)

            if ep_reward > best_ep_reward:
                best_ep_reward = ep_reward
                # 模型保存
                save_agent_model(agent, cfg, info='[ now_reward-SAVE ]')

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


@torch.no_grad()
def play(env_in, env_agent, cfg, episode_count=2, action_contiguous=False, play_without_seed=False, render=True):
    """
    对训练完成的Agent进行游戏
    """
    env = copy.deepcopy(env_in)
    try:
        env_agent.eval()
    except Exception as e:
        pass
    ep_reward_record = []
    for e in range(episode_count):
        final_seed = np.random.randint(0, 9999) if play_without_seed else cfg.seed
        s, _ = env.reset(seed=final_seed)
        done = False
        episode_reward = 0
        episode_cnt = 0
        while not done:
            if render:
                env.render()
            a = env_agent.policy(s)
            if action_contiguous:
                c_a = Pendulum_dis_to_con(a, env, cfg.action_dim)
                n_state, reward, terminated, truncated, _ = env.step([c_a])
            else:
                n_state, reward, terminated, truncated, _ = env.step(a)
            
            done = terminated or truncated
            episode_reward += reward
            episode_cnt += 1
            s = n_state
            if done:
                break
            if (episode_reward >= 3 * cfg.max_episode_rewards) or (episode_cnt >= 3 * cfg.max_episode_steps):
                break
        ep_reward_record.append(episode_reward)

        print(f'[ seed={final_seed} ] Get reward {episode_reward}. Last {episode_cnt} times')
    
    if render:
        env.close()

    try:
        env_agent.train()
    except Exception as e:
        pass

    return np.percentile(ep_reward_record, 50)



def train_on_policy(env, agent, cfg, 
                    wandb_flag=False, 
                    train_without_seed=False, 
                    step_lr_flag=False, 
                    step_lr_kwargs=None, 
                    test_ep_freq=100,
                    online_collect_nums=None,
                    test_episode_count=3,
                    done_fix_flag=False,
                    *args,
                    **kwargs
                    ):
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
    best_ep_reward = -np.inf
    buffer_ = replayBuffer(cfg.off_buffer_size)
    online_collect_flag = online_collect_nums is not None
    online_collect_deque_list = []
    online_collect_count = 0
    policy_idx = 0
    for i in tq_bar:
        if update_flag:
            buffer_ = replayBuffer(cfg.off_buffer_size)
            
        if (1 + i) % test_ep_freq == 0 and policy_idx > 0:
            ep_reward = play(env, agent, cfg, episode_count=test_episode_count, play_without_seed=train_without_seed, render=False)
            
            if ep_reward > best_ep_reward:
                best_ep_reward = ep_reward
                # 模型保存
                save_agent_model(agent, cfg)

        tq_bar.set_description(f'Episode [ {i+1} / {cfg.num_episode} ](minibatch={mini_b})')    
        rand_seed = np.random.randint(0, 9999)
        final_seed = rand_seed if train_without_seed else cfg.seed
        s, _ = env.reset(seed=final_seed)
        done = False
        episode_rewards = 0
        steps = 0
        while not done:
            if sum([len(i) for i in online_collect_deque_list]) < cfg.off_minimal_size and policy_idx == 0:
                a = random_action(env)
            else:
                a = agent.policy(s)
                if policy_idx == 0:
                    rewards_list = []
                    print('Finished collect orginal data')
                    policy_idx += 1

            n_s, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            steps += 1
            mem_done = done 
            if done_fix_flag:
                mem_done = done_fix(done, steps, cfg.max_episode_steps)
            buffer_.add(s, a, r, n_s, mem_done)
            s = n_s
            episode_rewards += r
            if (episode_rewards >= cfg.max_episode_rewards) or (steps >= cfg.max_episode_steps):
                break
        
        if online_collect_flag and online_collect_count < online_collect_nums:
            online_collect_count += len(buffer_)
            online_collect_deque_list.append(buffer_.buffer)
            update_flag = True
        elif(online_collect_flag and online_collect_count >= online_collect_nums):
            update_flag = agent.update(online_collect_deque_list)
            online_collect_count = 0
            online_collect_deque_list = []
            # print('collect training')
        else:
            update_flag = agent.update(buffer_.buffer)

        if step_lr_flag:
            schedule.step()
        rewards_list.append(episode_rewards)
        now_reward = np.mean(rewards_list[-10:])
        if (bf_reward < now_reward) and (i >= 10) and policy_idx > 0:
            # best 时也进行测试
            ep_reward = play(env, agent, cfg, episode_count=test_episode_count, play_without_seed=train_without_seed, render=False)

            if ep_reward > best_ep_reward:
                best_ep_reward = ep_reward
                # 模型保存
                save_agent_model(agent, cfg)

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

        
    env.close()
    if wandb_flag:
        wandb.finish()
    return agent



@torch.no_grad()
def random_play(env_in, episode_count=3, render=True, play_without_seed=False):

    env = copy.deepcopy(env_in)
    ep_reward_record = []
    for e in range(episode_count):
        final_seed = np.random.randint(0, 9999) if play_without_seed else 42
        s, _ = env.reset(seed=final_seed)
        done = False
        episode_reward = 0
        episode_cnt = 0
        while not done:
            if render:
                env.render()
            a = env_in.action_space.sample()
            n_state, reward, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            episode_reward += reward
            episode_cnt += 1
            s = n_state
            if done:
                break

        ep_reward_record.append(episode_reward)
        print(f'[ seed={final_seed} ] Get reward {episode_reward}. Last {episode_cnt} times')
    
    if render:
        env.close()
    return np.percentile(ep_reward_record, 50)

