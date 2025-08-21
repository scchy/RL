# python3
# Create Date: 2025-07-03
# Author: Scc_hy
# Func: Batch RL-training Func
# ===================================================================================
import gymnasium as gym
import swanlab as wandb
import numpy as np 
from tqdm.auto import tqdm 
import torch
from datetime import datetime
from loguru import logger 
import copy
from collections import deque
from .utils import load_mujoco_data, DataLoader, epDataset, rtgEpDataset

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
    
    logger.info(f'Save model -> {cfg.save_path} {info}')


@torch.no_grad()
def play(
        env_in, 
        env_agent, 
        cfg, 
        episode_count=2, 
        play_without_seed=False, 
        render=True, 
        rank=None
    ):
    """
    对训练完成的Agent进行游戏
    """
    try:
        env_agent.eval()
    except Exception as e:
        pass 
    max_steps = cfg.test_max_episode_steps if hasattr(cfg, "test_max_episode_steps") else cfg.max_episode_steps
    env = copy.deepcopy(env_in)
    ep_reward_record = []
    for e in range(episode_count):
        final_seed = np.random.randint(0, 999999) if play_without_seed else cfg.seed
        s, _ = env.reset(seed=final_seed)
        done = False
        episode_reward = 0
        episode_cnt = 0
        while not done:
            if render:
                env.render()
            try:
                a = env_agent.policy(s)
            except Exception as e:
                a, _ = env_agent.policy(s)

            try:
                n_state, reward, terminated, truncated, info = env.step(a)
            except Exception as e:  # Atari
                n_state, reward, terminated, truncated, info = env.step(a[0])

            done = (terminated or truncated)
            episode_reward += reward
            episode_cnt += 1
            # episode
            s = n_state
            if done:
                break
            if (episode_reward >= cfg.max_episode_rewards) or (episode_cnt >= max_steps):
                break
        
        ep_reward_record.append(episode_reward)
        add_str = f'(rk={rank})' if rank is not None else ''
        logger.info(f'[ {add_str}seed={final_seed} ] Get reward {episode_reward:.2f}. Last {episode_cnt} times')
    
    if render:
        env.close()

    try:
        env_agent.train()
    except Exception as e:
        pass 
    logger.info(f'[ {add_str}PLAY ] Get reward {np.mean(ep_reward_record):.2f}.')
    return np.mean(ep_reward_record) # np.percentile(ep_reward_record, 50)


def iter_ep_date(num_epoches, collected_data, bc_flag=False):
    idx = 1
    while (idx < num_epoches):
        for batch in collected_data.iterate_episodes():
            idx += 1
            if bc_flag and idx == 3:
                idx += num_epoches # 直接截断 仅用2个数据进行训练
            yield batch
            if idx - 1 >= num_epoches:
                break 

        if bc_flag:
            idx = 1


@logger.catch
def batch_rl_training(
        agnetAlgo, 
        cfg,
        env_name,
        data_level='simple',
        collected_data=None,
        test_episode_freq=10,
        episode_count=2,
        play_without_seed=False, 
        render=False, 
        wandb_flag=False,
        wandb_project_name='batch_rl_training',
        rank=None,
        name_str=''
    ):
    env = gym.make(env_name)
    if collected_data is None:
        collected_data = load_mujoco_data(env_name, data_level)
    if wandb_flag:
        wandb.login()
        cfg_dict = cfg.__dict__
        env_id = str(env).split('>')[0].split('<')[-1]
        algo = agnetAlgo.__class__.__name__
        now_ = datetime.now().strftime('%Y%m%d__%H%M')
        wandb.init(
            project=wandb_project_name,
            name=f"{algo}{name_str}__{env_id}__{now_}",
            config=cfg_dict,
            monitor_gym=True
        )
    iter_collecter = iter_ep_date(cfg.num_epoches, collected_data, getattr(agnetAlgo, 'bc_flag', False))
    ep_bar = tqdm(range(cfg.num_epoches), total=cfg.num_epoches)
    ep_que = deque(maxlen=10)
    best_r = recent_p = -np.inf
    for ep in ep_bar:
        episode_data = next(iter_collecter)
        ep_bar.set_description(f'[ {ep} / {cfg.num_epoches} ]')
        dloader = DataLoader(epDataset(episode_data), batch_size=cfg.batch_size, shuffle=True)
        min_q_collect = []
        for batch in dloader:
            if getattr(agnetAlgo, 'bc_flag', False):
                min_q = agnetAlgo.bc_update(batch, wandb_w=wandb if wandb_flag else None)
            else:
                min_q = agnetAlgo.update(batch, wandb_w=wandb if wandb_flag else None)
            min_q_collect.append(min_q)
        if (ep+1) % test_episode_freq == 0:
            ep_p = play(
                env, 
                agnetAlgo, 
                cfg, 
                episode_count=episode_count, 
                play_without_seed=play_without_seed, 
                render=render, 
                rank=None
            )
            ep_que.append(ep_p)
            recent_p = np.mean(ep_que)
            if best_r <= ep_p:
                best_r = ep_p
                save_agent_model(agnetAlgo, cfg, info='[ test_ep_freq-SAVE ]')

        min_q_ = np.mean(min_q_collect)
        ep_bar.set_postfix({
            'recent_reward':  f"{recent_p:.3f}", 
            'best_reward': f"{best_r:.3f}",
            'min_q': f'{min_q_:.3f}'
        })
        if wandb_flag:
            log_dict = {
                "recent_reward": recent_p,
                'best_reward': best_r
            }
            wandb.log(log_dict)

    if wandb_flag:
        wandb.finish()
    return agnetAlgo


@torch.no_grad()
def dt_play(
        env_in, 
        env_agent, 
        cfg, 
        episode_count=2, 
        play_without_seed=False, 
        render=False
    ):
    try:
        env_agent.eval()
    except Exception as e:
        pass 
    state_dim = cfg.state_dim
    act_dim = cfg.action_dim
    device = cfg.device
    max_steps = cfg.test_max_episode_steps if hasattr(cfg, "test_max_episode_steps") else cfg.max_episode_steps
    env = copy.deepcopy(env_in)
    ep_reward_record = []
    rtg_scale = cfg.DT_kwargs.get('rtg_scale', 1.0)
    for e in range(episode_count):
        final_seed = np.random.randint(0, 999999) if play_without_seed else cfg.seed
        # init 
        state, _ = env.reset(seed=final_seed)
        states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
        actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)
        ep_return = cfg.target_return/rtg_scale
        target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
        timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

        done = False
        episode_reward = 0
        episode_cnt = 0
        while not done:
            if render:
                env.render()
            # prepare for DT
            actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=device)])
            action = env_agent.get_action(
                # (states.to(dtype=torch.float32) - state_mean) / state_std,
                states.to(dtype=torch.float32),
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
            )
            actions[-1] = action
            action = action.detach().cpu().numpy()
            # env step
            try:
                n_state, reward, terminated, truncated, info = env.step(action)
            except Exception as e:  
                n_state, reward, terminated, truncated, info = env.step(action[0])

            done = (terminated or truncated)

            # prepare for DT
            cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)
            rewards[-1] = reward
            pred_return = target_return[0,-1] - (reward/rtg_scale)
            target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (episode_cnt + 1)], dim=1)

            episode_reward += reward
            episode_cnt += 1
            if done:
                break
            if (episode_reward >= cfg.max_episode_rewards) or (episode_cnt >= max_steps):
                break

        ep_reward_record.append(episode_reward)
        add_str = ''
        logger.info(f'[ {add_str}seed={final_seed} ] Get reward {episode_reward:.2f}. Last {episode_cnt} times')
    
    if render:
        env.close()

    try:
        env_agent.train()
    except Exception as e:
        pass 
    logger.info(f'[ {add_str}PLAY ] Get reward {np.mean(ep_reward_record):.2f}.')
    return np.mean(ep_reward_record)


@logger.catch
def DT_training(
        agnetAlgo, 
        cfg,
        env_name,
        data_level='simple',
        collected_data=None,
        test_episode_freq=10,
        episode_count=2,
        play_without_seed=False, 
        render=False, 
        wandb_flag=False,
        wandb_project_name='DT_training',
        rank=None,
        name_str=''
    ):
    env = gym.make(env_name)
    if collected_data is None:
        collected_data = load_mujoco_data(env_name, data_level)
    if wandb_flag:
        wandb.login()
        cfg_dict = cfg.__dict__
        env_id = str(env).split('>')[0].split('<')[-1]
        algo = agnetAlgo.__class__.__name__
        now_ = datetime.now().strftime('%Y%m%d__%H%M')
        wandb.init(
            project=wandb_project_name,
            name=f"{algo}{name_str}__{env_id}__{now_}",
            config=cfg_dict,
            monitor_gym=True
        )
    iter_collecter = iter_ep_date(cfg.num_epoches, collected_data)
    ep_bar = tqdm(range(cfg.num_epoches), total=cfg.num_epoches)
    ep_que = deque(maxlen=10)
    best_r = recent_p = -np.inf
    for ep in ep_bar:
        episode_data = next(iter_collecter)
        ep_bar.set_description(f'[ {ep} / {cfg.num_epoches} ]')
        dloader = DataLoader(rtgEpDataset(episode_data, K=cfg.K), batch_size=cfg.batch_size, shuffle=True)
        min_q_collect = []
        agnetAlgo.train()
        for batch in dloader:
            min_q = agnetAlgo.update(batch, wandb_w=wandb if wandb_flag else None)
            min_q_collect.append(min_q)

        if (ep+1) % test_episode_freq == 0:
            ep_p = dt_play(
                env, 
                agnetAlgo, 
                cfg, 
                episode_count=episode_count, 
                play_without_seed=play_without_seed, 
                render=render
            )
            ep_que.append(ep_p)
            recent_p = np.mean(ep_que)
            if best_r <= ep_p:
                best_r = ep_p
                save_agent_model(agnetAlgo, cfg, info='[ test_ep_freq-SAVE ]')

        min_q_ = np.mean(min_q_collect)
        ep_bar.set_postfix({
            'recent_reward':  f"{recent_p:.3f}", 
            'best_reward': f"{best_r:.3f}",
            'min_q': f'{min_q_:.3f}'
        })
        if wandb_flag:
            log_dict = {
                "recent_reward": recent_p,
                'best_reward': best_r
            }
            wandb.log(log_dict)

    if wandb_flag:
        wandb.finish()
    return agnetAlgo