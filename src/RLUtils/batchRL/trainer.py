# python3
# Create Date: 2025-07-03
# Author: Scc_hy
# Func: Batch RL-training Func
# ===================================================================================
import gymnasium as gym
import numpy as np 
from tqdm.auto import tqdm 
import torch
from datetime import datetime
from loguru import logger 
import copy
from collections import deque
from .utils import load_mujoco_data, DataLoader, epDataset

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


# def iter_ep_date(num_epoches, collected_data):
#     for ep in range(num_epoches):
#         collected_data.iterate_episodes()
    

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
        rank=None
    ):
    env = gym.make(env_name)
    if collected_data is None:
        collected_data = load_mujoco_data(env_name)
    if wandb_flag:
        wandb.login()
        cfg_dict = cfg.__dict__
        env_id = str(env).split('>')[0].split('<')[-1]
        algo = agnetAlgo.__class__.__name__
        now_ = datetime.now().strftime('%Y%m%d__%H%M')
        wandb.init(
            project=wandb_project_name,
            name=f"{algo}__{env_id}__{now_}",
            config=cfg_dict,
            monitor_gym=True
        )
    ep_bar = tqdm(zip(range(cfg.num_epoches), collected_data.iterate_episodes()), total=cfg.num_epoches)
    ep_que = deque(maxlen=10)
    best_r = recent_p = -np.inf
    for ep, episode_data in ep_bar:
        ep_bar.set_description(f'[ {ep} / {cfg.num_epoches} ]')
        dloader = DataLoader(epDataset(episode_data), batch_size=cfg.batch_size, shuffle=True)
        for batch in dloader:
            agnetAlgo.update(batch)

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

        ep_bar.set_postfix({'recent_reward':  f"{recent_p:.3f}", 'best_reward': f"{best_r:.3f}"})


