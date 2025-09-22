# python3
# Create Date: 2025-09-18
# Author: Scc_hy
# Func: offline
# Referenece: https://github.com/berkeleydeeprlcourse/homework_fall2023/blob/main/hw5/cs285/scripts/run_hw5_offline.py
# Doc:
# Tip:
# ===============================================================================================
import pickle
import time
import os 
import argparse
from agents import agents as agent_types
from agents.dqn_agent import DQNAgent
import gymnasium as gym 
import numpy as np 
from tqdm.auto import tqdm
import torch 
from env_configs.config import make_config 
from env_configs.env import Pointmass
from utils.replay_buffer import ReplayBuffer
from utils.utools import (
    from_numpy, to_numpy, Logger,
    make_logger, init_gpu,
    sample_n_trajectories, sample_trajectories, 
    RandomPolicy, device, from_numpy
)


def run_training_loop(config: dict, logger: Logger, args: argparse.Namespace):
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment
    env = config["make_env"]()
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    assert discrete, "DQN only supports discrete action spaces"
    agent_cls = agent_types[config["agent"]]
    agent = agent_cls(
        env.observation_space.shape,
        env.action_space.n,
        **config["agent_kwargs"],
    )
    ep_len = env.spec.max_episode_steps or env.max_episode_steps

    with open(f"{config['dataset_name']}", "rb") as f:
        dataset = pickle.load(f)
    eval_r = -float("inf")
    tq_bar = tqdm(range(config["training_steps"]))
    for step in tq_bar:
        # Train with offline RL
        batch = dataset.sample(config["batch_size"])
        batch = {
            k: from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in batch.items()
        }
        metrics = agent.update(
            batch["observations"],
            batch["actions"],
            batch["rewards"],
            batch["next_observations"],
            batch["dones"],
            step,
        )
        if step % args.log_interval == 0:
            for k, v in metrics.items():
                logger.log_scalar(v, k, step)

        if step % args.eval_interval == 0:
            # Evaluate
            trajectories = sample_n_trajectories(
                env,
                agent,
                args.num_eval_trajectories,
                ep_len,
            )
            returns = [t["episode_statistics"]["r"] for t in trajectories]
            ep_lens = [t["episode_statistics"]["l"] for t in trajectories]
            eval_r = np.mean(returns)
            logger.log_scalar(eval_r, "eval_return", step)
            logger.log_scalar(np.mean(ep_lens), "eval_ep_len", step)

            if len(returns) > 1:
                logger.log_scalar(np.std(returns), "eval/return_std", step)
                logger.log_scalar(np.max(returns), "eval/return_max", step)
                logger.log_scalar(np.min(returns), "eval/return_min", step)
                logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", step)
                logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", step)
                logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", step)

            env_pointmass: Pointmass = env.unwrapped
            figs_ = [
                env_pointmass.plot_trajectory(trajectory["next_observation"]) 
                for trajectory in trajectories
            ]
            logger.log_figures(
                figs_,
                "trajectories",
                step,
                "eval"
            )
            fig = figs_[-1]
            if not (os.path.exists(os.path.join(args.dataset_dir, "eval_visualization"))):
                os.makedirs(os.path.join(args.dataset_dir, "eval_visualization"))
            filename = os.path.join(args.dataset_dir, "eval_visualization", "eval_.png")
            fig.savefig(filename)


        metrics['eval_r'] = eval_r
        tq_bar.set_postfix(
            {k: np.round(v, 3) for k, v in metrics.items()}
        )



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)

    parser.add_argument("--eval_interval", "-ei", type=int, default=10000)
    parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=10)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument('--logdir', type=str, default='./data/run_summary')
    # parser.add_argument("--dataset_dir", type=str, required=True)

    args = parser.parse_args()

    # create directory for logging
    logdir_prefix = "hw5_offline_"  # keep for autograder

    config = make_config(args.config_file)
    logger = make_logger(args.logdir, logdir_prefix, config)
    args.dataset_dir = logger.get_log_dir()
    os.makedirs(args.dataset_dir, exist_ok=True)
    run_training_loop(config, logger, args)


if __name__ == "__main__":
    main()

