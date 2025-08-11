import os
import time
import yaml

from agents.soft_actor_critic import SoftActorCritic
import gymnasium as gym
import numpy as np
import torch
import tqdm

from utils.utools import (
    from_numpy, to_numpy, init_gpu,
    Logger, make_logger,
    sample_n_trajectories
)
from env_configs.config import make_config
from utils.replay_buffer import MemoryEfficientReplayBuffer, ReplayBuffer
import tqdm
import argparse


def run_training_loop(config: dict, logger: Logger, args: argparse.Namespace):
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment
    env = config["make_env"]()
    eval_env = config["make_env"]()
    render_env = config["make_env"](render=False)

    ep_len = config["ep_len"] or env.spec.max_episode_steps
    batch_size = config["batch_size"] or batch_size

    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    assert (
        not discrete
    ), "Our actor-critic implementation only supports continuous action spaces. (This isn't a fundamental limitation, just a current implementation decision.)"

    ob_shape = env.observation_space.shape
    ac_dim = env.action_space.shape[0]

    # simulation timestep, will be used for video saving
    if "model" in dir(env):
        fps = 1 / env.model.opt.timestep
    else:
        fps = env.env.metadata["render_fps"]

    # initialize agent
    agent = SoftActorCritic(
        ob_shape,
        ac_dim,
        **config["agent_kwargs"],
    )

    replay_buffer = ReplayBuffer(config["replay_buffer_capacity"])

    observation, _ = env.reset()
    tq_bar = tqdm.trange(config["total_steps"], dynamic_ncols=True)
    for step in tq_bar:
        agent.train()
        if step < config["random_steps"]:
            action = env.action_space.sample()
        else:
            # TODO(student): Select an action
            action = agent.get_action(observation)

        # Step the environment and add the data to the replay buffer
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = np.logical_or(terminated, truncated)
        replay_buffer.insert(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=done
        )

        if done:
            logger.log_scalar(info["episode"]["r"], "train_return", step)
            logger.log_scalar(info["episode"]["l"], "train_ep_len", step)
            observation, _ = env.reset()
        else:
            observation = next_observation

        # Train the agent
        update_info = None
        if step >= config["training_starts"]:
            agent.train()
            # TODO(student): Sample a batch of config["batch_size"] transitions from the replay buffer
            batch = replay_buffer.sample(config["batch_size"])
            update_info = agent.update(
                from_numpy(batch["observations"]),
                from_numpy(batch["actions"]),
                from_numpy(batch["rewards"]),
                from_numpy(batch["next_observations"]),
                from_numpy(batch["dones"]),
                step
            ) 

            # Logging
            update_info["actor_lr"] = agent.actor_lr_scheduler.get_last_lr()[0]
            update_info["critic_lr"] = agent.critic_lr_scheduler.get_last_lr()[0]

            if step % args.log_interval == 0:
                for k, v in update_info.items():
                    logger.log_scalar(v, k, step)
                    logger.log_scalars
                logger.flush()

        # Run evaluation
        if step % args.eval_interval == 0:
            agent.eval()
            trajectories = sample_n_trajectories(
                eval_env,
                agent,
                args.num_eval_trajectories,
                ep_len,
            )
            returns = [t["episode_statistics"]["r"] for t in trajectories]
            ep_lens = [t["episode_statistics"]["l"] for t in trajectories]

            logger.log_scalar(np.mean(returns), "eval_return", step)
            logger.log_scalar(np.mean(ep_lens), "eval_ep_len", step)

            p_info = {
                "EReward": np.mean(returns),
                "ERewardMax": np.max(returns),
                "ELen": np.mean(ep_lens),
            }
            if update_info is not None:
                for k, v in update_info.items():
                    if any(j in k for j in ['value', 'loss']):
                        p_info[k] = np.round(v, 5)
            tq_bar.set_postfix(p_info)

            if len(returns) > 1:
                logger.log_scalar(np.std(returns), "eval/return_std", step)
                logger.log_scalar(np.max(returns), "eval/return_max", step)
                logger.log_scalar(np.min(returns), "eval/return_min", step)
                logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", step)
                logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", step)
                logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", step)

            if  args.video_log_freq != -1 and step % args.video_log_freq == 0:
                video_trajectories = sample_n_trajectories(
                    render_env,
                    agent,
                    args.video_log_freq,
                    ep_len,
                    render=True,
                )

                logger.log_paths_as_videos(
                    video_trajectories,
                    step,
                    fps=fps,
                    max_videos_to_save=args.video_log_freq,
                    video_title="eval_rollouts",
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)

    parser.add_argument("--eval_interval", "-ei", type=int, default=5000)
    parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=10)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-g", default=0)
    parser.add_argument("--log_interval", type=int, default=1000)
    parser.add_argument('--logdir', type=str, default='./data/run_summary')
    parser.add_argument("--video_log_freq", "-nvid", type=int, default=-1)
    args = parser.parse_args()

    # create directory for logging
    logdir_prefix = "hw3_sac_"  # keep for autograder

    config = make_config(args.config_file)
    logger = make_logger(args.logdir, logdir_prefix, config)
    run_training_loop(config, logger, args)


if __name__ == "__main__":
    main()
