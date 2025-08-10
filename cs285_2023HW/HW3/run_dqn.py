import time
import argparse
from agents.dqn_agent import DQNAgent
from env_configs.config import make_config
import os
import time
import gymnasium as gym
from gymnasium import wrappers
import numpy as np
import torch
from utils.utools import (
    from_numpy, to_numpy, init_gpu,
    Logger, make_logger,
    sample_n_trajectories
)
from utils.replay_buffer import MemoryEfficientReplayBuffer, ReplayBuffer
import tqdm


MAX_NVIDEO = 2


def run_training_loop(config: dict, logger: Logger, args: argparse.Namespace):
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment
    env = config["make_env"]()
    eval_env = config["make_env"]()
    render_env = config["make_env"](render=True)
    exploration_schedule = config["exploration_schedule"]
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    assert discrete, "DQN only supports discrete action spaces"

    agent = DQNAgent(
        env.observation_space.shape,
        env.action_space.n,
        **config["agent_kwargs"],
    )
    agent.train()
    # simulation timestep, will be used for video saving
    if "model" in dir(env):
        fps = 1 / env.model.opt.timestep
    elif "render_fps" in env.env.metadata:
        fps = env.env.metadata["render_fps"]
    else:
        fps = 4

    ep_len = env.spec.max_episode_steps

    observation = None

    # Replay buffer
    if len(env.observation_space.shape) == 3:
        stacked_frames = True
        frame_history_len = env.observation_space.shape[0]
        assert frame_history_len == 4, "only support 4 stacked frames"
        replay_buffer = MemoryEfficientReplayBuffer(
            frame_history_len=frame_history_len
        )
    elif len(env.observation_space.shape) == 1:
        stacked_frames = False
        replay_buffer = ReplayBuffer()
    else:
        raise ValueError(
            f"Unsupported observation space shape: {env.observation_space.shape}"
        )

    def reset_env_training():
        nonlocal observation

        observation, _ = env.reset()

        assert not isinstance(
            observation, tuple
        ), "env.reset() must return np.ndarray - make sure your Gym version uses the old step API"
        observation = np.asarray(observation)

        if isinstance(replay_buffer, MemoryEfficientReplayBuffer):
            replay_buffer.on_reset(observation=observation[-1, ...])

    reset_env_training()
    tq_bar = tqdm.trange(config["total_steps"], dynamic_ncols=True)
    for step in tq_bar:
        agent.train()
        epsilon = exploration_schedule.value(step)
        
        # TODO(student): Compute action
        action = agent.get_action(observation, epsilon)

        # TODO(student): Step the environment
        n_s, reward, terminated, truncated, info = env.step(action)
        next_observation = np.asarray(n_s)
        done = np.logical_or(terminated, truncated)

        # TODO(student): Add the data to the replay buffer
        if isinstance(replay_buffer, MemoryEfficientReplayBuffer):
            # We're using the memory-efficient replay buffer,
            # so we only insert next_observation (not observation)
            replay_buffer.insert(
                action,
                reward,
                next_observation,
                done
            )
        else:
            # We're using the regular replay buffer
            replay_buffer.insert(
                observation,
                action,
                reward,
                next_observation,
                done
            )

        # Handle episode termination
        if done:
            reset_env_training()

            logger.log_scalar(info["episode"]["r"], "train_return", step)
            logger.log_scalar(info["episode"]["l"], "train_ep_len", step)
        else:
            observation = next_observation
        
        update_info = None
        # Main DQN training loop
        if step >= config["learning_starts"]:
            agent.train()
            # TODO(student): Sample config["batch_size"] samples from the replay buffer
            batch = replay_buffer.sample(config["batch_size"])

            # Convert to PyTorch tensos
            # TODO(student): Train the agent. `batch` is a dictionary of numpy arrays,
            update_info = agent.update(
                from_numpy(batch["observations"]),
                from_numpy(batch["actions"]),
                from_numpy(batch["rewards"]),
                from_numpy(batch["next_observations"]),
                from_numpy(batch["dones"]),
                step
            ) 
            # Logging code
            update_info["epsilon"] = epsilon
            update_info["lr"] = agent.lr_scheduler.get_last_lr()[0]

            if step % args.log_interval == 0:
                for k, v in update_info.items():
                    logger.log_scalar(v, k, step)
                logger.flush()

        if step % args.eval_interval == 0:
            agent.eval()
            # Evaluate
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

    parser.add_argument("--eval_interval", "-ei", type=int, default=10000)
    parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=10)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--log_interval", type=int, default=1000)
    parser.add_argument('--logdir', type=str, default='./data/run_summary')
    parser.add_argument("--video_log_freq", "-nvid", type=int, default=-1)
    args = parser.parse_args()

    # create directory for logging
    logdir_prefix = "hw3_dqn_"  # keep for autograder

    config = make_config(args.config_file)
    logger = make_logger(args.logdir, logdir_prefix, config)

    run_training_loop(config, logger, args)


if __name__ == "__main__":
    main()