# python3
# Create Date: 2025-09-17
# Author: Scc_hy
# Func: explore
# Referenece:
# Doc:
# Tip:
# ===============================================================================================
import os 
import time 
import argparse
import pickle 
from agents import agents as agent_types
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


def visualize(env: Pointmass, agent, observations: torch.Tensor):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 10))

    num_subplots = agent.num_aux_plots() + 1 if hasattr(agent, "num_aux_plots") else 1

    axes = fig.subplots(1, num_subplots)
    ax = axes[0] if num_subplots > 1 else axes
    env.plot_walls(ax)
    ax.scatter(
        observations[:, 0],
        observations[:, 1],
        alpha=0.1
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    env.plot_keypoints(ax)

    ax.axis("equal")

    if hasattr(agent, "plot_aux"):
        agent.plot_aux(axes[1:])

    return fig


def run_training_loop(config: dict, logger: Logger, args: argparse.Namespace):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)
    env = config["make_env"]()
    exploration_schedule = config.get("exploration_schedule", None)
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    assert discrete, "DQN only supports discrete action spaces"

    agent_cls = agent_types[config["agent"]]
    agent = agent_cls(
        env.observation_space.shape,
        env.action_space.n,
        **config["agent_kwargs"],
    )
    ep_len = env.spec.max_episode_steps or env.max_episode_steps

    observation = None

    replay_buffer = ReplayBuffer(capacity=config["total_steps"])

    observation, _ = env.reset()

    recent_observations = []
    tq_bar = tqdm(range(config["total_steps"]))
    for step in tq_bar:
        if exploration_schedule is not None:
            epsilon = exploration_schedule.value(step)
            action = agent.get_action(observation, epsilon)
        else:
            epsilon = None
            action = agent.get_action(observation)

        n_s, reward, terminated, truncated, info = env.step(action)
        next_observation = np.asarray(n_s)
        done = np.logical_or(terminated, truncated)

        replay_buffer.insert(
            observation=observation,
            action=action,
            reward=reward,
            done=done and not truncated,
            next_observation=next_observation,
        )
        recent_observations.append(observation)

        # Handle episode termination
        if done:
            observation, _ = env.reset()

            logger.log_scalar(info["episode"]["r"], "train_return", step)
            logger.log_scalar(info["episode"]["l"], "train_ep_len", step)
        else:
            observation = next_observation

        # Main training loop
        batch = replay_buffer.sample(config["batch_size"])

        # Convert to PyTorch tensors
        update_info = agent.update(
            from_numpy(batch["observations"]),
            from_numpy(batch["actions"]),
            from_numpy(batch["rewards"]) * (1 if config.get("use_reward", False) else 0),
            from_numpy(batch["next_observations"]),
            from_numpy(batch["dones"]),
            step,
        )
        # Logging code
        if epsilon is not None:
            update_info["epsilon"] = epsilon

        if step % args.log_interval == 0:
            for k, v in update_info.items():
                logger.log_scalar(v, k, step)
            logger.flush()


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

            logger.log_scalar(np.mean(returns), "eval_return", step)
            logger.log_scalar(np.mean(ep_lens), "eval_ep_len", step)

            if len(returns) > 1:
                logger.log_scalar(np.std(returns), "eval/return_std", step)
                logger.log_scalar(np.max(returns), "eval/return_max", step)
                logger.log_scalar(np.min(returns), "eval/return_min", step)
                logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", step)
                logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", step)
                logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", step)

        if step % args.visualize_interval == 0:
            env_pointmass: Pointmass = env.unwrapped
            observations = np.stack(recent_observations)
            recent_observations = []
            logger.log_figure(
                visualize(env_pointmass, agent, observations),
                "exploration_trajectories",
                step,
                "eval"
            )

    # Save the final dataset
    dataset_file = os.path.join(args.dataset_dir, f"{config['dataset_name']}.pkl")
    with open(dataset_file, "wb") as f:
        pickle.dump(replay_buffer, f)
        print("Saved dataset to", dataset_file)
    
    # Render final heatmap
    fig = visualize(env_pointmass, agent, replay_buffer.observations[:config["total_steps"]])
    fig.suptitle("State coverage")
    if not (os.path.exists(os.path.join(args.dataset_dir, "exploration_visualization"))):
        os.makedirs(os.path.join(args.dataset_dir, "exploration_visualization"))
    filename = os.path.join(args.dataset_dir, "exploration_visualization", f"{config['log_name']}.png")
    fig.savefig(filename)
    print("Saved final heatmap to", filename)


banner = """
======================================================================
Exploration

Generating the dataset for the {env} environment using algorithm {alg}.
The results will be stored in {dataset_dir}.
======================================================================
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)

    parser.add_argument("--eval_interval", "-ei", type=int, default=10000)
    parser.add_argument("--visualize_interval", "-vi", type=int, default=1000)
    parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=10)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument('--logdir', type=str, default='./data/run_summary')
    parser.add_argument("--use_reward", action="store_true")
    # parser.add_argument("--dataset_dir", type=str, required=True)
    args = parser.parse_args()

    # create directory for logging
    logdir_prefix = "hw5_explore_"  # keep for autograder

    config = make_config(args.config_file)
    logger = make_logger(args.logdir, logdir_prefix, config)
    args.dataset_dir = logger.get_log_dir()
    os.makedirs(args.dataset_dir, exist_ok=True)
    print(banner.format(env=config["env_name"], alg=config["agent"], dataset_dir=args.dataset_dir))

    run_training_loop(config, logger, args)


if __name__ == "__main__":
    main()


