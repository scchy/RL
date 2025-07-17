# python3 
# Create Date: 2025-07-16
# Func:  Runs behavior cloning and DAgger for homework 1
# ===============================================================

import pickle
import os
import time
import gymnasium as gym
import numpy as np
from tqdm.auto import tqdm 
import torch
from loguru import logger as lg
from policies.utils import ( 
    to_numpy, from_numpy,  init_gpu,
    compute_metrics, 
    sample_trajectories, 
    sample_n_trajectories,
    ReplayBuffer,
    all_seed,
    Logger
)
from policies.MLP_policy import MLPPolicySL
from policies.loaded_gaussian_policy import LoadedGaussianPolicy


# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40  # we overwrite this in the code below
MJ_ENV_NAMES = ["Ant-v4", "Walker2d-v4", "HalfCheetah-v4", "Hopper-v4"]


@lg.catch()
def run_training_loop(params):
    """
    Runs training with the specified parameters
    (behavior cloning or dagger)

    Args:
        params: experiment parameters
    """
    # init
    # --------------------
    # Get params, create logger, create TF session
    logger = Logger(params['logdir'])
    seed = params['seed']
    all_seed()
    init_gpu(
        use_gpu=not params['no_gpu'],
        gpu_id=params['which_gpu']
    )
    # Set logger attributes
    log_video = True
    log_metrics = True

    # ENV
    # --------------------
    env = gym.make(params['env_name'], render_mode=None)
    env.reset(seed=seed)
    # Maximum length for episodes
    params['ep_len'] = params['ep_len'] or env.spec.max_episode_steps
    MAX_VIDEO_LEN = params['ep_len']

    assert isinstance(env.action_space, gym.spaces.Box), "Environment must be continuous"
    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]
    # simulation timestep, will be used for video saving
    if 'model' in dir(env):
        fps = 1/env.model.opt.timestep
    else:
        fps = env.env.metadata['render_fps']

    # AGENT
    # --------------------
    # TODO: Implement missing functions in this class.
    device = params['device']
    actor = MLPPolicySL(
        ac_dim,
        ob_dim,
        params['n_layers'],
        params['size'],
        learning_rate=params['learning_rate'],
        device=device
    )
    
    # replay buffer
    replay_buffer = ReplayBuffer(params['max_replay_buffer_size'])

    # LOAD EXPERT POLICY
    # --------------------
    print('Loading expert policy from...' + params['expert_policy_file'])
    expert_policy = LoadedGaussianPolicy(params['expert_policy_file'])
    expert_policy.to(device)
    print('Done restoring expert policy...')
    
    # TRAINING LOOP
    # --------------------
    total_envsteps = 0
    start_time = time.time()
    n_iters = params['n_iter']
    for itr in range(n_iters):
        print(f"\n\n********** Iteration {itr} ************")
        # decide if videos should be rendered/logged at this iteration
        log_video = False # ((itr % params['video_log_freq'] == 0) and (params['video_log_freq'] != -1))
        # decide if metrics should be logged
        log_metrics = (itr % params['scalar_log_freq'] == 0)
        print("\nCollecting data to be used for training...")
        if itr == 0:
            # BC training from expert data. 
            paths = pickle.load(open(params['expert_data'], 'rb'))
            envsteps_this_batch = 0
        else:
            # DAGGER training from sampled data relabeled by expert
            assert params['do_dagger']
            # TODO: collect `params['batch_size']` transitions
            # HINT: use utils.sample_trajectories
            # TODO: implement missing parts of utils.sample_trajectory
            paths, envsteps_this_batch = sample_trajectories(
                env, 
                expert_policy, 
                # todo: check 
                params['batch_size'], # min_timesteps_per_batch:  n*episode >= min_timesteps_per_batch
                params['ep_len'],     # max_path_length: episode length 
                render=False
            )
        
        total_envsteps += envsteps_this_batch
        # BC: only user expert Data
        # DAgger: expert Data +  expert_policy 生成的数据
        replay_buffer.add_rollouts(paths)
        # train agent (using sampled data from replay buffer)
        print('\nTraining agent using sampled data from replay buffer...')
        training_logs = []
        tq_bar = tqdm(range(params['num_agent_train_steps_per_iter']), total=params['num_agent_train_steps_per_iter'])
        tq_bar.set_description(f'[ {itr+1}/{n_iters}|(paths={len(paths)}) ]')
        for _ in tq_bar:
            # TODO: sample some data from replay_buffer
            # HINT1: how much data = params['train_batch_size']
            # HINT2: use np.random.permutation to sample random indices
            # HINT3: return corresponding data points from each array (i.e., not different indices from each array)
            # for imitation learning, we only need observations and actions.  
            ob_batch, ac_batch = replay_buffer.sample(params['train_batch_size'])

            # use the sampled data to train an agent
            train_log = actor.update(ob_batch, ac_batch)
            training_logs.append(train_log)
            tq_bar.set_postfix({k: np.round(v, 3) for k, v in train_log.items()})

        # log/save
        print('\nBeginning logging procedure...')
        if log_video:
            # save eval rollouts as videos in tensorboard event file
            print('\nCollecting video rollouts eval')
            eval_video_paths = sample_n_trajectories(env, actor, MAX_NVIDEO, MAX_VIDEO_LEN, True)
            # save videos
            if eval_video_paths is not None:
                logger.log_paths_as_videos(
                    eval_video_paths, itr,
                    fps=fps,
                    max_videos_to_save=MAX_NVIDEO,
                    video_title='eval_rollouts'
                )

        if log_metrics:
            # save eval metrics
            print("\nCollecting data for eval...")
            eval_paths, eval_envsteps_this_batch = sample_trajectories(
                env, actor, params['eval_batch_size'], params['ep_len'])

            logs = compute_metrics(paths, eval_paths)
            # compute additional metrics
            logs.update(training_logs[-1]) # Only use the last log for now
            logs["Train_EnvstepsSoFar"] = total_envsteps
            logs["TimeSinceStart"] = time.time() - start_time
            if itr == 0:
                logs["Initial_DataCollection_AverageReturn"] = logs["Train_AverageReturn"]

            # perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')
            logger.flush()

        if params['save_params']:
            print('\nSaving agent params')
            actor.save('{}/policy_itr_{}.pt'.format(params['logdir'], itr))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_policy_file', '-epf', type=str, required=True)  # relative to where you're running this script from
    parser.add_argument('--expert_data', '-ed', type=str, required=True) #relative to where you're running this script from
    parser.add_argument('--env_name', '-env', type=str, help=f'choices: {", ".join(MJ_ENV_NAMES)}', required=True)
    parser.add_argument('--exp_name', '-exp', type=str, default='pick an experiment name', required=True)
    parser.add_argument('--do_dagger', action='store_true')
    parser.add_argument('--ep_len', type=int)

    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1000)  # number of gradient steps for training policy (per iter in n_iter)
    parser.add_argument('--n_iter', '-n', type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=1000)  # training data collected (in the env) during each iteration
    parser.add_argument('--eval_batch_size', type=int,
                        default=1000)  # eval data collected (in the env) for logging metrics
    parser.add_argument('--train_batch_size', type=int,
                        default=100)  # number of sampled data points to be used per gradient/train step

    parser.add_argument('--n_layers', type=int, default=2)  # depth, of policy to be learned
    parser.add_argument('--size', type=int, default=64)  # width of each layer, of policy to be learned
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)  # LR for supervised learning

    parser.add_argument('--video_log_freq', type=int, default=5)
    parser.add_argument('--scalar_log_freq', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', type=int, default=0)
    parser.add_argument('--max_replay_buffer_size', type=int, default=1000000)
    parser.add_argument('--save_params', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./data/run_summary')
    args = parser.parse_args()

    # convert args to dictionary
    params = vars(args)
    if args.do_dagger:
        # Use this prefix when submitting. The auto-grader uses this prefix.
        logdir_prefix = 'q2_'
        assert args.n_iter > 1, ('DAGGER needs more than 1 iteration (n_iter>1) of training, to iteratively query the expert and train (after 1st warmstarting from behavior cloning).')
    else:
        # Use this prefix when submitting. The auto-grader uses this prefix.
        logdir_prefix = 'q1_'
        assert args.n_iter==1, ('Vanilla behavior cloning collects expert data just once (n_iter=1)')

    run_training_loop(params)


if __name__ == "__main__":
    """ 
    python cs285/scripts/run_hw1.py \
        --expert_policy_file cs285/policies/experts/Ant.pkl \
        --env_name Ant-v4--exp_name bc_ant--n_iter 1 \
        --expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
        --video_log_freq-1
    """
    main()

