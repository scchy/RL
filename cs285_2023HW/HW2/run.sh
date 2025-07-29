#!/bin/bash
#!encoding:utf-8
source ~/.bashrc

runType=$1;
echo "==================== [  runType=${runType} ] ========================="

CUR_PATH=$(cd "$(dirname "$0")";pwd);
echo "CUR_PATH=${CUR_PATH}";
echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV}"


python train_loop.py  --seed 202507 \
    --env_name CartPole-v1 -n 100 -b 512 -lr 2.0e-3 \
    --exp_name cartpole_rtg \
    --discount 0.99 \
    --video_log_freq -1 \
    --logdir ${CUR_PATH}/data/run_summary \
    --use_reward_to_go


python train_loop.py  --seed 202507 \
    --env_name CartPole-v1 -n 100 -b 512 -lr 2.0e-3 \
    --exp_name cartpole_rtg_na \
    --discount 0.99 \
    --video_log_freq -1 \
    --logdir ${CUR_PATH}/data/run_summary \
    --use_reward_to_go \
    --normalize_advantages

# ----------------------------
python train_loop.py  --seed 202507 \
    --env_name CartPole-v1 -n 100 -b 512 -lr 2.0e-3 \
    --exp_name cartpole_rtg_base \
    --discount 0.99 \
    --video_log_freq -1 \
    --logdir ${CUR_PATH}/data/run_summary \
    --use_reward_to_go \
    --use_baseline -bgs 2 \
    --baseline_learning_rate 3.0e-3


python train_loop.py  --seed 202507 \
    --env_name CartPole-v1 -n 100 -b 512 -lr 2.0e-3 \
    --exp_name cartpole_rtg_base_na \
    --discount 0.99 \
    --video_log_freq -1 \
    --logdir ${CUR_PATH}/data/run_summary \
    --use_reward_to_go \
    --normalize_advantages \
    --use_baseline -bgs 2 \
    --baseline_learning_rate 3.0e-3

# ----------------------------
python train_loop.py  --seed 202507 \
    --env_name CartPole-v1 -n 100 -b 512 -lr 2.0e-3 \
    --exp_name cartpole_rtg_gae \
    --discount 0.99 \
    --video_log_freq -1 \
    --logdir ${CUR_PATH}/data/run_summary \
    --use_reward_to_go \
    --use_baseline -bgs 2 \
    --baseline_learning_rate 3.0e-3 \
    --gae_lambda 0.95 


python train_loop.py  --seed 202507 \
    --env_name CartPole-v1 -n 100 -b 512 -lr 2.0e-3 \
    --exp_name cartpole_rtg_gae_na \
    --discount 0.99 \
    --video_log_freq -1 \
    --logdir ${CUR_PATH}/data/run_summary \
    --use_reward_to_go \
    --use_baseline -bgs 2 \
    --baseline_learning_rate 3.0e-3 \
    --normalize_advantages \
    --gae_lambda 0.95 

