#!/bin/bash
#!encoding:utf-8
source ~/.bashrc

runEnv=$1;
echo "==================== [  runEnv=${runEnv} ] ========================="

CUR_PATH=$(cd "$(dirname "$0")";pwd);
echo "CUR_PATH=${CUR_PATH}";
echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV}"

if [ $runEnv == "CartPole" ]; then
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

elif [ $runEnv == "HalfCheetah" ]; then
# 4 Using a Neural Network Baseline 
# HalfCheetah-v4  
python train_loop.py  --seed 202507 \
    --env_name HalfCheetah-v4 -n 100 -b 1024 -lr 1.0e-2 \
    --exp_name HalfCheetah_rtg_base_na \
    --discount 0.95 \
    --video_log_freq -1 \
    --logdir ${CUR_PATH}/data/run_summary \
    --use_reward_to_go \
    --use_baseline -bgs 5 \
    --baseline_learning_rate 1.0e-2 \
    --normalize_advantages  

elif [ $runEnv == "LunarLander" ]; then
# 5 Implementing Generalized Advantage Estimation
lambdaList=(0.0 0.95 0.98 0.99 1.0);
# LunarLander
for lambda in "${lambdaList[@]}"; do
echo "==================== [  runEnv=${runEnv}[gae_lambda=${lambda}] ] ========================="
python train_loop.py  --seed 202507 \
    --env_name LunarLander-v2 -n 300 -b 1024 -lr 1.0e-3 \
    --exp_name LunarLander_rtg_gae_na_${lambda} \
    --ep_len 1000 \
    --discount 0.99 \
    --n_layers 3 --layer_size 128 \
    --video_log_freq -1 \
    --logdir ${CUR_PATH}/data/run_summary \
    --use_reward_to_go \
    --use_baseline -bgs 5 \
    --baseline_learning_rate 1.0e-3 \
    --normalize_advantages  \
    --gae_lambda ${lambda}

done;
elif [ $runEnv == "InvertedPendulum" ]; then
# InvertedPendulum-v4
# 6 Hyperparameters and Sample Efficiency
python train_loop.py  --seed 202507 \
    --env_name InvertedPendulum-v4 -n 100 -b 1024 \
    --exp_name InvertedPendulum_rtg_base_na \
    --video_log_freq -1 \
    -rtg --use_baseline -na 

elif [ $runEnv == "Humanoid" ]; then
# 7 Extra Credit: Humanoid
# python train_loop.py  --seed 202507 \
#     --env_name Humanoid-v4 -n 1000 -b 4096 -lr 1.0e-3 \
#     --ep_len 1000 \
#     --discount 0.99 \
#     --n_layers 3 --layer_size 256 \
#     --video_log_freq -1 \
#     --use_baseline -na -rtg -bgs 25 \
#     --gae_lambda 0.97 \
#     --exp_name humanoid_rtg_gae_na_bgs25 
# 4.5e-4  1.5e-3 20480
python train_loop.py  --seed 202507 \
    --env_name Humanoid-v4 -n 1800 -b 10240 -lr 1.5e-3 \
    --norm_obs \
    --max_grad_norm 1.0 \
    --ep_len 680 \
    --eval_ep_len 1000 \
    --discount 0.99 \
    --n_layers 3 --layer_size 128 \
    --video_log_freq -1 \
    --use_baseline --normalize_advantages -rtg -bgs 5 \
    --baseline_learning_rate 3.5e-3 \
    --gae_lambda 0.97 \
    --exp_name humanoid_rtg_gae_na_bgs5_nobs

else
    echo "Please input runEnv: CartPole, HalfCheetah, LunarLander, InvertedPendulum, Humanoid"
fi
