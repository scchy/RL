#!/bin/bash
#!encoding:utf-8
source ~/.bashrc

runType=$1;
echo "==================== [  runType=${runType} ] ========================="

CUR_PATH=$(cd "$(dirname "$0")";pwd);
echo "CUR_PATH=${CUR_PATH}";
echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV}"

if [ $runType == "bc" ]; then
# conda activate sccRL
python train_loop.py \
    --expert_policy_file  ${CUR_PATH}/data/Ant.pkl \
    --env_name Ant-v4 --exp_name bc_ant \
    --n_iter 1 \
    --num_agent_train_steps_per_iter 5000 \
    --expert_data ${CUR_PATH}/data/expert_data_Ant-v4.pkl \
    --video_log_freq -1 \
    --logdir ${CUR_PATH}/data/run_summary
else
python train_loop.py \
    --expert_policy_file ${CUR_PATH}/data/Ant.pkl \
    --env_name Ant-v4 --exp_name dagger_ant \
    --n_iter 10 \
    --do_dagger \
    --expert_data ${CUR_PATH}/data/expert_data_Ant-v4.pkl \
    --video_log_freq -1 \
    --logdir ${CUR_PATH}/data/run_summary
fi;




