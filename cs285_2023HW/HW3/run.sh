#!/bin/bash
#!encoding:utf-8
source ~/.bashrc

runEnv=$1;
echo "==================== [  runEnv=${runEnv} ] ========================="

CUR_PATH=$(cd "$(dirname "$0")";pwd);
echo "CUR_PATH=${CUR_PATH}";
echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV}"

# tensorboard --logdir ./data/run_summary/
# python run_dqn.py -cfg ./benchmark/dqn_carpole.yaml \
#     --seed 1 \
#     --video_log_freq -1 \
#     -ei 1000


python run_dqn.py -cfg ./benchmark/dqn_lunarlander.yaml \
    --seed 1 \
    --video_log_freq -1 \
    -ei 2000  # 10000

