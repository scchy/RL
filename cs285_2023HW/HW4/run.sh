#!/bin/bash
#!encoding:utf-8
source ~/.bashrc

runEnv=$1;
echo "==================== [  runEnv=${runEnv} ] ========================="

CUR_PATH=$(cd "$(dirname "$0")";pwd);
echo "CUR_PATH=${CUR_PATH}";
echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV}"

python run_mbrl.py -cfg ./benchmark/halfcheech_0_iter.yaml \
    --seed 1 \
    --video_log_freq -1 \
    -ei 5000
