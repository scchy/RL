#!/bin/bash
#!encoding:utf-8
source ~/.bashrc

runEnv=$1;
echo "==================== [  runEnv=${runEnv} ] ========================="

CUR_PATH=$(cd "$(dirname "$0")";pwd);
echo "CUR_PATH=${CUR_PATH}";
echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV}"

# random policy
# python run_explore.py -cfg ./benchmark/exploration/pointmass_easy_random.yaml
# python run_explore.py -cfg ./benchmark/exploration/pointmass_medium_random.yaml
# python run_explore.py -cfg ./benchmark/exploration/pointmass_hard_random.yaml

# Random Network Distillation
# python run_explore.py -cfg ./benchmark/exploration/pointmass_easy_rnd.yaml
python run_explore.py -cfg ./benchmark/exploration/pointmass_medium_rnd.yaml
python run_explore.py -cfg ./benchmark/exploration/pointmass_hard_rnd.yaml


