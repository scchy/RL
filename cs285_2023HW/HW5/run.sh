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

# 3.2 Random Network Distillation
# python run_explore.py -cfg ./benchmark/exploration/pointmass_easy_rnd.yaml
# python run_explore.py -cfg ./benchmark/exploration/pointmass_medium_rnd.yaml
# python run_explore.py -cfg ./benchmark/exploration/pointmass_hard_rnd.yaml


# 4 Offline RL 
# 4.1 CQL
# python run_offline.py -cfg ./benchmark/offline/pointmass_easy_cql.yaml
# python run_offline.py -cfg ./benchmark/offline/pointmass_easy_dqn.yaml


# python run_offline.py -cfg ./benchmark/offline/pointmass_medium_cql.yaml
# python run_offline.py -cfg ./benchmark/offline/pointmass_medium_dqn.yaml

# python run_offline.py -cfg ./benchmark/offline/pointmass_hard_cql.yaml
# python run_offline.py -cfg ./benchmark/offline/pointmass_hard_dqn.yaml

# 4.2 IQL AWAC
# python run_offline.py -cfg ./benchmark/offline/pointmass_hard_awac.yaml
# python run_offline.py -cfg ./benchmark/offline/pointmass_hard_iql.yaml

# 5 online fine-tunning 
python run_funtinue.py -cfg ./benchmark/finetunning/pointmass_hard_cql_finetune.yaml --use_reward
