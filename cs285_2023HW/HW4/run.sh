#!/bin/bash
#!encoding:utf-8
source ~/.bashrc

runEnv=$1;
echo "==================== [  runEnv=${runEnv} ] ========================="

CUR_PATH=$(cd "$(dirname "$0")";pwd);
echo "CUR_PATH=${CUR_PATH}";
echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV}"

# p1 
# python run_mbrl.py -cfg ./benchmark/halfcheech_0_iter.yaml \
#     --seed 1 \
#     --video_log_freq -1 \
#     -ei 5000

# p2 & 3
# python run_mbrl.py -cfg ./benchmark/obstacles_x_iter.yaml \
#     --seed 1 \
#     --video_log_freq -1 \
#     -ei 5000

# python run_mbrl.py -cfg ./benchmark/reacher_x_iter.yaml \
#     --seed 1 \
#     --video_log_freq -1 \
#     -ei 5000

# python run_mbrl.py -cfg ./benchmark/halfcheech_multi_iter.yaml \
#     --seed 1 \
#     --video_log_freq -1 \
#     -ei 5000

# p5 CEM
# python run_mbrl.py -cfg ./benchmark/halfcheetah_cem.yaml \
#     --seed 1 \
#     --video_log_freq -1  

# python run_mbrl.py -cfg ./benchmark/p4/reacher_ablation.yaml --seed 1 --video_log_freq -1 &
# python run_mbrl.py -cfg ./benchmark/p4/reacher_es2.yaml --seed 1 --video_log_freq -1 & 
# python run_mbrl.py -cfg ./benchmark/p4/reacher_es4.yaml --seed 1 --video_log_freq -1 &
# python run_mbrl.py -cfg ./benchmark/p4/reacher_h14.yaml --seed 1 --video_log_freq -1 

# python run_mbrl.py -cfg ./benchmark/p4/reacher_h6.yaml --seed 1 --video_log_freq -1 &
# python run_mbrl.py -cfg ./benchmark/p4/reacher_nas1500.yaml --seed 1 --video_log_freq -1 & 
# python run_mbrl.py -cfg ./benchmark/p4/reacher_nas500.yaml --seed 1 --video_log_freq -1 

# P6 
python run_mbrl.py -cfg ./benchmark/halfcheetah_mbpo.yaml \
    --seed 1 --video_log_freq -1 \
    --sac_config_file ./benchmark/halfcheetah_clipq.yaml
