# TODO

- [x] 增加训练log wandb
- [x] baseSkipGrame
- [x] RLHF VS R1
- [x] A3C 
  - [x] hogwild
  - [x] 实现A2C 
    - [OpenAI Gym:ucsd_ece_276](https://chihhuiho.github.io/project/ucsd_ece_276/report.pdf)
    - [ 多阶段协作决策框架 A2C: A Modular Multi-stage Collaborative Decision Framework for Human-AI Teams](https://arxiv.org/abs/2401.14432)
    - [stable_baselines3 a2c](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html)
  - [x] 实现A3C
- [x] Soft Q 2017 
  - [Reinforcement Learning with Deep Energy-Based Policies](https://arxiv.org/abs/1702.08165)
  - [Soft Q-Learning论文阅读笔记](https://zhuanlan.zhihu.com/p/76681229)
  - [Equivalence Between Policy Gradients and Soft Q-Learning](https://ar5iv.labs.arxiv.org/html/1704.06440)
- [x] CQL [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/pdf/2006.04779)
  - [x] GitHub [CQL d4rl/rlkit/torch/sac/cql.py](https://github.com/aviralkumar2907/CQL/blob/master/d4rl/rlkit/torch/sac/cql.py)
  - [x]  [离线强化学习研究综述](https://lib.zjsru.edu.cn/25-2.24-5.pdf)
  - [ ]  混合策略下学习CQL学习反而不如random, 解决方案
    - 单一策略：在单一策略（如随机策略或专家策略）生成的数据集中，CQL 的保守性正则化能够更好地发挥作用。
    - 混合策略：在混合策略生成的数据集中，CQL 的保守性正则化可能无法有效区分不同策略的质量，导致性能下降
    - [x]  目前简单解决`min_q_weight=12.75, reward_scale=1.25`
    - [X]  [WHEN SHOULD WE PREFER DECISION TRANSFORMERS FOR OFFLINE REINFORCEMENT LEARNING](https://arxiv.org/pdf/2305.14550)
- [X] Soft Actor-Critic 2018    [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)
  - HW3
- [X] DT: Decision Transformer: Reinforcement Learning via Sequence Modeling
  - [X] [Github: kzl/decision-transformer](https://github.com/kzl/decision-transformer/blob/master/gym/decision_transformer/models/decision_transformer.py)
- [ ] [Settling the Reward Hypothesis](https://arxiv.org/pdf/2212.10420)
- [ ] TT: Trajectory Transformer Offline Reinforcement Learning as One Big Sequence Modeling Problem
  - [ ] [Github:trajectory-transformer](https://github.com/JannerM/trajectory-transformer)


