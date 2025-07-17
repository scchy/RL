
# Assignment 1: Imitation Learning

- [berkeley-PDF-HW01](https://rail.eecs.berkeley.edu/deeprlcourse/deeprlcourse/static/homeworks/hw1.pdf)
- [Github-hw1](https://github.com/berkeleydeeprlcourse/homework_fall2023/blob/main/hw1/README.md)

目标->熟悉模仿学习 (基于expert policy 数据)
- 行为克隆(`behavioral cloning`)
- DAgger算法(`DAgger algorithm`)


# Analysis

1. problem of imitation learning
   1. export policy $\pi^\star$, imitation policy $\pi_\theta$
   2. 训练集内同状态下action差异最小：$E_{p_{\pi^\star}(s)}[\pi_\theta(a \neq \pi^\star(s) | s)] = \frac{1}{T}\sum_{t=1}^TE_{p_{\pi^\star}(s_t)}[\pi_\theta(a_t \neq \pi^\star(s_t) | s_t)] \le \epsilon$
      1. $\sum_{s_t}| p_{\pi_\theta}(s_t) - p_{\pi^\star}(s_t) | \le 2T\epsilon$
         - 强假设 $\pi_\theta(s_{t+1} \neq  \pi^\star(s_t) | s_t) \le \epsilon$
   3. $|r(s_t)| \le R_{max}$
      1. $J(\pi) = \sum_{t=1}^T E_{p_{\pi}(s_t) }[r(s_t)]$


# BC VS DAgger

> BC 是“一次性模仿”，DAgger 是“边学边问”; 
> BC 快速但脆弱，DAgger 更鲁棒但代价高。
> 
> 如需在资源受限或对专家依赖低的场景中使用，可考虑 BC；
> 若对策略鲁棒性要求高、专家可互动，则推荐使用 DAgger 或其改进版本（如 DAgger by coaching、DMD 等）

| 维度       | BC（行为克隆）                           | DAgger（数据集聚合）                            |
| -------- | ---------------------------------- | ---------------------------------------- |
| **学习方式** | 监督学习：将状态-动作对作为训练数据，直接拟合策略 π(s) → a | 在线迭代模仿学习：通过**不断交互**获取新数据，修正策略                |
| **数据需求** | 仅需一次性收集的专家轨迹(expert_data)                       | 初始专家数据(expert_data) + 每次迭代中当前策略生成的轨迹（需专家标注） (expert_policy)          |
| **训练方式** | 离线训练一次即可                           | 多轮迭代：每轮训练策略 → 运行策略 → 请专家标注新数据 → 加入数据集再训练 |
| **优点**   | 简单、易于实现、训练快                       | 缓解复合误差（compounding error），提升泛化能力 |
| **缺点**   | 容易在未见状态犯错，误差累积严重（covariate shift） | 需要**频繁查询专家**，成本高，难以扩展            |
| **适用场景** | 专家数据充足、状态分布变化小、任务简单               | 状态空间大、任务复杂、需要长期稳定策略              |
