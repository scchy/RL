

## WHEN SHOULD WE PREFER DECISION TRANSFORMERS FOR OFFLINE REINFORCEMENT LEARNING

> 帮助你判断在什么条件下应该 优先选择 DT、在什么条件下应该 弃用或替换 DT
> 在稀疏奖励、低质量或长 horizon、人类演示条件下，优先考虑 Decision Transformer； 
> 在数据量极少、高随机或密集奖励场景下，CQL 或 Filtered-BC 往往更划算

paper Link: [ICLR 2024: WHEN SHOULD WE PREFER DECISION TRANSFORMERS FOR OFFLINE REINFORCEMENT LEARNING](https://arxiv.org/pdf/2305.14550)
- [2025: Should We Ever Prefer Decision Transformer?](https://arxiv.org/abs/2507.10174)



## DT 的设计假设与优势

| 维度       | DT 的做法                                                                     | 带来的优势                             |
| -------- | -------------------------------------------------------------------------- | --------------------------------- |
| **建模框架** | 把 RL 问题变成**序列建模**（state → action → reward → …），用 GPT 式 Transformer 自回归预测动作 | 不需要 Bellman 迭代，天然擅长**长程依赖**       |
| **学习目标** | **Return-Conditioned BC**：给定目标回报 R-to-go 与历史，直接预测动作                        | 对**稀疏奖励**不敏感，只要 R-to-go 能区分好坏轨迹即可 |
| **数据结构** | 支持任意长度轨迹，**无需 MDP 假设**                                                     | 对**非马尔可夫数据**（如人类演示）鲁棒             |


## 实证结论：

### 什么时候 DT 值得优先用

| 场景特征                      | 结论                     | 原因简述                                      |
| ------------------------- | ---------------------- | ----------------------------------------- |
| **奖励稀疏**                  | ✅ **DT 显著优于 CQL & BC** | CQL 的 TD 传播在稀疏奖励下难以学习；DT 只需用 R-to-go 过滤轨迹 |
| **数据质量低**（次优/噪声大）         | ✅ **DT 更鲁棒**           | 保守 Q-Learning 会过度悲观；DT 通过序列建模降低外推误差       |
| **任务 horizon 长**（>1000 步） | ✅ **DT & BC 优于 CQL**   | CQL 的 Bellman 迭代在长序列中误差累积大                |
| **数据来源为人类演示**             | ✅ **DT 与 BC 占优**       | 人演示往往非马尔可夫、轨迹间分布差异大，DT 无需对齐 MDP           |
| **数据量充足**                 | ✅ **DT 可随数据扩展**        | 5× 数据 → Atari 上 2.5× 性能提升，呈现正比例放大         |


### 什么时候 DT 并不合适

| 场景特征              | 结论                         | 原因 & 替代方案                                                                |
| ----------------- | -------------------------- | ------------------------------------------------------------------------ |
| **数据量极少**         | ❌ **CQL 更好**               | DT 需要更大样本才能匹配 CQL 的性能                                                    |
| **环境高度随机**（高噪声转移） | ❌ **CQL 显著更好**             | 随机性破坏序列一致性，DT 难以建模                                                       |
| **稀疏奖励且数据可过滤**    | ❌ **Filtered-BC（FBC）可能更好** | Dong et al. (2025) 指出：在相同稀疏奖励机器人任务上，**简单过滤轨迹后再 BC** 比 DT 表现更好、训练更快、参数量更小 |
| **密集奖励 + 低随机性**   | ❌ **CQL 通常更优**             | 既有文献与 D4RL 原始任务均显示 CQL 领先                                                |


## 决策流程图
```mermaid
flowchart TD
    %% 起点
    Start([开始选择离线RL算法])

    %% 判断节点
    RewardSparse{数据稀疏奖励？}
    DataEnough{数据量够大？}
    HighRandom{数据高随机？}
    LongHorizon{任务horizon长？}

    %% 算法节点
    UseDT[[使用 DT]]
    UseFBC[[使用 Filtered-BC]]
    UseCQL[[使用 CQL]]
    UseBC[[使用 BC/DT]]

    %% 连线
    Start --> RewardSparse
    RewardSparse -- 是 --> DataEnough
    RewardSparse -- 否 --> HighRandom

    DataEnough -- 是 --> UseDT
    DataEnough -- 否 --> UseFBC

    HighRandom -- 是 --> UseCQL
    HighRandom -- 否 --> LongHorizon

    LongHorizon -- 是 --> UseBC
    LongHorizon -- 否 --> UseCQL
```

## 实践 Checklist（工程落地）

| 步骤         | 建议                                                                                         |
| ---------- | ------------------------------------------------------------------------------------------ |
| **Step 1** | 检查奖励密度：若 reward<0.01 占比 >90%，优先考虑 DT                                                       |
| **Step 2** | 估计数据量：轨迹数 <1 k 时先尝试 FBC，≥2 k 再考虑 DT                                                        |
| **Step 3** | 评估随机性：环境转移噪声 σ>0.1 时，放弃 DT 换 CQL                                                           |
| **Step 4** | 若 horizon>1 k 步，DT 训练时用 **context length ≥ 3×horizon** 并加 **gradient checkpointing** 防 OOM |
| **Step 5** | 训练 DT 时把 **return-to-go 离散化**（如 0/1 稀疏标签）可进一步加速收敛                                          |
