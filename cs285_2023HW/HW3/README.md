
# Assignment 3:  Q-Learning and Actor-Critic Algorithms

- [berkeley-PDF-HW03](https://rail.eecs.berkeley.edu/deeprlcourse/deeprlcourse/static/homeworks/hw3.pdf)
- [Github-hw3](https://github.com/berkeleydeeprlcourse/homework_fall2023/tree/main/hw3)


# 1 Multistep Q-Learning

$$y_{j,t}=(\sum_{t^\prime}^{t+N-1} \gamma^{t^\prime -t} r_{j, t^\prime}) + \gamma^N \max_{a_{j, t+N}} Q_{\phi_k}(s_{j,t+N}, a_{j, t+N}) ... (1)$$

$$\phi_{k+1} \leftarrow \argmin_{\phi \in \Phi} \sum (y_{j,t} - Q_{\phi_k}(s_{j,t}, a_{j, t}))^2 .... (2)$$

策略更新：

$$\pi_{k+1}(a_t | s_t) \leftarrow \left\{\begin{matrix} 1; a_t=\argmax_{a_t} Q_{\phi_{k+1}}(s_t, a_t) \\ 0; otherwise  \end{matrix}\right. .... (3)$$



for k = 0 to K-1 do
1. 采样阶段： 执行策略 $\pi_k$，收集 B 条轨迹，加入经验回放池 $D_{k+1}$
2. 目标计算（公式 (1)）对每个样本 $(s_{j,t}, a_{j,t}, r_{j,t}, s_{j,t+1}, ...) \in D_{k+1}$，计算 N-step 回报目标：
   1. $y_{j,t} \leftarrow \sum_{t'=t}^{t+N-1} \gamma^{t'-t} r_{j,t'} + \gamma^N \max_{a'} Q_{\phi_k}(s_{j,t+N}, a')$
3. Q函数更新（公式 (2)）通过最小化平方误差更新 Q 函数参数：
   1. $\phi_{k+1} \leftarrow \argmin_{\phi \in \Phi} \sum_{j,t} (y_{j,t} - Q_\phi(s_{j,t}, a_{j,t}))^2$
4. 策略更新公式 (3)根据更新后的 Q 函数，更新策略为贪婪策略
   1. $\pi_{k+1} \leftarrow (3)$

end for



## 1.1  TD-Learning Bias

Q-Learning : $B^\star\hat{Q}=r + \gamma \max_{a^\prime} \hat{Q}(s^\prime, a^\prime)$

期望和最大值运算不能交换顺序：
$$E[\max_{a^\prime} \hat{Q}(s^\prime, a^\prime)] \ge \max_{a^\prime} E[\hat{Q}(s^\prime, a^\prime)]=\max_{a^\prime} \hat{Q}(s^\prime, a^\prime) $$

因此，Bellman backup $B^\star\hat{Q}$ 是**有偏的**。

SARSA: $y=r + \gamma Q(s^\prime, a^\prime);  a^\prime \sim \pi(\cdot | s^\prime)$

$$E[B^\pi\hat{Q}(s, a)]=r + \gamma E_{a^\prime \sim \pi(\cdot | s^\prime)}[Q(s^\prime, a^\prime)]=B^\pi Q(s, a)$$

所以SARSA， Bellman backup $B^\pi\hat{Q}$ 是无偏估计

## 1.2  Tabular Learning
$Q_{\phi_k}$ 是真实最优$Q^\star$的估计值，在下列情况下：
1. $Q_{\phi_{k+1}}$ 是上一个策略$\pi_k$的Q函数$Q^{\pi_k}$ 的无偏估计
2. 算法迭代极限(反复运行算法足够多次-但每轮只用固定数量的数据): 当$k \rightarrow \infin$且批量B固定时, $Q_{\phi_k}$是$Q^\star$的无偏估计， $\lim_{k\rightarrow \infin} \mathbb{E}[Q_{\phi_k}(s, a) - Q^\star(s, a)] = 0$
3. 算法 + 数据极限: 在无限数据和迭代下，我们恢复最优$Q^\star$ $\lim_{k,B\rightarrow \infin} \mathbb{E}[||Q_{\phi_k}- Q^\star||_\infin] = 0$

**这里的“无偏”不是指 Bellman 目标本身，而是指 最小二乘拟合后的 Q 函数**

补充假设
- 状态和动作均是有限的 (tabular)
- 每个state-action对，在batch中至少出现一次 
- 在表格设置中，$Q_{\phi_k}$ 可以是任意函数(无函数近似误差)  $\{ Q_{\phi_k}: \phi \in \Phi \} = \mathbb{R}^{S \times A}$
  - $Q(s, a) = Q(s, a) + \beta (y - Q(s, a)) $

用buffer $D_k$ 进行bellman backup的时候
- *on-policy*: 只使用当前策略产出的数据
- *off-policy*: 使用不同策略产出的数据



tabular + on-policy + 无函数误差 的设定下
- 我们可以 精确拟合 Bellman 目标（因为 Q 是查表）
- 数据分布 就是当前策略 $\pi_k$ 的真实分布
- 因此，拟合出来的 Q 函数就是 $Q^{\pi_k}$ 的无偏估计


| 情况                | I | II-算法迭代极限 | III=算法+数据极限 |
| ----------------- | - | -- | --- |
| N-step = 1, on-policy  | √ |  √  |  √  |
| N-step = 1, off-policy | X-分布偏差 | X-分布偏差 | √  |
| N-step > 1, on-policy  | X-NStep偏差 | X-NStep偏差   |  √  |
| N-step > 1, off-policy | X-分布偏差+NStep偏差  |  X-分布偏差+NStep偏差  |  √  |
| N-step → ∞, on-policy  | √-等于同策略下路径回报  | √-等于同策略下路径回报 |  √  |
| N-step → ∞, off-policy | X-分布偏差-NStep偏差  |  X-分布偏差-NStep偏差  |  √  |

## 1.3 Variance of Q Estimate

三种情况下，哪一种方差较大。再固定数据大小B, 无限次迭代下
| 情况                            | Q 值估计的方差来源                                        | 方差高低   |
| ----------------------------- | ------------------------------------------------- | ------ |
| N = 1 (单步 bootstrap)   | 只用单步奖励 + bootstrap 值，**bootstrap 误差大**，但**回报路径短** | **最低** |
| N > 1 (多步 bootstrap)   | 用 N 步真实奖励，**减少了 bootstrap 误差**，但**路径变长**，累积奖励方差增大 | **中等** |
| N → ∞ (无 bootstrap，纯蒙特卡洛) | 完全依赖整条轨迹的累积奖励，**无 bootstrap 误差**，但**轨迹方差极大**      | **最高** |


## 1.4 Function Approximation

用函数近似Q。假设任意确定性策略$\pi$(包含最优策略$\pi^\star$) ,函数近似可以代表真实的$Q^\pi$ (函数近似可精确表示),下列哪些描述是正确的？
1. <font color=darkred>X</font> | N=1, $Q_{\phi_{k+1}}$ 是 上一次策略 $Q^{\pi_k}$的Q函数的无偏估计。(非查表Max一直是高估)
2. <font color=darkred>√</font> | N=1, $k,B\rightarrow \infin$(算法+数据极限), $Q_{\phi_k}$ 最终会收敛到 $Q^\star$
3. <font color=darkred>√</font> | N>1, $k,B\rightarrow \infin$(算法+数据极限), $Q_{\phi_k}$ 最终会收敛到 $Q^\star$
4. <font color=darkred>√</font> | N→ ∞, $k,B\rightarrow \infin$(算法+数据极限), $Q_{\phi_k}$ 最终会收敛到 $Q^\star$
   1.  N→∞ 即纯 Monte-Carlo 回报，无 bootstrap 偏差

| 维度             | 1.2 tabular on-policy        | 1.4 函数近似                             |
| -------------- | ---------------------------- | ------------------------------------ |
| **估计对象**       | 期望是 **对数据分布** 的期望            | 期望是 **对 Bellman 目标本身** 的期望           |
| **max 偏差是否存在** | ✅ 存在                         | ✅ 存在                                 |
| **是否能被“平均掉”**  | ✅ 可以，因为 **每个 (s,a) 可被无限次采样** | ❌ 不能，因为 **函数只能拟合期望目标**，不能消除 max 偏差   |
| **结果**         | 查表可精确收敛到 **Q^{\pi_k} 的真值**    | 即使网络无限容量，也只能收敛到 **有偏的 Bellman 目标期望** |


## 1.5 Multistep Importance Sampling

我们可以使用重要性抽样(importance sampling )使N步更新策略与从随机策略中提取的轨迹协同工作。重新公式(2) 

- 轨迹重要性权重:
  - $\rho_{t:t+n-1} = \prod_{t^\prime=t}^{t+n-1} \frac{\pi(a_{t^\prime} | s_{t^\prime})}{\pi^\prime(a_{t^\prime} | s_{t^\prime})}$
- 单步更新(N=1):
  - $\rho_{t}=\frac{\pi(a_{t} | s_{t})}{\pi^\prime(a_{t} | s_{t})}$


$\phi_{k+1} \leftarrow \argmin_{\phi \in \Phi}  \sum_{j, t} \rho_{t:t+n-1}^j (y_{j,t} - Q_{\phi_k}(s_{j,t}, a_{j, t}))^2 .... (2)$
- $y_{j,t}$ 计算不变


当N=1时公式2需要改变么，N→∞ 呢？
当N=1：
- $\phi_{k+1} \leftarrow \argmin_{\phi \in \Phi}  \sum_{j, t}  \frac{\pi(a_{t} | s_{t})}{\pi^\prime(a_{t} | s_{t})}  (y_{j,t} - Q_{\phi_k}(s_{j,t}, a_{j, t}))^2$

当N→∞ 
- $\phi_{k+1} \leftarrow \argmin_{\phi \in \Phi}  \sum_{j, t} \rho_{t:\infin}^j   (y_{j,t} - Q_{\phi_k}(s_{j,t}, a_{j, t}))^2$


实际实现中，通常会使用 加权经验回放 (weighted experience replay) 或 V-trace 等方法来稳定 off-policy 训练。


# 2 Deep Q-Learning


