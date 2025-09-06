## Policy gradient methods for reinforcement learning with function approximation

paper Link: [Policy gradient methods for reinforcement learning with function approximation](https://web.eecs.umich.edu/~baveja/Papers/PolicyGradientNIPS99.pdf)


> 首次把策略梯度与任意可微函数逼近器结合，给出了带基线的 REINFORCE / Actor-Critic的通用理论框架，证明了梯度无偏性，奠定了深度策略学习的基础。
> 这篇 1999 的论文把「策略梯度 = 梯度定理 + 任意函数逼近 + 基线减方差」写成了公式，为 20 年后的深度策略学习铺好了路。


| 点         | 内容                                               |     
| --------- | ------------------------------------------------ |  
| **问题设定**  | 连续或高维状态/动作空间，无法查表 → 必须用函数逼近（线性/神经网络）。            |    
| **策略参数化** | $\pi_\theta(a\|s)$，$\theta$ 为任意可微参数（线性权重或 NN）。 |
| **梯度定理**  | $\frac{\partial J}{\partial \theta} = E_\pi [ \Delta_\theta log \pi_\theta Q_\pi ]$（无偏）。                |             
| **基线减方差** | 用状态值函数 $V_\theta(s)$ 或优势 $A = Q − V$，证明仍无偏。                |        
| **算法实例**  | 1. REINFORCE + 基线 2. Actor-Critic（策略网络 + 值网络） |    



1. 实验亮点:
用 线性函数逼近 在 Acrobot、Mountain Car 上验证：
Actor-Critic 收敛速度显著优于 REINFORCE。
首次展示 策略梯度 + 函数逼近 在控制任务上的可行性。


2. 影响
理论：现代所有 Deep RL 策略算法（A3C, PPO, SAC, DPG …）皆可视为该框架的 深度网络实例。
实践：为后续 Actor-Critic、Advantage Actor-Critic (A2C/A3C) 提供了梯度公式与方差缩减范式。


### 简介

我们探讨了强化学习中函数逼近的一种替代方法。我们并不先近似一个值函数，再据此计算确定性策略；而是直接近似一个随机策略，使用一套独立的、拥有自身参数的函数逼近器来完成 $a = \pi_\theta(s)$

梯度的无偏估计：$\Delta \theta \approx \alpha \frac{\partial{\rho}}{\partial{\theta}}$
- $\rho$ : 对应策略的性能（例如，每步的平均奖励）

Williams（1988, 1992）提出的 REINFORCE 算法同样能够获得梯度的一个无偏估计，但并未借助任何已学习的值函数。学习一个值函数，并利用它来降低梯度估计的方差，对于实现快速学习似乎是必不可少的。Jaakkola、Singh 和 Jordan（1995）针对与表格化 POMDP 相对应的函数逼近这一特殊情况，证明了一个与我们非常相似的结果。我们的结果不仅强化了他们的结论，而且将其推广到任意可微的函数逼近器。

我们的结果还为证明基于“actor-critic”或策略迭代架构的多种算法的收敛性提供了一种途径（例如 Barto, Sutton 和 Anderson, 1983；Sutton, 1984；Kimura 和 Kobayashi, 1998）。在本文中，我们首次朝这一方向迈出第一步：首次证明一个带有一般可微函数逼近的策略迭代版本能够收敛到局部最优策略。

Baird 和 Moore（1999）为其 VAPS 方法族获得了一个较弱但表面上类似的结果。与策略梯度方法一样，VAPS 也包含分别参数化的策略和价值函数，并通过梯度方法进行更新。然而，VAPS 方法并非沿着性能（期望长期回报）的梯度上升，而是沿着一种结合了`performance`和`value-function accuracy`的度量上升。因此，VAPS 不会收敛到局部最优策略，除非在完全不考虑价值函数准确性的情况下，此时 VAPS 退化为 REINFORCE。类似地，Gordon（1995）的拟合值迭代也是收敛的且基于价值，但同样无法找到局部最优策略。

因此，VAPS 并不会收敛到局部最优策略，除非完全不考虑价值函数的准确性——此时 VAPS 退化为 REINFORCE。同样地，Gordon（1995）提出的拟合值迭代虽然收敛且基于价值，但也无法找到局部最优策略。

### 1 Policy Gradient Theorem


$t \in \{0, 1, 2, ...\}, s_t \in \mathcal{S}, a_t\in \mathcal{A}, r_t \in R$

状态转移概率： $P^a_{ss^\prime} = P_r(s_{t+1}=s^\prime|s_t=s, a_t=a)$  
奖励：$R^a_s = E(r_{t+1}|s_t=s, a_t=a), \forall s, s^\prime \in \mathcal{S}, a\in \mathcal{A}$  
策略：$\pi_\theta(s, a) = P_r(a_t=a|s_t=s, \theta), \forall s\in\mathcal{S}, a\in \mathcal{A}$  
目标定义：
1. 目标定义：1-平均回报形式(`verage reward formulation,`)。其中策略按照其每步长期期望回报 $\rho(\pi)$ 来排序
   - $\rho(\pi)=\lim_{n\rightarrow \infin} \frac{1}{n} E[\sum_{t=1}^\infin r_t | \pi]=\sum_s d^\pi(s) \sum_a \pi(s, a) R_s^a$
     - 是策略 $\pi$下的状态平稳分布，我们假设它存在，并且对所有策略而言都与初始状态 s₀ 无关 $d^\pi(s) = \lim_{t\rightarrow \infin} P_r(s_t=s|s_0, \pi)$
   - 平均回报函数，给定策略时状态–动作对的价值定义为
     - $Q^\pi(s,a) = \sum_{t=1}^\infin E[r_t - \rho(\pi)|s_0=s, a_0=a,\pi], \forall s\in \mathcal{S}, a\in \mathcal{A}$
2. 目标定义：2-长期回报（`long-term reward`）。存在一个指定的起始状态 s₀，而我们只关心从该状态出发所获得的长期回报。我们将仅给出一次结果，但在下述定义下，这些结果同样适用于这一形式。
   1. $\rho(\pi)=\lim_{n\rightarrow \infin} E[\sum_{t=1}^\infin \gamma^{t-1}r_t |s_0 \pi]$
   2. $Q^\pi(s,a) = E[\sum_{k=1}^\infin  \gamma^{k-1}t_{t+k} | s_t=s, a_t=a, \pi]$
   3. $\gamma \in [0, 1]$
   4. $d^\pi(s) = \sum_{t=0}^\infin \gamma^t P_r(s_t=s|s_0, \pi)$


**Theorem 1 (policy Gradient)**

对于任意 MDP，无论是采用平均回报形式还是起始状态形式，策略梯度最终如下： 

$$\frac{\partial \rho}{\partial \theta} = \sum_s d^\pi(s) \sum_a \frac{\partial \pi(s, a)}{\partial \theta} Q^\pi(s, a) ....(2)$$

补充推导：  
1. 假设状态转移概率不受策略参数 $\theta$ 的梯度影响
2. $s \sim \pi_\theta$: s 从 $\pi_\theta$策略的轨迹中采样出来, 那么$\frac{\partial \pi(s, a)}{\partial \theta} Q^\pi(s, a)$ 就是 $\frac{\partial{\rho}}{\partial{\theta}}$的无偏估计
3. 价值预估函数：$Q^\pi(s, a)$ 
   1. 可以用实际动作奖励近似： $Q^\pi(s, a) = R_t = \sum_{k=1}^\infin r_{t+k}-\rho(\pi)$ (起始状态时去掉$-\rho(\pi)$)
      1. $\rho(\pi)$ 是策略 $\pi$ 的平均单步回报, 把 原始回报 转换成 优势量


REINFORCE algorithm: $\Delta \theta _t \propto \frac{\partial \pi(s, a)}{\partial \theta} R_t / \pi(s_t, a_t)$ 
- $1/ \pi(s_t, a_t)$: 纠正策略 π 对其偏好动作的过度采样



### 2. Policy Gradient with Approximation
对Singh 和 Jordan（1995）针对与表格化 POMDP 相对应的函数逼近进行拓展 

$Q^\pi$: $f_w: \mathcal{s} \times \mathcal{A} \rightarrow R$

参数$w$迭代
$\Delta w_t \propto \frac{\partial}{\partial w}[\hat{Q}^\pi (s_t, a_t) - f_w(s_t, a_t)]^2 \propto [\hat{Q}^\pi (s_t, a_t) - f_w(s_t, a_t)]\frac{\partial f_w(s_t, a_t)}{\partial w}$ 
- $\hat{Q}^\pi (s_t, a_t)$ 是$Q^\pi (s_t, a_t)$的无偏估计，比如$R_t$


当这一过程收敛到局部最优时，便满足：
$$\sum_s d^\pi(s) \sum_a \pi(s, a)[Q^\pi (s_t, a_t) - f_w(s_t, a_t)]\frac{\partial f_w(s_t, a_t)}{\partial w} = 0 ... (3)$$


Theorem 2 (Policy Gradient with Function Approximation).

当$f_w$满足(3)时候,与策略参数化兼容：

$\frac{\partial f_w(s_t, a_t)}{\partial w} = \frac{\partial \pi(s, a)}{\partial \theta} / \pi(s, a)$

$\rightarrow \sum_s d^\pi(s) \sum_a \frac{\partial \pi(s, a)}{\partial \theta}  [Q^\pi (s_t, a_t) - f_w(s_t, a_t)] = 0$


$$\frac{\partial \rho}{\partial \theta} = \sum_s d^\pi(s) \sum_a \frac{\partial \pi(s, a)}{\partial \theta} Q^\pi(s, a) - 0$$
$$= \sum_s d^\pi(s) \sum_a \frac{\partial \pi(s, a)}{\partial \theta} Q^\pi(s, a) - \sum_s d^\pi(s) \sum_a \frac{\partial \pi(s, a)}{\partial \theta}  [Q^\pi (s_t, a_t) - f_w(s_t, a_t)] $$
$$= \sum_s d^\pi(s) \sum_a \frac{\partial \pi(s, a)}{\partial \theta}  [Q^\pi (s_t, a_t) - Q^\pi (s_t, a_t) + f_w(s_t, a_t)] $$
$$= \sum_s d^\pi(s) \sum_a \frac{\partial \pi(s, a)}{\partial \theta}   f_w(s_t, a_t) ....(5)$$


### 3 Application to Deriving Algorithms and Advantages


举例： $\pi(s, a) = \frac{e^{\theta^T \phi_{sa}} }{\sum_b e^{\theta^T \phi_{sb}}}, \forall s \in \mathcal{S},  a \in \mathcal{A}$

其中每个 $\phi_{sa}$ 是一个 l 维特征向量，用于刻画状态-动作对 (s, a)。
要使兼容性条件 (4) 得以满足，需要
$\frac{\partial f_w(s_t, a_t)}{\partial w} = \frac{\partial \pi(s, a)}{\partial \theta} / \pi(s, a)= \phi_{sa} - \sum_b \pi(s, b)\phi _{sb}$

$f_w(s_t, a_t)=w^T[ \phi_{sa} - \sum_b \pi(s, b)\phi _{sb}]$

细心的读者会注意到，上述给出的 $f_w$形式要求它在每个状态下都具有零均值：
$\sum_a \pi (s, a) f_w(s, a) = 0，\forall s \in \mathcal{S}$。
从这个意义上说，将 $f_w$ 视为优势函数 $A^\pi(s, a) = Q^\pi(s, a) − V^\pi(s)$ 的近似（类似于 Baird, 1993）而非 $Q^\pi$ 的近似，会更为恰当。

我们的收敛要求 (3) 实质上在于：$f_w$ 必须在每个状态内正确反映动作之间的相对价值，而无需追求绝对数值，也无需刻画状态间的变化。
因此，我们的结果可被视为对“优势函数作为强化学习值函数逼近目标”这一特殊地位的正式辩护。

事实上，我们的式 (2)、(3) 和 (5) 都可以推广为：在值函数或其近似中任意添加一个仅依赖于状态s的函数。

例如，(5) 可推广为
$\frac{\partial \rho}{\partial \theta} =\sum_s d^\pi(s) \sum_a \frac{\partial \pi(s, a)}{\partial \theta} [f_w(s_t, a_t)  + v(s)]$ ,
- $v : S \rightarrow \mathbb{R}$ 是任意函数。（该式成立是因为 $\sum_a \frac{\partial \pi(s, a)}{\partial \theta} = 0，\forall s \in \mathcal{S}$）

v 的选择不影响我们的任何定理，但会显著改变梯度估计量的方差。这里的问题与早期工作中强化基线的使用完全类似（例如 Williams, 1992；Dayan, 1991；Sutton, 1984）。在实践中，v 显然应设为当前最佳的 $V^\pi$ 近似。我们的结果确立：这一近似过程可以在不影响 $f_w$  和 $\pi$  的期望演化的前提下进行。


### 4 Convergence of Policy Iteration with Function Approximation

Theorem 3 (Policy Iteration with Function Approximation).

于是，对于任意具有有界回报的 MDP，由任意初始 $θ_0$、策略序列 $\pi_k = \pi(·,·,\pi_k)$ 以及满足(3)

$$\sum_s d^\pi(s) \sum_a \pi(s, a)[Q^\pi (s_t, a_t) - f_w(s_t, a_t)]\frac{\partial f_w(s_t, a_t)}{\partial w} = 0$$

的权重 $w_k = w$，通过更新
$$\theta_{k+1} = \theta_k + \alpha_k \sum_s d^\pi(s) \sum_a \frac{\partial \pi(s, a)}{\partial \theta}   f_w(s_t, a_t)$$
所生成的序列 {(θₖ, wₖ)}，收敛于
$$\lim_{k\rightarrow} \frac{\partial \rho(\pi_k)}{\partial \theta} = 0$$


证明：我们的定理 2 保证 $θ_k$ 的更新方向是梯度方向；策略二阶导数 $\frac{\partial ^2 \pi(s,a)}{\partial \theta_i \partial \theta_j}$ 以及 MDP 回报的有界性共同确保 $∂²ρ/∂θᵢ∂θⱼ$ 也有界。这些条件，连同步长要求，正是应用 Bertsekas & Tsitsiklis (1996) 第 96 页命题 3.5 所需的全部前提，该命题保证收敛到局部最优。证毕。



