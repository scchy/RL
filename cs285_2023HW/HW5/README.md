
# Assignment 5:  Exploration Strategies and Offline Reinforcement Learning

- [berkeley-PDF-HW05](https://rail.eecs.berkeley.edu/deeprlcourse/deeprlcourse/static/homeworks/hw5.pdf)
- [Github-hw5](https://github.com/berkeleydeeprlcourse/homework_fall2023/tree/main/hw5)



## Analysis 

分析如何利用奖励奖金来应对离线强化学习中的分布偏移(`OOD`)问题

MDP $M = (\mathcal{S}, \mathcal{A}, r, p)$, 离线策略$\pi _\beta$ 收集数据 $\mathcal{D}$, 我们将用 $p_\pi$ 表示策略 $\pi$ 所诱导的状态分布。

以下均基于 SAC 算法阐述。对优化目标进行调整  
$E_{\pi(a|s)}[Q(s, a)] \rightarrow E_{\pi(a|s)}[Q(s, a) + \lambda b(s,a)]$
- $b(s,a) = -log \pi(a|s)$

我们也可以通过直接将熵奖励加到即时奖励上的方式来施加类似形式的熵正则化。在期望意义上，我们将优化
$\mathbb{E}_{s,a\sim p_\pi}[r(s, a) + \lambda b(s,a)]=1/H J(\pi) - \lambda \mathbb{E}_{s\sim p_\pi}\mathbb{E}_{\pi(a|s)} log \pi(a|s)=1/H J(\pi) + \lambda \mathbb{E}_{s\sim p_\pi} \mathcal{H}[\pi(a|s)]$

这种做法在期望上与向策略梯度添加负熵项完全等价，但实现上无需改动策略网络，仅需在奖励函数中额外加入熵奖励即可。

我们希望从离线数据 D 中学习一个 Q 函数和策略 π，并在约束 $D(\pi, \pi_\beta) \le \epsilon$ 下执行如下更新：  
$Q(s, a) \leftarrow \hat{r}(s, a) + E_{a^\prime \sim \pi}[Q(s^\prime, a^\prime)]$
- $\pi = \argmax_{\pi}\mathbb{E}_{s, a\sim p_\pi}[Q(s, a)]$

把约束转为拉格朗乘数：

$\pi = \argmax_{\pi}\mathbb{E}_{s, a\sim p_\pi}[Q(s, a)] - \lambda D(\pi, \pi_\beta)$

1. KL-divergence(Reverse KL) : $D(\pi, \pi_\beta) = E_{s\sim p_\pi} D_{KL}(\pi(a|s) || \pi_\beta(a|s))$
     - $P: \pi_\beta(a|s); Q:\pi(a|s) \rightarrow D_{KL}(Q||P)=\sum Q log \frac{Q}{P}$
     - Q为预估分布
     - Q-Answer: $\hat{r}(s, a) = r(s, a) + \lambda b(s, a); b(s, a) \rightarrow 1/D(\pi, \pi_\beta)$

2. f-散度: $D(\pi, \pi_\beta) = E_{s\sim p_\pi} D_f[\pi(a|s) || \pi_\beta(a|s)] = E_{s\sim p_\pi} \pi_\beta(a|s) f(\frac{\pi(a|s)}{\pi_\beta(a|s)})$
     - $D_f[P||Q]=\int Q(x)f\frac{P(x)}{Q(x)}$ 
     - 其中 $f:\mathbb{R} \rightarrow \mathbb{R}$ 是一个凸函数且 $f(1)=0$
     - f-散度约束使我们能够指定 $\pi$ 与 $\pi_\beta$ 之间散度的多种替代约束形式。例如，  
       - 取 $f(x) = 1/2 |x − 1|$ 时，f-散度等价于总变差距离（TVD）；  
       - 取 $f(x) = −(x + 1) log((x + 1)/2) + x log x$时，它退化为 Jensen-Shannon 散度（JSD）。  
       - 当取 $f(x) = x log x \rightarrow Qf(P/Q)= Q \frac{P}{Q} log(\frac{P}{Q})=Plog(\frac{P}{Q})$  时，我们得到标准的 KL 散度。


3. 假设 MDP M 具有有限 horizon H，我们希望对 $\pi$ 与 $\pi_\beta$ 下状态轨迹分布的散度加以约束。可将轨迹 $τ = (s_1, s_2, …, s_H)$ 的状态分布之间的 KL 散度表示为：
   - $D(\pi, \pi_\beta) = D_{KL}[p^\pi(\tau) || p^{\pi_\beta}(\tau)]$


## Code

1. RND(random network distillation): 使用 RND 探索程序收集数据
2. CQL: conservative Qlearning
3. AWAC: Advantage Weighted Actor Critic
4. IQL: Implicit Q-Learning 

我们将使用**不同难度的网格域**(`easy`, `medium`, `hard`)来训练Agent 
- `easy`: agent需要沿两条走廊前进，并在中间右转
- `medium`: 一个需要多次转弯的迷宫；
- `hard`: “四房间”任务，agent必须穿过狭窄通道在多个房间之间来回移动，才能抵达目标位置。
- `ex-hard`: 加分部分


### random network distillation (RND) algorithm 

一种常见的探索方式是去访问“某个量”预测误差大的状态，例如 TD 误差，甚至是随机函数的误差。
第 13 讲介绍的 RND 算法正是利用这一思想：通过鼓励探索策略更多地经过“随机神经网络函数预测误差高”的转移，从而促进探索。

形式上，令$f^\star _\theta(s^\prime)$ 为一个随机选取的、由神经网络标识的向量值函数。
RND训练另一个神经网络 $\hat{f}_\phi(s^\prime)$, 使其在缓冲区域数据分布下逼近 $f^\star _\theta(s^\prime)$ 的输出， 具体如下：


$$
\phi^* = \arg\min_\phi \; \mathbb{E}_{s,a,s'\sim\mathcal{D}}\!\left[  \underbrace{\bigl\|\hat{f}_\phi(s')-f^*_\theta(s')\bigr\|}_{\epsilon_{\phi}(s')}\right]
$$

如果 $(s, a, s^\prime )$ 在缓存中，预估错误$\epsilon_{\phi}(s')$需要比较小。其他未见过的 state-action 预估误差会比较大。
为了把这一预测误差用作探索奖励加成，RND 训练两个 Critic：
- exploitation-Critic $Q_R(s,a)$：估计策略在**真实奖励函数下的回报**——正常RL-Agent, 比如DQN
- exploration-Critic $Q_\epsilon (s,a)$：估计探索奖励——RND
  - RND-Target-Net[ $f^\star _\theta(s^\prime)$ ]: 一个随机初始化且永远固定的神经网络
  - RND-Net[ $\hat{f}_\phi(s^\prime)$ ]: 一个可学习的网络去模仿`RND-Target-Net`
  - 两网络在从未见过的状态上必然预测误差大，就把这个误差当成内在探索奖励——误差越大越“新鲜”，从而驱动策略主动去这些“陌生”地区收集数据。


在实际操作中，我们会先把误差归一化，因为该误差在不同状态间的量级可能差异很大，直接输入会导致优化过程不稳定。
在本题中，我们用随机神经网络来表示 RND 所使用的随机函数 $f^\star _\theta(s^\prime)$ 与  $\hat{f}_\phi(s^\prime)$。
为了避免网络一开局就输出零预测误差，我们在 `agents/rnd_agent.py` 里对目标网络$f^\star _\theta(s^\prime)$采用了另一种权重初始化方案。


### Conservative Q-Learning (CQL) Algorithm

CQL的目标是防止高估策略。

conservative, lower-bound Q-function: 
- 通过在标准`Bellman error`边上增加额外最小化Q值。
- 训练的时候增加正则项： $\alpha \left[ \frac{1}{N} \sum_{i=1}^N (log( \sum_a e^{Q(s_i, a)} ) - Q(s_i, a_i)) \right]$
  - 在$s_i$下用$\pi$采样action, 使得OOD state-action的Q预估值变小

$$J=\left[ \frac{1}{N} \sum_{i=1}^N (Q(s_i, a_i) - (r + \gamma Q(s_i^\prime, a_i^\prime)))^ 2 \right] + \alpha \left[ \frac{1}{N} \sum_{i=1}^N (log( \sum_a e^{Q(s_i, a)} ) - Q(s_i, a_i)) \right]$$


###  Advantage Weighted Actor Critic (AWAC) Algorithm 

增强actor训练

$$\theta \leftarrow \argmax_\theta \mathbb{E}_{s, a\sim \mathcal{B}} \left[  log \pi_\theta(a|s) exp(\frac{1}{\lambda} \mathcal{A}^{\pi _k}(s, a) ) \right]$$
- $\mathcal{A}^{\pi _k} = V^{\pi _k}(s) - Q(s, a)$
- $V^{\pi _k}(s) = \sum_a \pi _k (a|s) Q(s, a)$
  - SAC: $V^{\pi _k}(s) = \sum_a \pi _k(a|s) \left[ Q(s, a) -  log\pi _k(a|s) \right]$

这个迭代和 wieghted behavior cloning 很像。我们让策略偏向于选择那些在我们学到的 Q 函数下取值较高的动作。在上述的迭代中，智能体以较大权重向高优势动作回归，几乎忽略低优势动作
这种 Actor 更新等价于加权最大似然（即监督学习）：
只需从回放缓冲区 β 中采样 (s, a)，用学得的价值网络预测出的优势值对当前数据集中的状态-动作对进行重加权，即可获得训练目标；无需显式学习任何参数化行为模型。

Q函数使用TDLoss 进行迭代
$$\mathbb{E}_D[ (Q(s, a) - (r(s, a) + \gamma \mathbb{E}_{a^\prime \sim \pi}[ Q_{\phi_{k-1}}(s^\prime, a^\prime) ] ) ) ^ 2]$$

- $a^\prime$ 是从策略$\pi$中采样的，意味如果 π 成功拟合了（加权后的）行为策略，就不会采样到 OOD（分布外）动作。


### Implicit Q-Learning (IQL) Algorithm

IQL 通过使用implicit Bellman backup 而不是显式地考虑在特定策略下的backup，将学习评论家（critic）的问题与策略学习解耦。
它通过**学习 Q 的期望分位数（expectile）**来实现——这是一种类似于分位数的统计量，是分布所达到最大值的“软”版本。
对于随机变量 X，期望分位数 $m_\tau(X)$ 由最小化以下目标给出：
$$\argmin_{m_\tau} \mathbb{E}_{x \sim X}[L_2(x-m_\tau)], L_2(\mu) = |\tau - \mathbb{1}\{\mu \le 0 \}|\mu ^2$$


为了避免对状态转移过于乐观，我们需要学习一个独立的价值函数 V(s) 来承担这种乐观性，

$$L_V(\phi) = \mathbb{E}_{(s, a) \sim D}[ L_2^\tau (Q_\theta (s, a) - V_\phi (s)) ]$$
$$L_Q(\theta) = \mathbb{E}_{(s, a, s^\prime) \sim D}[(r(s, a) + \gamma V_\phi(s^\prime) - Q_\theta (s, a))^2  ]$$

critic 学习过程有2个好的属性
1. 因为迭代没有用到action，所以永远没有使用OOD的action, 从而完全避免了分布外（OOD）过估计问题
   1. $a^\prime \sim \pi$ 任意策略
2. 这一过程可以在不更新 Actor 的情况下进行，因此如果你愿意，可以先单独训练 Critic，最后再训练 Actor。

Actor 的更新与 Critic 的更新是解耦的（因此称为 implicit Q-learning），并且它使用与 AWAC 相同的目标函数进行学习：

$$L_\pi(\psi)=-\mathbb{E}_{s, a \sim \mathcal{B}}\left[ log \pi_\psi (a|s) exp(\frac{1}{\lambda} \mathcal{A}^{\pi _k}(s, a)) \right]$$

