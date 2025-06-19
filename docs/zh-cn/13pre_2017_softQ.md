## Reinforcement Learning with Deep Energy-Based Policies

paper Link: 
- [Reinforcement Learning with Deep Energy-Based Policies](https://arxiv.org/abs/1702.08165)
- [Equivalence Between Policy Gradients and Soft Q-Learning](https://ar5iv.labs.arxiv.org/html/1704.06440)

其他相关链接: 
- [Soft Q-Learning论文阅读笔记](https://zhuanlan.zhihu.com/p/76681229)
- [www.lamda.nju.edu.cn slides](https://www.lamda.nju.edu.cn/xufeng/websites/tlrg/slides/pre_16.pdf)
- [Learning Diverse Skills via Maximum Entropy Deep Reinforcement Learning](https://bair.berkeley.edu/blog/2017/10/06/soft-q-learning/)

Github:
- [softqlearning](https://github.com/haarnoja/softqlearning)

### 1. 背景
对于无模型强化学习算法，我们从探索(exploration)的角度考虑。尽管随机策略(stochastic policy)看起来负责探索，但是这种探索一般都是启发式的，比如像DDPG算法通过添加噪声，或者像TRPO算法在随机策略基础上增加熵。然而我们需要一个更加直接的探索，也就是最大化期望收益的同时引入最大熵，这样会让策略面对扰动的时候更加稳定：

现在问题变成了针对任意分布的策略去完成最大化熵。这篇文章借用了基于能量的(energy-based)模型(EBM)来表示策略，可以适应多模式，并这反过来揭示了Q-learning，演员评论家算法和概率推理之间的有趣联系。



### 2. 最大熵策略框架

$$\pi^\star_{\text{MaxEnt}} = \argmax_\pi \mathbf{E}_{\pi}[\sum_{t=0}^T r_t + \alpha H(\pi(.|s_t))]$$
- Ziebart 2010
- 熵正则化增加了强化学习算法的探索程度，$\alpha$越大，探索性就越强，有助于加速后续的策略学习，并减少策略陷入较差的局部最优的可能性
- $H(\pi(.|s_t))=-\int_{a_t} \pi (a_t |s_t)log(\pi(a_t|s_t)) da_t = E_{a_t \sim \pi}[-log(\pi(a_t|s_t)]$

注意到，最大熵并不是只关注当前状态下的最大熵，而是考虑到从t时间开始的未来的所有熵的和。 在实际应用中，我们更加偏好最大熵RL模型(`maximum-entropy models`)，因为他们在匹配观测信息时对未知量的假设最小。

#### 2.1 Soft Bellman Equation & Soft Q-Learning 

general energy-based policies of the form:
$$\pi (a_t | s_t) \propto e^{-\mathbf{\epsilon}(s_t, a_t)} $$

定义`Q-function`:
$$Q^\star_{soft}(s_t, a_t)=r_t + E_{(s_{t+1}...) \sim \rho_\pi} [\sum_{l=1}^\infin \gamma^l(r_{t+l} + \alpha H(\pi^\star_{\text{MaxEnt}}(.|s_{t+l})))]$$
- `soft Value function`: smooth maximum
  - $V^\star_{soft}(s_t) = \alpha log \int_A e^{\frac{1}{\alpha}Q^\star_{soft}(s_t, a^\prime )}da^\prime$
- $\pi^\star_{\text{MaxEnt}}(a_t|s_{t})=e^{\frac{1}{\alpha}(Q^\star_{soft}(s_t, a_t) - V^\star_{soft}(s_t) )}$

定义`soft Bellman equation`:
$$Q^\star_{soft}(s_t, a_t)=r_t + \gamma  E_{s_{t+1}\sim p_s}[V^\star_{soft}(s_{t+1})]$$
证明如下：

对`Q-function`做变换
$Q^\star_{soft}(s_t, a_t)=r_t + \gamma E_{s_{t+1}\sim p_s}[\alpha H(\pi(.|s_{t+1})) + E_{a_{t+1}\sim\pi(.|s_{t+1})}[Q^\pi_{soft}(s_{t+1}, a_{t+1})] ]  $
$=r+\gamma E_{s_{t+1}, a_{t+1}}[\alpha H(\pi(a_{t+1}|s_{t+1})) + Q^\pi_{soft}(s_{t+1}, a_{t+1})]$
$=r+\gamma E_{s_{t+1}, a_{t+1}}[-\alpha log(\pi(a_{t+1}|s_{t+1})) + Q^\pi_{soft}(s_{t+1}, a_{t+1})]$
$=r+\gamma E_{s_{t+1}, a_{t+1}}[Q^\pi_{soft}(s_{t+1}, a_{t+1}) -\alpha log(e^{\frac{1}{\alpha}(Q^\star_{soft}(s_{t+1}, a_{t+1}) - V^\star_{soft}(s_{t+1}) )}) ]$
$=r+\gamma E_{s_{t+1}}[V^\star_{soft}(s_{t+1})]$



定义`Soft Q-Iteration`

$Q^\star_{soft}(s_t, a_t) \leftarrow r_t + \gamma E_{s_{t+1} \sim p_s}[V_{soft}(s_{t+1})], \forall s_t, a_t $
$V^\star_{soft}(s_t) \leftarrow  \alpha log \int_A e^{\frac{1}{\alpha}Q^\star_{soft}(s_t, a^\prime )}da^\prime, \forall s_t$


##### Loss定义
$\theta$: `Q Loss`
$J_{Q}(\theta) = MSE(Q_{tar}, Q_{soft}^\theta (s_t, a_t))$
$Q_{tar} = r_t + \gamma E_{s_{t+1}\sim p_s}[V^{\overline{\theta}}_{soft}(s_{t+1})]=\gamma E_{s_{t+1}\sim p_s}[\alpha log E_{q_{a^\prime}}[\frac{e^{\frac{1}{\alpha}Q^{\overline{\theta}}_{soft}(s_t, a^\prime )}}{q_{a^\prime}(a^\prime)}]]$

- $\overline{\theta}$ 目标网络参数
- $V^{\overline{\theta}}_{soft}(s_t)=\alpha log E_{q_{a^\prime}}[\frac{e^{\frac{1}{\alpha}Q^{\overline{\theta}}_{soft}(s_t, a^\prime )}}{q_{a^\prime}(a^\prime)}]$
  - $q_{a^\prime}$ 是动作空间的任意分布, 
    - 可以是均匀分布(`uniform distribution`), 但是在高纬空间表现较差
    - 可以用当前策略，当前策略产出`soft value`的无偏估计


$\phi$: `actor Loss`
$\pi^\phi (a_t|s_t) = a_t=f^\phi(\xi;s_t ); \xi \sim N(0, I)$ 
- 映射随机噪声到正态高斯函数，或者其他分布
- 同时希望其接近`energy-based distribution`，用正向KL离散度($D_{KL}(P||Q)$, P为真实分布)作为损失

$J_{\pi}(\phi; s_t)=D_{KL}(\pi^\star_{MaxEnt}=e^{\frac{1}{\alpha}(Q^\star_{soft}(s_t, a_t) - V^\star_{soft}(s_t) )}||\pi^\phi (a_t|s_t))$


```python

```





