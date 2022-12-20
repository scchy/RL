# 策略梯度算法

## 一、基于价值的方法与基于策略的方法
Q-learning、DQN及DQN改进算法都是**基于价值**(`value-based`)的方法，其中Q-learning是处理有限状态的算法，而DQN可以用来处理连续状态的问题。而策略梯度算法是**基于策略**(`Policy-based`)的方法。
两者异同：
- 差异：
  - 基于价值的方法，<font color=darkred>主要学习**值函数**，学习过程并不是一个显式的策略</font>
  - 基于策略的方法，<font color=darkred>是直接显式的学习一个目标策略</font>

## 二、策略梯度
我们希望能够针对当前状态学习到最优策略，用该策略可以在环境中得到最大的期望收益。所以我们可以将策略学习的目标函数定义如下：
$$J(\theta)=E_{s_0}[V^{\pi_\theta}(s_0)]$$
注：
- $V^{\pi_\theta}(s_0)$ 是状态$s_0$下的策略函数，参数为$\theta$

### 2.1 策略梯度证明
$\nabla_\theta V^{\pi_\theta}(s)=\nabla_\theta (\sum_{a \in A}\pi_\theta(a|s) Q^{\pi_\theta}(s, a))$

$=\sum_{a \in A}(\nabla_\theta\pi_\theta(a|s) Q^{\pi_\theta}(s, a) + \pi_\theta(a|s) \nabla_\theta Q^{\pi_\theta}(s, a))$

$=\sum_{a \in A}(\nabla_\theta\pi_\theta(a|s) Q^{\pi_\theta}(s, a) + \pi_\theta(a|s) \nabla_\theta \sum_{s', r}p(s',r|s, a)(r + \gamma V^{\pi_\theta}(s'))$

$=\sum_{a \in A}(\nabla_\theta\pi_\theta(a|s) Q^{\pi_\theta}(s, a) + \gamma\pi_\theta(a|s) \nabla_\theta \sum_{s'}p(s'|s, a)V^{\pi_\theta}(s'))$

令$\phi(s)=\sum_{a \in A}(\nabla_\theta\pi_\theta(a|s) Q^{\pi_\theta}(s, a)$简化推导

