
# Assignment 1: Imitation Learning

- [berkeley-PDF-HW02](https://rail.eecs.berkeley.edu/deeprlcourse/deeprlcourse/static/homeworks/hw2.pdf)
- [Github-hw2](https://github.com/berkeleydeeprlcourse/homework_fall2023/blob/main/hw2/README.md)

目标->试验策略梯度及其变体, 包括方差减少技巧(baseline)
- implementing reward-to-go 
- neural network baselines.

# Review

## 1. Policy gradient

policy gradient
$$\nabla_\theta J(\theta) = \nabla_\theta \int \pi_\theta(\tau)r(\tau)d\tau\\=\int \pi_\theta(\tau) \nabla_\theta log \pi_\theta (\tau) r(\tau)d\tau\\=E_{\tau \sim \pi_\theta (\tau)}[\nabla_\theta log \pi_\theta (\tau)r(\tau)]$$

approximated from a batch of N sampled trajectories
$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \nabla_\theta log \pi_\theta (\tau)r(\tau)\\=\frac{1}{N} \sum_{i=1}^N (\sum_{t=0}^{T-1}\nabla_\theta log \pi_\theta (a_{it}|s_{it}))(\sum_{t=0}^{T-1}r(a_{it}, s_{it}))$$

## 2. VarianceReduction
### 2.1 Reward-to-go
$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N (\sum_{t=0}^{T-1}\nabla_\theta log \pi_\theta (a_{it}|s_{it}))(\sum_{t^\prime=t}^{T-1}r(a_{it^\prime}, s_{it^\prime}))$$


### 2.2 Discounting 
- full 
$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N (\sum_{t=0}^{T-1}\nabla_\theta log \pi_\theta (a_{it}|s_{it}))(\sum_{t=0}^{T-1}\gamma^{t - 1} r(a_{it}, s_{it}))$$

- Reward-to-go
$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N (\sum_{t=0}^{T-1}\nabla_\theta log \pi_\theta (a_{it}|s_{it}))(\sum_{t^\prime=t}^{T-1}\gamma^{t^\prime - t} r(a_{it^\prime}, s_{it^\prime}))$$

### 2.3 Baseline

$$\nabla_\theta J(\theta) = \nabla_\theta E_{\tau \sim \pi_\theta (\tau)}[ log \pi_\theta (\tau)(r(\tau) -b )]$$

$E_{\tau \sim \pi_\theta (\tau)}[ \nabla_\theta log \pi_\theta (\tau)b]=\int \pi_\theta(\tau)  \nabla_\theta log \pi_\theta (\tau)b  d\tau=\nabla_\theta \int  \pi_\theta (\tau)b d\tau = \nabla_\theta 1 b$

- if b is not depend of $\pi_\theta$ $\nabla_\theta 1 b=0$

reward-to-go + baseline
$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N (\sum_{t=0}^{T-1}\nabla_\theta log \pi_\theta (a_{it}|s_{it}))(\sum_{t^\prime=t}^{T-1}\gamma^{t^\prime - t} r(a_{it^\prime}, s_{it^\prime}) - V^\pi_\phi(s_{it}))$$


### 2.4 GAE: Generalized Advantage Estimation

### advantage function

$A^\pi (s_t, a_t)=\sum_{t^\prime=t}^{T-1}\gamma^{t^\prime - t} r(a_{it^\prime}, s_{it^\prime}) - V^\pi_\phi(s_{t})=Q^\pi(s_t, a_t) - V^\pi_\phi(s_{t})$

$A^\pi (s_t, a_t)\approx \delta_t=r(s_t, a_t) + \gamma V^\pi_\phi(s_{t+1}) - V^\pi_\phi(s_{t})$

### GAE

$A_n^\pi (s_t, a_t) = \sum_{t^\prime = t}^{t+n} \gamma^{t^\prime - t}r(s_{t^\prime}, a_{t^\prime}) + \gamma^n V^\pi_\phi(s_{t+n+1}) - V^\pi_\phi(s_{t})$


$A_1^\pi (s_t, a_t) = r(s_t, a_t) + \gamma V^\pi_\phi(s_{t+1}) - V^\pi_\phi(s_{t})=\delta_t$  
$A_2^\pi (s_t, a_t) = r(s_t, a_t) + \gamma (r(s_{t+1}, a_{t+1})+ \gamma V^\pi_\phi(s_{t+2})) - V^\pi_\phi(s_{t})\\=r_t+\gamma r_{t+1} + \gamma^2 V_{t+2} - V_t\\=\delta_t - \gamma V_{t+1} + \gamma r_{t+1} + \gamma^2 V_{t+2}\\=\delta_t + \gamma \delta_{t+1}$

$A_n = \sum_{i=1}^{n} \gamma ^{(i-1)} \delta_i$


$A_{GAE}(t)=\sum_{i}^T \lambda^i A_t^i = \sum_{i}^T \lambda^i \sum_j^{i}\gamma^{j-1} \delta_j$

展开如下

$A_{GAE}(t)= \delta _t + \lambda (\delta _t + \gamma \delta_{t+1})+ \lambda^2 (\delta _t + \gamma \delta_{t+1} + \gamma^2 \delta_{t+2}) + ...=\delta_t (1 + \lambda + \lambda ^ 2 + ...) + \gamma \delta_{t+1}(\lambda + \lambda ^ 2 + ...)+ \gamma^2 \delta_{t+2}(\lambda ^ 2 + ...)\\=\delta_t \frac{1*(1-\lambda^n)}{1-\lambda} + \gamma \delta_{t+1} \frac{\lambda*(1-\lambda^{n-1})}{1-\lambda} + \gamma^2 \delta_{t+2} \frac{\lambda^2*(1-\lambda^{n-2})}{1-\lambda}$

$\lim_{n \rightarrow \infin} (1-\lambda) A_{GAE}(t)\\=\delta _t + \lambda \gamma \delta _{t+1} + (\lambda \gamma)^2 \delta _{t+2}... \\= \sum_i^n (\lambda \gamma)^{i-1}\delta_i$

reward-to-go  and define $A_{GAE}(t) \propto (1-\lambda) A_{GAE}(t)$  

$A_{GAE}(t) = \sum_{t^\prime=t}^T (\lambda \gamma)^{t^\prime-t}\delta_{t^\prime}$

$A_{GAE}(t) = \delta_t + \lambda \gamma A_{GAE}(t+1)$

```python
def compute_GAE(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
```

# 8- Analysis

