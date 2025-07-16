
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

