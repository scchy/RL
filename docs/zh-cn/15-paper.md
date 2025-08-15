## 1980年代
- [x] 1983: **Neuronlike adaptive elements that can solve difficult learning control problems**
- [x] 1984: Temporal credit assignment in reinforcement learning
- [x] 1988: **Learning to predict by the methods of temporal differences**

## 1990年代
- [x] 1990: Time-derivative models of Pavlovian reinforcement
- [x] 1991: Dyna, an integrated architecture for learning, planning, and reacting
- [x] 1992: Adapting bias by gradient descent: An incremental version of delta-bar-delta
- [x] 1992: Gain adaptation beats least squares?
- [x] 1991: Reinforcement learning is direct adaptive optimal control
- [x] 1996: Generalization in reinforcement learning: Successful examples using sparse coarse coding
- [x] 1996: Sutton’s 1996 Menu 《**A menu of designs for reinforcement learning over time**》
    - 需要iEEE订阅，等后续再看源文
    - Kim： ask: 帮忙阅读并总结下 Sutton’s 1996 Menu: 《A Menu of Designs for Reinforcement Learning Over Time》
- [ ] [1996: Reinforcement learning with replacing eligibility traces](https://link.springer.com/article/10.1007/BF00114726)
    - 36页
- [ ] 1998: BOOK 《**Reinforcement learning: An introduction**》
- [ ] 1999: **Policy gradient methods for reinforcement learning with function approximation**
- [ ] 1999: **Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning**

## 2000年代
- [ ] 2000: Eligibility traces for off-policy policy evaluation
- [ ] 2008: Incremental natural actor-critic algorithms
- [ ] 2009: Fast gradient-descent methods for temporal-difference learning with linear function approximation

## 2010年代
- [ ] 2011: Gradient temporal-difference learning algorithms
- [ ] 2015: Developing a predictive approach to knowledge
- [ ] 2017: Integral Policy Iterations for Reinforcement Learning Problems in Continuous Time and Space
- [ ] 2017: A first empirical study of emphatic temporal difference learning
- [ ] 2017: Multi-step off-policy learning without importance sampling ratios

## 2020年代
- [ ] 2018: **Reinforcement Learning: An Introduction, 2nd edition**
- [ ] 2024: **The big world hypothesis and its ramifications for artificial intelligence**

----------------------- 

### 1996: Sutton’s 1996 Menu 《**A menu of designs for reinforcement learning over time**》

一句话速览
Sutton 把 “时间维度” 拆成 “何时更新 / 更新什么 / 用什么误差” 三栏菜单，像点菜一样组合即可快速拼出各种 RL 算法，成为后续 20 年 TD、多尺度、元 RL 的共同语法基础。

1. 写作动机
   - 90 年代中期，RL 研究者对“如何显式地处理时间”缺乏统一框架。
   - 作者用 “菜单”隐喻 把复杂算法拆成可组合的 三栏选项，降低设计门槛。

2. 菜单三栏（核心框架）

| 栏                | 典型选项                           | 说明 & 示例                                  |
| ---------------- | ------------------------------ | ---------------------------------------- |
| **更新时机** (When)  | step, episode, batch, delayed  | Q-learning (step), Monte-Carlo (episode) |
| **更新对象** (What)  | value, policy, model           | TD 更新 V, REINFORCE 更新 π                  |
| **目标函数** (Which) | return, advantage, model-error | SARSA (return), Actor-Critic (advantage) |

4 × 3 × 3 = 36 种算法原型 可一键生成。
研究者只需“换一道菜”即可改进收敛、方差或样本效率。


3. 对 TD 的新诠释
    - TD 不是“近似 MC”，而是 “跨时间尺度的自举”。
    - 提出 λ 是时间折扣谱系 的观点，统一了 eligibility trace 的解释。

4. 实践影响

    - 模块化设计：90 年代后期的 λ-policy-gradient、Dyna-2、model-based RL 均可按菜单逐行勾出。
    - 教学价值：成为 RL 课程里解释 “算法族谱” 的标准教具。

5. 一句话总结

这篇 6 页短文把 RL 时间维度做成 “时机-对象-误差” 三栏菜单，让复杂算法像乐高一样可拼装，奠定了 时间差分、多尺度学习、元 RL 的共同语法。



