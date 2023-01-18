
# 基于价值的DRL方法总结

|方法| action描述 | state描述 | QNet | TagetQNet | action选取 | QTarget | loss | 备注 |
|-|-|-|-|-|-|-|-|-|
| DQN | 仅离散动作 | 支持连续状态 | QNet(state) -> q | deepcopy(QNet) | $a=max(TagetQNet(s_{t+1}))$| $q_{t+1}=TagetQNet(s_{t+1})[a];\\  QTarget=r + \gamma * q_{t+1}$ | MSE(QNet(state), QTarget)| 对传统Qtable状态空间有限的拓展 |
| doubleDQN | 仅离散动作 | 支持连续状态 | QNet(state) -> q | deepcopy(QNet) | <font color=darkred>$a=max(QNet(s_{t+1}))$</font>| $q_{t+1}=TagetQNet(s_{t+1})[a];\\  QTarget=r + \gamma * q_{t+1}$ | MSE(QNet(state), QTarget)| 对DQN Qtarget高估的修正|
| DuelingDQN | 仅离散动作 | 支持连续状态 | VNet(state) -> <font color=darkred>V + A - mean(A) </font>-> q | deepcopy(VNet) | $a=max(TagetQNet(s_{t+1}))$| $q_{t+1}=TagetQNet(s_{t+1})[a];\\ QTarget=r + \gamma * q_{t+1}$ | MSE(QNet(state), QTarget)| 拆分成价值函数和优势函数计算q,另一种修正QTagret高估方法 |
