## Learning to predict by the methods of temporal differences

paper Link: [Time-derivative models of Pavlovian reinforcement](https://www.researchgate.net/publication/235419082_Time-Derivative_Models_of_Pavlovian_Reinforcement)

### 简介

文章提出了一个假设：经典条件反射中的强化（reinforcement）是先天（US-无条件刺激）和习得（CS-条件刺激）关联的复合体的时间导数。这种假设认为，强化信号是基于当前刺激与预期之间的差异，而不是单纯的刺激出现。
文章将时间导数模型与Rescorla-Wagner模型进行了对比。Rescorla-Wagner模型假设学习发生在事件违背预期时，即实际的US水平与预期水平之间的差异。时间导数模型则进一步将这种差异形式化为时间导。
本文从动物学习理论的角度对这一理论进行了激励和解释，并表明TD模型用更简单的时间导数模型解决了ISI问题（先天US-习得CS期间的刺激）和其他问题。


### 理论框架

> discuss time-derivative theories. A learning theory should predict how the associations between CSs and USs change.
> 经典条件反射是动物试图根据CS提供的线索预测US的表现。

$\Delta V = (\text{level of US processing}) \times (\text{level of CS processing}) \ \ .... (1)$
$\Delta V = (\text{Reinforcement}) \times (\text{Eligibility}) \ \ .... (2)$
- $\text{Reinforcement} \in \mathbf{R}$
- $\text{Eligibility} \ge 0$

>  Models that treat entire trials as wholes are called **trial-level models**
>  Models that apply continuously, on a moment by moment basis, are called **real-time models**

#### Time-Derivative Theories of Reinforcement

Rescorla-Wagner 模型通过引入预测误差的概念，强调了学习是一个动态的、基于预期的过程，这使得它在心理学和行为科学中具有重要的理论和实践价值
$\Delta V = \beta (\lambda - \hat{V}) \times \alpha X$
其中：
- $\Delta V$：条件刺激与非条件刺激之间关联强度的变化。
- $\beta \ge 0$：非条件刺激(UC)的显著性。
- $\lambda \ge 0$：关联强度的最大可能值（渐近线），表示对非条件刺激的完全预期。
- $\hat{V}$：当前条件刺激与非条件刺激之间的关联强度的期望或是预测值
- $\alpha \ge 0$：条件刺激(CS)的显著性（或注意力）。
- $X = 1$: 

关键概念
- 预测误差（Prediction Error）：模型的核心概念是预测误差，即预期结果与实际结果之间的差异。如果实际结果出乎意料，学习速度会加快；如果结果符合预期，学习速度则会减慢。
- 阻断效应（Blocking Effect）：当一个已经与非条件刺激建立强关联的条件刺激存在时，新加入的条件刺激很难再与非条件刺激建立关联。
- 学习曲线（Learning Curve）：学习初期进展迅速，随着时间推移，学习速度逐渐减慢，直到接近最大关联强度。
- 遮蔽效应（Overshadowing Effect）：当两个条件刺激同时出现时，较强的刺激会“遮蔽”较弱的刺激，限制其与非条件刺激建立关联。


Ẏ 理论: 在任何时间点的强化信号等于当前刺激与反应之间关联的净时间导数
$Ẏ(t) = Y(t) - Y(t-\Delta t)$ 能够完全解释 Rescorla-Wagner 模型的所有预测。

#### Real-Time Theories of Eligibility



1. 强化与时间导数的关系
   - 文章提出了一个核心观点：Pavlovian 强化可以被视为刺激与反应之间关联的净时间导数。
   - 具体来说，强化信号（reinforcement signal）在刺激（CS）的出现和消失时刻产生，分别对应正向和负向的强化。这种时间导数模型能够解释经典条件反射中的许多现象，例如阻断（blocking）和遮蔽（overshadowing）。
2. 时间导数模型（ $Y$ 理论）
   - 文章提出了一个基于时间导数的强化理论，称为  Y˙理论。该理论认为，在任何时间点的强化信号等于当前刺激与反应之间关联的净时间导数。这个理论能够完全解释 Rescorla-Wagner 模型的所有预测。
3. 时间导数模型的变体, 文章讨论了多种时间导数模型的变体，包括：
   - **简单时间导数模型**：基于  Y˙的强化信号，但在某些情况下（如长间隔的延迟条件反射）存在问题。
   - **Klopf 的驱动强化模型（DR 模型）**：通过在刺激出现时触发资格（eligibility）并使其遵循固定的时间过程来解决长间隔问题。
   - **时间差分（TD）模型**：通过引入折扣因子（discount factor）来解决时间导数模型中的问题。TD 模型基于预测未来强化信号的加权和，能够更好地解释实验数据。
4. TD 模型的优势: TD 模型被认为是时**间导数模型中最为成功的一种，它能够解释多种经典条件反射现象**，包括：
   - 阻断现象：如果第一个阶段的训练达到渐近水平，那么在第二个阶段加入的新刺激不会形成新的关联。
   - 延迟条件反射：在长间隔条件下，TD 模型预测延迟条件反射的效果会减弱，这与实验数据一致。
   - 序列复合实验：TD 模型能够更好地解释序列复合实验中的数据。
5. 理论意义与局限性
   - 文章指出，时间导数模型提供了一个统一的框架来解释经典条件反射中的多种现象，包括单刺激学习、高阶条件反射和序列效应。然而，这些模型也存在局限性，例如未能直接解释注意力、刺激显著性、配置效应等现象。



### 结论

文章总结认为，基于时间导数的强化理论能够解释经典条件反射中的许多现象，并且提供了关于学习功能的计算理论基础。尽管如此，这些模型仍有待进一步完善，以更好地整合其他经典条件反射现象的解释。
这篇文章为理解经典条件反射中的强化机制提供了重要的理论基础，特别是在时间导数模型和 TD 模型的提出上，对后续研究产生了深远影响



