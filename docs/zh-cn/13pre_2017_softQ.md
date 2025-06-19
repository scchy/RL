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


#### 2.2 Loss定义

#### 2.2.1 $\theta$: `Q Loss`
$J_{Q}(\theta) = MSE(Q_{tar}, Q_{soft}^\theta (s_t, a_t))$
$Q_{tar} = r_t + \gamma E_{s_{t+1}\sim p_s}[V^{\overline{\theta}}_{soft}(s_{t+1})]=\gamma E_{s_{t+1}\sim p_s}[\alpha log E_{q_{a^\prime}}[\frac{e^{\frac{1}{\alpha}Q^{\overline{\theta}}_{soft}(s_t, a^\prime )}}{q_{a^\prime}(a^\prime)}]]$

- $\overline{\theta}$ 目标网络参数
- $V^{\overline{\theta}}_{soft}(s_t)=\alpha log E_{q_{a^\prime}}[\frac{e^{\frac{1}{\alpha}Q^{\overline{\theta}}_{soft}(s_t, a^\prime )}}{q_{a^\prime}(a^\prime)}]=log E_{q_{a^\prime}}[e^{Q^{\overline{\theta}}_{soft}(s_t, a^\prime )}]- \alpha log E_{q_{a^\prime}}[q_{a^\prime}(a^\prime)]$
  - $q_{a^\prime}$ 是动作空间的任意分布, 
    - 可以是均匀分布(`uniform distribution`), 但是在高纬空间表现较差
    - 可以用当前策略，当前策略产出`soft value`的无偏估计

实际Code实现
  - 随机采样N个动作$a^\prime$： (-1, 1)均匀分布: 
    - `random_actions = torch.rand(1, self.value_n_particles, self.action_dim).to(self.device);`
    - `tar_action = 2 * random_actions - 1`
  - 计算$V^{\overline{\theta}}_{soft}(s_t)=log E_{q_{a^\prime}}[e^{Q^{\overline{\theta}}_{soft}(s_t, a^\prime )}]- \alpha log E_{q_{a^\prime}}[q_{a^\prime}(a^\prime)]$
    - `q_next = self.tar_critic(next_state[:, None, :], tar_action);`
    - $log E_{q_{a^\prime}}[e^{Q^{\overline{\theta}}_{soft}(s_t, a^\prime )}]$ logsumexp平滑 : 
      - `torch.logsumexp(q_next, dim=1) - torch.log(torch.tensor(self.value_n_particles).to(self.device))
      q_next += self.action_dim * np.log(2)`
      - 减去粒子数量的对数来得到平均值
    - $-\alpha log E_{q_{a^\prime}}[q_{a^\prime}(a^\prime)]$: 
      - $\alpha = 1$
      - 调整随机动作分布 $q_{a^\prime}(a^\prime)$ 的熵，确保在计算$Q_{tar}$时考虑了动作分布的熵。
        - 均匀分布 $[−1,1)$ 其熵为 $H(X) = -\frac{1}{2}log\frac{1}{2} \int^1_{-1}1dx= -log\frac{1}{2} = log2$，动作维度为`action_dim`
      - `q_next += self.action_dim * np.log(2)`

**实际Code: Python-Torch**
```python
    def _create_td_update(self, state, action, reward, next_state, done):
        # 生成 [0, 1) 范围内的随机数
        random_actions = torch.rand(1, self.value_n_particles, self.action_dim).to(self.device)
        # 将范围调整到 [-1, 1)
        tar_action = 2 * random_actions - 1
        q_next = self.tar_critic(next_state[:, None, :], tar_action)
        # Equation 10:
        q_next = torch.logsumexp(q_next, dim=1) - torch.log(torch.tensor(self.value_n_particles).to(self.device))
        q_next += self.action_dim * np.log(2)
        
        q_tar = self.reward_scale * reward + self.gamma * q_next * (1 - done)
        q = self.critic(state, action)

        # # Equation 11:
        critic_loss = torch.mean(0.5 * (q_tar.detach() - q) ** 2)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
```



#### 2.2.2 $\phi$: `actor Loss`
$\pi^\phi (a_t|s_t) = a_t=f^\phi(\xi;s_t ); \xi \sim N(0, I)$ 
- 映射随机噪声到正态高斯函数，或者其他分布
- 同时希望其接近`energy-based distribution`，用正向KL离散度($D_{KL}(P||Q)$, P为真实分布)作为损失

$J_{\pi}(\phi; s_t)=D_{KL}(\pi^\star_{MaxEnt}=e^{\frac{1}{\alpha}(Q^\star_{soft}(s_t, a_t) - V^\star_{soft}(s_t) )}||\pi^\phi (a_t|s_t))$

做SVGD近似：
- $\triangle f^{\phi}(.;s_t) = E_{a_t \sim \pi ^ \phi}[\mathbf{k}(a_t, f^{\phi}(.;s_t))\nabla _{a^\prime} Q^\theta_{soft}(s_t, a^\prime)|_{a\prime=a_t} + \alpha \nabla _{a^\prime} \mathbf{k}(a_t, f^{\phi}(.;s_t))|_{a^\prime = a_t}]$
- $\frac{\partial{J_\pi (\phi; s_t)}}{\partial \phi} \propto E_{\xi }[\triangle f^{\phi}(\xi;s_t) \frac{\partial{f^{\phi}(\xi;s_t)}}{\partial \phi}]$

实际Code实现
  - 采样N个动作 & 拆分成`fixed_actions, updated_actions`:
    -  `act = self.actor(state, n_action_samples=self.kernel_n_particles)`
    -  `fixed_actions, updated_actions = torch.split(act,  [n_fixed_actions, n_updated_actions],  dim=1)`
  - 计算 $\triangle f^{\phi}(.;s_t) = E_{a_t \sim \pi ^ \phi}[\mathbf{k}(a_t, f^{\phi}(.;s_t))\nabla _{a^\prime} Q^\theta_{soft}(s_t, a^\prime)|_{a\prime=a_t} + \alpha \nabla _{a^\prime} \mathbf{k}(a_t, f^{\phi}(.;s_t))|_{a^\prime = a_t}]$
  1. 用`fixed_actions`, 计算Q函数梯度：$\nabla _{a^\prime} Q^\theta_{soft}(s_t, a^\prime)|_{a\prime=a_t}$
```python
        svgd_tar_values = self.critic(state[:, None, :], fixed_actions)
        # Target log-density. Q_soft in Equation 13:
        squash_corr = torch.sum(torch.log(1 - fixed_actions**2 + EPS), dim=-1)
        log_p = svgd_tar_values.squeeze(2) + squash_corr
        # 计算 log_p 对 fixed_actions 的梯度
        grad_log_p = torch.autograd.grad(
            log_p, fixed_actions, 
            grad_outputs=torch.ones_like(log_p), 
            create_graph=True
        )[0]
        grad_log_p = grad_log_p.unsqueeze(2).detach()
```
  1. 用`fixed_actions`和`updated_actions`计算核函数（`gaussian_kernel`）结果： $\mathbf{k}(a_t, f^{\phi}(.;s_t))$ 
```python
kernel_dict = self.kernel_fn(xs=fixed_actions, ys=updated_actions)
kappa = torch.unsqueeze(kernel_dict["output"], dim=3)
```
  1. 核函数（`gaussian_kernel`）梯度：$\alpha \nabla _{a^\prime} \mathbf{k}(a_t, f^{\phi}(.;s_t))|_{a^\prime = a_t}$ 
  2. 最终$\triangle f^{\phi}(.;s_t)$
```python
action_gradients = torch.mean(kappa * grad_log_p + kernel_dict["gradient"], dim=1)
```
- 计算$\frac{\partial{f^{\phi}(\xi;s_t)}}{\partial \phi}$
```python
gradients = torch.autograd.grad(
            outputs=updated_actions,
            inputs=self.actor.parameters(),
            grad_outputs=action_gradients,
            create_graph=True
        )
```
- 计算$E_{\xi }[\triangle f^{\phi}(\xi;s_t) \frac{\partial{f^{\phi}(\xi;s_t)}}{\partial \phi}]$
```python
actor_loss = -sum(
    torch.sum(w * g.detach()) for w, g in zip(self.actor.parameters(), gradients)
)
```

**实际Code: Python-Torch**
```python
    def _svgd_update(self, state):
        act = self.actor(state, n_action_samples=self.kernel_n_particles)
        # SVGD requires computing two empirical expectations over actions
        # (see Appendix C1.1.). To that end, we first sample a single set of
        # actions, and later split them into two sets: `fixed_actions` are used
        # to evaluate the expectation indexed by `j` and `updated_actions`
        # the expectation indexed by `i`.
        n_updated_actions = int(self.kernel_n_particles * self.kernel_update_ratio)
        n_fixed_actions = self.kernel_n_particles - n_updated_actions
        fixed_actions, updated_actions = torch.split(
            act, 
            [n_fixed_actions, n_updated_actions], 
            dim=1
        )
        # eqution13: part_1-2 \nabla _{a^\prime} Q^\theta_{soft}(s_t, a^\prime)|_{a\prime=a_t}
        svgd_tar_values = self.critic(state[:, None, :], fixed_actions)
        # Target log-density. Q_soft in Equation 13:
        squash_corr = torch.sum(torch.log(1 - fixed_actions**2 + EPS), dim=-1)
        log_p = svgd_tar_values.squeeze(2) + squash_corr
        # 计算 log_p 对 fixed_actions 的梯度
        grad_log_p = torch.autograd.grad(
            log_p, fixed_actions, 
            grad_outputs=torch.ones_like(log_p), 
            create_graph=True
        )[0]
        grad_log_p = grad_log_p.unsqueeze(2).detach()
        # print(f'{grad_log_p.shape=} {grad_log_p.mean()=} {grad_log_p.min()=}')
        
        kernel_dict = self.kernel_fn(xs=fixed_actions, ys=updated_actions)
        # Kernel function in Equation 13:
        kappa = torch.unsqueeze(kernel_dict["output"], dim=3)
        # eqution13: part_1 kappa * grad_log_p part_2 k_gradient
        action_gradients = torch.mean(kappa * grad_log_p + kernel_dict["gradient"], dim=1)
        # Propagate the gradient through the policy network (Equation 14).
        # 计算 gradients
        gradients = torch.autograd.grad(
            outputs=updated_actions,
            inputs=self.actor.parameters(),
            grad_outputs=action_gradients,
            create_graph=True
        )
        # 计算 surrogate_loss
        actor_loss = -sum(
            torch.sum(w * g.detach()) for w, g in zip(self.actor.parameters(), gradients)
        )
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
```


完整实现[SoftQNew.py](../../src/RLAlgo/SoftQNew.py)


### 3. 试验

完整实现[test_softQ.py < sqlnew_Walker2d_v4_test > ](../../src/test/test_softQ.py)
```python
def sqlnew_Walker2d_v4_test():
    env_name = 'Walker2d-v4'
    gym_env_desc(env_name)
    env = gym.make(env_name)
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    cfg = Config(
        env, 
        num_episode=1500,
        save_path=os.path.join(path_, "test_models" ,f'SQL-{env_name}.ckpt'), 
        actor_hidden_layers_dim=[256, 256],
        critic_hidden_layers_dim=[256, 256],
        actor_lr=1.5e-4,
        critic_lr=5.5e-4,
        sample_size=256,
        off_buffer_size=204800*2,
        off_minimal_size=2048*2,
        max_episode_rewards=2048,
        max_episode_steps=800,
        gamma=0.99,
        SQL_kwargs=dict(
            value_n_particles=16,
            kernel_n_particles=16,
            kernel_update_ratio=0.5,
            critcic_traget_update_freq=100,
            reward_scale=1
        )
    )
    agent = SQLNew(
        state_dim=cfg.state_dim,
        action_dim=cfg.action_dim, 
        actor_hidden_layers_dim=cfg.actor_hidden_layers_dim,
        critic_hidden_layers_dim=cfg.critic_hidden_layers_dim,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        gamma=cfg.gamma,
        SQL_kwargs=cfg.SQL_kwargs,
        device=cfg.device
    )
    train_off_policy(
        env, agent, cfg, train_without_seed=True, wandb_flag=False, 
        wandb_project_name=f'SQL-{env_name}',
        test_episode_count=5,
        step_lr_flag=True, 
        step_lr_kwargs={'step_size': 1000, 'gamma': 0.9}
    )
    agent.actor.load_state_dict(
        torch.load(cfg.save_path, map_location='cpu')
    )
    agent.eval()
    cfg.max_episode_steps = 200
    env = gym.make(env_name, render_mode='human')
    play(env, agent, cfg, episode_count=2, play_without_seed=True, render=False)
```

![pic](sql_Walker2d-v4.png)

