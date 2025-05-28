# python3
# Author: Scc_hy
# Create Date: 2025-05-26
# Func: A2C
# ============================================================================




class A2C:
    """ 
    actor:  \pi(a|s, \theta)
    critic-value_net: V(s)
    policyLoss = \frac{\partial log(\pi(a|s, \theta))}{\partial \theta} A
    Advantage: A = r+\gamma V(s_{t+1}) - V(s_t)
    """
    def __init__(
        self,
        state_dim: int,
        actor_hidden_layers_dim: typ.List[int],
        critic_hidden_layers_dim: typ.List[int],
        action_dim: int,
        actor_lr: float,
        critic_lr: float,
        gamma: float,
        A2C_kwargs: typ.Dict,
        device: torch.device   
    ):
        self.state_dim = state_dim
        self.actor_hidden_layers_dim = actor_hidden_layers_dim
        self.critic_hidden_layers_dim = critic_hidden_layers_dim
        self.action_dim = action_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.device = device


        
        
        
        