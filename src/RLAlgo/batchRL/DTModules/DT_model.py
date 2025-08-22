# python3
# Create Date: 2025-08-19
# Reference: https://github.com/kzl/decision-transformer/blob/master/gym/decision_transformer/models/decision_transformer.py
# ==============================================================================================

import numpy as np 
import torch 
from torch import nn 
import transformers
from transformers import GPT2Model as orgGPT2Model
from transformers import GPT2Config, DecisionTransformerGPT2Model, DecisionTransformerModel as orgDecisionTransformer
try:
    from .trajectory_gpt2 import GPT2Model
except Exception as e:
    import sys
    import os 
    path_ = os.path.dirname(__file__) 
    if path_ not in sys.path:
        sys.path.append(path_) 
    from trajectory_gpt2 import GPT2Model
# from transformers import GPT2LMHeadModel, GPT2PreTrainedModel


class TrajectoryModel(nn.Module):
    def __init__(self, state_dim, act_dim, max_length=None):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length
        self.register_parameter(name="obs_mean", param=nn.Parameter(torch.tensor(0.0, dtype=torch.float), requires_grad=False))
        self.register_parameter(name="obs_var", param=nn.Parameter(torch.tensor(0.0, dtype=torch.float), requires_grad=False))

    def forward(self, state, action, rtg, masks=None, attention_mask=None):
        # "masked" tokens or unspecified inputs can be passed in as None
        return None, None, None

    def get_action(self, states, actions, rewards, **kwargs):
        # these will come as tensors on the correct device
        return torch.zeros_like(actions[-1])



class DecisionTransformer(TrajectoryModel):
    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """
    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)
        self.hidden_size = hidden_size 
        self.max_length = max_length
        print(f"{kwargs=}")
        config = GPT2Config(
            vocab_size=1, # doesnt matter _ we dont use the word embedding 
            n_embd=hidden_size,
            **kwargs    
        )
        config.n_ctx = max_ep_len
        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll  add those ourselves)
        self.transformer = GPT2Model(config)
        self.hidden_size = hidden_size
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_state = nn.Linear(self.state_dim, hidden_size)
        self.embed_action = nn.Linear(self.act_dim, hidden_size)
        
        self.embed_ln = nn.LayerNorm(hidden_size)
        
        # note: we [dont] predict [states or returns] for the paper
        self.predict_state = nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = nn.Linear(hidden_size, 1)

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]
        # seq_length for stack frames
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long).to(states.device)
        
        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        # print(f'{returns_to_go.shape=}') # batch//K, K, 1
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        #  org-stack: [B//K, K, dim] -> [B//K, 3, K, dim]  ->  [B//K, K, 3, dim] -> [B//K, 3*K, dim]
        #  my-concat: [B//K, K, dim] -> [B//K, 3*K, dim]
        # stacked_inputs = torch.stack(
        #     (returns_embeddings, state_embeddings, action_embeddings),
        #     dim=1
        # ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.hidden_size)
        stacked_inputs = torch.concat(
            (returns_embeddings, state_embeddings, action_embeddings),
            dim=1
        )
        stacked_inputs = self.embed_ln(stacked_inputs)
        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        
        # We feed in the input embeddings 
        tf_out = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            position_ids=torch.zeros(stacked_attention_mask.shape, device=stacked_inputs.device, dtype=torch.long),
        )
        x = tf_out['last_hidden_state']
        
        # reshape x so that the second dimension corresponds to the original 
        # returns (0), states (1), or actions (2); i.e. x[:, 1, t] is token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
        
        # get predictions
        return_preds = self.predict_return(x[:,2])  # predict next return given state and action
        state_preds = self.predict_state(x[:,2])    # predict next state given state and action
        action_preds = self.predict_action(x[:,1])  # predict next action given state

        return state_preds, action_preds, return_preds

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # we don't care about the past rewards in this model
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)
        
        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None
        _, action_preds, return_preds = self.forward(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)

        return action_preds[0,-1]


if __name__ == '__main__':
    # compare
    kwargs = {}
    hidden_size = 24
    config = GPT2Config(
        vocab_size=1,  # doesn't matter -- we don't use the vocab
        n_embd=hidden_size,
        hidden_size=hidden_size,
        n_ctx=12,
        max_ep_len=1000,
        state_dim=17,
        act_dim=7,
        action_tanh=True,
        **kwargs
    )

    # note: the only difference between this GPT2Model and the default Huggingface version
    # is that the positional embeddings are removed (since we'll add those ourselves)
    transformer = orgGPT2Model(config)
    transformer2 = GPT2Model(config)
    dt_tf = orgDecisionTransformer(config)
    dt = DecisionTransformer(
            state_dim=config.state_dim,
            act_dim=config.act_dim,
            hidden_size=config.n_embd,
            max_ep_len=config.max_ep_len,
            action_tanh=True
    )
    print("orgGPT2Model=", transformer )
    print("GPT2Model=", transformer2 )
    print('--'*25)
    print("orgDecisionTransformer=", dt_tf )
    print("DecisionTransformer=", dt )

