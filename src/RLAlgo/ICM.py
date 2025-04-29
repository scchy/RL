# python3
# Author: Scc_hy
# Create Date: 2025-04-29
# Func: ICM
# paper: https://pathak22.github.io/noreward-rl/
# ============================================================================
from  torch import nn 
import torch 


class cnnICM(nn.Module):
    def __init__(
        self, 
        channel_dim,
        state_dim, 
        action_dim
    ):
        super(cnnICM, self).__init__()
        self.state_dim = state_dim
        self.channel_dim = channel_dim
        self.cnn_encoder_feature = nn.Sequential(
            nn.Conv2d(channel_dim, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        cnn_out_dim = self._get_cnn_out_dim()
        self.cnn_encoder_header = nn.Sequential(
            nn.Linear(cnn_out_dim, 512),
            nn.ReLU()
        )
        
        self.forward_model = nn.Sequential(
            nn.Linear(512 + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
        )
        self.inverse_model = nn.Sequential(
               nn.Linear(512 + 512, 256),
               nn.ReLU(),
               nn.Linear(256, num_actions)
        )
    
    @torch.no_grad
    def _get_cnn_out_dim(self):
        pic = torch.randn((1, self.channel_dim, self.state_dim, self.state_dim))
        return self.cnn_encoder_feature(pic).shape[1]  
    
    def encode_pred(self, state):
        return self.cnn_encoder_header(self.cnn_encoder_feature(state))
    
    def forward_pred(self, phi_s, action):
        return self.forward_model(phi_s, action)
    
    def inverse_pred(self, phi_s, phi_s_next):
        return self.inverse_model(phi_s, phi_s_next)
    
    def forward(self, state, n_state, action, mask):
        # encode
        phi_s = self.encode_pred(state)
        phi_s_next = self.encode_pred(n_state)

        # forward  不用于训练Encoder
        hat_phi_s_next = self.forward_pred(phi_s.detach(), action)
        # intrinisc reward & forward_loss  
        r_i = 0.5 * nn.MSELoss(reduction='none')(hat_phi_s_next, phi_s_next.detach())
        r_i = r_i.mean(dim=2) * mask 
        forward_loss = r_i.mean()
        
        # inverse 同时用于训练Encoder
        hat_a = self.inverse_pred(phi_s.detach(), phi_s_next)
        # inverse loss 
        inv_loss = (nn.CrossEntropyLoss(reduction='none')(hat_a, action) * mask).mean()
        return r_i, inv_loss, forward_loss

