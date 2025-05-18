# python3
# Author: Scc_hy
# Create Date: 2025-04-29
# Func: ICM
# paper: https://pathak22.github.io/noreward-rl/
# reference: https://github.com/bonniesjli/icm/blob/master/model.py
#            https://github.com/Stepan-Makarenko/ICM-PPO-implementation/blob/master/ICM.py
# ============================================================================
from  torch import nn 
import numpy as np
import torch 
import torch.nn.functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * F.sigmoid(x)


def normal_init(layer, std_ = np.sqrt(1.0 / 2)):
    nn.init.normal_(layer.weight, mean=0.0, std=std_)


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
        self.action_dim = action_dim
        self.cnn_encoder_feature = nn.Sequential(
            nn.Conv2d(channel_dim, channel_dim * 8, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(channel_dim * 8, channel_dim * 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(channel_dim * 16, channel_dim * 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        cnn_out_dim = self._get_cnn_out_dim()
        self.cnn_encoder_header = nn.Sequential(
            nn.Linear(cnn_out_dim, 512),
            # nn.ReLU()
        )
        # 离散动作
        actim_emb_dim = 4
        self.action_emb = nn.Embedding(self.action_dim, actim_emb_dim)
        self.forward_model = nn.Sequential(
            nn.Linear(512 + actim_emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
        )
        self.inverse_model = nn.Sequential(
               nn.Linear(512 + 512, 256),
               nn.ReLU(),
               nn.Linear(256, 256),
               nn.ReLU(),
               nn.Linear(256, action_dim)
        )
        self.__init()

    @torch.no_grad
    def _get_cnn_out_dim(self):
        pic = torch.randn((1, self.channel_dim, self.state_dim, self.state_dim))
        return self.cnn_encoder_feature(pic).shape[1]  
    
    def encode_pred(self, state):
        return self.cnn_encoder_header(self.cnn_encoder_feature(state))
    
    def forward_pred(self, phi_s, action):
        return self.forward_model(torch.concat([phi_s, self.action_emb(action)], dim=1))

    def inverse_pred(self, phi_s, phi_s_next):
        return self.inverse_model(torch.concat([phi_s, phi_s_next], dim=1))

    def forward(self, state, n_state, action, mask):
        # 离散动作
        action = action.type(torch.LongTensor).reshape(-1).to(state.device)
        # encode
        phi_s = self.encode_pred(state)
        phi_s_next = self.encode_pred(n_state)

        # forward  不用于训练Encoder
        hat_phi_s_next = self.forward_pred(phi_s, action) # self.forward_pred(phi_s.detach(), action)
        # intrinisc reward & forward_loss  
        r_i = 0.5 * nn.MSELoss(reduction='none')(hat_phi_s_next, phi_s_next) # phi_s_next.detach())
        r_i = r_i.mean(dim=1) * mask 
        forward_loss = r_i.mean()
        
        # inverse 同时用于训练Encoder
        hat_a = self.inverse_pred(phi_s, phi_s_next) # self.inverse_pred(phi_s.detach(), phi_s_next)
        # inverse loss 
        inv_loss = (nn.CrossEntropyLoss(reduction='none')(hat_a, action) * mask).mean()
        return r_i, inv_loss, forward_loss

    def __init(self):
        # cnn
        for layer in self.cnn_encoder_feature:
            classname = layer.__class__.__name__
            if (classname.find('Conv2d') != -1) and ('Pool' not in classname):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)

