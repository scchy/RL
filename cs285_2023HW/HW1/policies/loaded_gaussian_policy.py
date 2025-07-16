# python3 
# Create Date: 2025-07-16
# ===============================================================
import numpy as np
from .base_policy import BasePolicy
from .utils import from_numpy, to_numpy
from torch import nn
import torch
import pickle


def create_linear_layer(W, b) -> nn.Linear:
    out_features, in_features = W.shape
    linear_layer = nn.Linear(
        in_features,
        out_features,
    )
    linear_layer.weight.data = from_numpy(W.T)
    linear_layer.bias.data = from_numpy(b[0])
    return linear_layer


def read_layer(l):
    assert list(l.keys()) == ['AffineLayer']
    assert sorted(l['AffineLayer'].keys()) == ['W', 'b']
    return l['AffineLayer']['W'].astype(np.float32), l['AffineLayer'][
        'b'].astype(np.float32)


class LoadedGaussianPolicy(BasePolicy, nn.Module):
    def __init__(self, filename, **kwargs):
        super(LoadedGaussianPolicy, self).__init__(**kwargs)
        # todo 改成模型结构读取参数
        with open(filename, 'rb') as f:
            data = pickle.loads(f.read())
        
        self.nonlin_type = data['nonlin_type']
        if self.nonlin_type == 'lrelu':
            self.non_lin = nn.LeakyReLU(0.01)
        elif self.nonlin_type == 'tanh':
            self.non_lin = nn.Tanh()
        else:
            raise NotImplementedError()

        policy_type = [k for k in data.keys() if k != 'nonlin_type'][0]
        assert policy_type == 'GaussianPolicy', (
            'Policy type {} not supported'.format(policy_type)
        )
        self.policy_params = data[policy_type]
        assert set(self.policy_params.keys()) == {
            'logstdevs_1_Da', 'hidden', 'obsnorm', 'out'
        }
        # Build the policy. First, observation normalization.
        assert list(self.policy_params['obsnorm'].keys()) == ['Standardizer']
        obsnorm_mean = self.policy_params['obsnorm']['Standardizer']['mean_1_D']
        obsnorm_meansq = self.policy_params['obsnorm']['Standardizer']['meansq_1_D']
        obsnorm_stdev = np.sqrt(np.maximum(0, obsnorm_meansq - np.square(obsnorm_mean)))
        self.obs_dim = obsnorm_mean.shape[-1]
        print('obs', obsnorm_mean.shape, obsnorm_stdev.shape)
        self.obs_norm_mean = nn.Parameter(from_numpy(obsnorm_mean))
        self.obs_norm_std = nn.Parameter(from_numpy(obsnorm_stdev))
        self.hidden_layers = nn.ModuleList()
        # Hidden layers next
        assert list(self.policy_params['hidden'].keys()) == ['FeedforwardNet']
        layer_params = self.policy_params['hidden']['FeedforwardNet']
        for layer_name in sorted(layer_params.keys()):
            l = layer_params[layer_name]
            W, b = read_layer(l)
            linear_layer = create_linear_layer(W, b)
            self.hidden_layers.append(linear_layer)

        # Output layer
        W, b = read_layer(self.policy_params['out'])
        self.output_layer = create_linear_layer(W, b)
        
    def forward(self, obs):
        # Truncate or pad the observation if it does not match.
        if self.obs_dim != obs.shape[-1]:
            n_pad = self.obs_dim - obs.shape[-1]
            if n_pad > 0:
                obs = nn.functional.pad(obs, (0, n_pad), 'constant')
            else:
                obs = obs[:, :self.obs_dim]
        normed_obs = (obs - self.obs_norm_mean) / (self.obs_norm_std + 1e-6)
        h = normed_obs
        for layer in self.hidden_layers:
            h = layer(h)
            h = self.non_lin(h)
        return self.output_layer(h)


    def update(self, obs_no, acs_na, adv_n=None, acs_labels_na=None):
        raise NotImplementedError("""
            This policy class simply loads in a particular type of policy and
            queries it. Do not try to train it.
        """)

    def get_action(self, obs):
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None, :]
        
        observation = from_numpy(observation.astype(np.float32))
        action = self(observation)
        return to_numpy(action)

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)


    