import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class MlpNet(nn.Module):
    def __init__(self, obs_sp, act_sp, hidden_size=[64,64], hidden_activ=nn.ReLU, last_activ=None, lay_norm=False):
        super(MlpNet, self).__init__()
        self.input_size = obs_sp
        self.output_size = act_sp
        self.h_activ = hidden_activ
        self.last_activ = last_activ
        
        self.lay_norm = lay_norm
        in_size = hidden_size[-1] if len(hidden_size) > 0 else self.input_size
        
        self.feature_extractor = self._build_module(hidden_size)
        self.output_layer = nn.Linear(in_size, self.output_size)
        self.reset_parameters()
    
    def _build_module(self, h_size):
        in_size = self.input_size
        modules = []
        for n_units in h_size:
            modules.append(nn.Linear(in_size, n_units))
            modules.append(self.h_activ())
            if self.lay_norm:
                modules.append(nn.LayerNorm(n_units))
            in_size = n_units
        return nn.Sequential(*modules)
    
    def reset_parameters(self):
        for lay in self.feature_extractor:
            if isinstance(lay, nn.Linear):
                lay.weight.data.uniform_(*hidden_init(lay))
        self.output_layer.weight.data.uniform_(-3e-3, 3e-3)
         
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.output_layer(x)
        if self.last_activ is not None:
            x = self.last_activ(x)
        return x
    
class GumbelMlpNet(MlpNet):
    def __init__(self, obs_sp, act_sp, hidden_size=[64,64], hidden_activ=nn.ReLU, tau=1., lay_norm=False):
        super(GumbelMlpNet, self).__init__(obs_sp=obs_sp, act_sp=act_sp, hidden_size=[64,64], hidden_activ=nn.ReLU, lay_norm=lay_norm)
        self.tau = tau
        
    def forward(self, x):
        x = super().forward(x)
        x = F.gumbel_softmax(x, tau=self.tau, hard=False)
        return x
    
class ContinuousCritic(nn.Module):
    def __init__(self, obs_sp, act_sp, hidden_size=[64,64]):
        super(ContinuousCritic, self).__init__()
        if len(hidden_size)!=2:
            raise ValueError("'hidden_size' must be of length 2")
        self.action_space = act_sp
        self.obs_space = obs_sp

        self.linear1 = nn.Linear(obs_sp, hidden_size[0])
        self.ln1 = nn.LayerNorm(hidden_size[0])

        self.linear2 = nn.Linear(hidden_size[0] + self.action_space, hidden_size[1])
        self.ln2 = nn.LayerNorm(hidden_size[1])

        self.output_layer = nn.Linear(hidden_size[1], 1)

        self.reset_parameters()

    def reset_parameters(self):
        for lay in [self.linear1, self.linear2]:
            if isinstance(lay, nn.Linear):
                lay.weight.data.uniform_(*hidden_init(lay))
        self.output_layer.weight.data.uniform_(-3e-3, 3e-3)
         

    def forward(self, obs, act):
        x = obs

        # Extractor features obs
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)

        # Concat features obs and actions
        x = torch.cat((x, act), 1)  # Insert the actions
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)

        outputs = self.output_layer(x)
        return outputs