import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class MlpNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=[64,64], hidden_activ=nn.ReLU, last_activ=None):
        super(MlpNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.h_activ = hidden_activ
        self.last_activ = last_activ
        
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
    def __init__(self, input_size, output_size, hidden_size=[64,64], hidden_activ=nn.ReLU, tau=1.):
        super(GumbelMlpNet, self).__init__(input_size=input_size, output_size=output_size, hidden_size=[64,64], hidden_activ=nn.ReLU)
        self.tau = tau
        
    def forward(self, x):
        x = super().forward(x)
        x = F.gumbel_softmax(x, tau=self.tau, hard=False)
        return x
    

    
############# En Cours ##############
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    def __init__(self, state_shape, action_size, seed, fc1_units=256, fc2_units = 128):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_shape, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x

class Critic(nn.Module):
    def __init__(self, full_obs_shape, output_shape, seed, fc1_units=256, fc2_units=128):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(full_obs_shape, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, m_obs, o_obs, m_actions, o_actions):
        full_ob = torch.cat((m_obs, o_obs, m_actions, o_actions), dim = 1)
        x = F.leaky_relu(self.fc1(full_ob))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return x