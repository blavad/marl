import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .policy import Policy
from marl.model import QApprox, QTable

class QPolicy(Policy):
    def __init__(self, model, observation_space=None, action_space=None):
        self.observation_space = observation_space
        self.action_space = action_space
            
        self.q_value = marl.model.make(model, self.observation_space, self.action_space)
       
    @property 
    def Q(self):
        return self.q_value
        
    def __call__(self, state):
        return torch.max(self.Q(state), 1)[1]
    
    def load(self, filename):
        self.Q.load(filename)

    def save(self, filename):
        self.Q.save(filename)

class PolicyApprox(nn.Module, Policy):
    def __init__(self, model, observation_space=None, action_space=None):
        super(PolicyApprox, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        
        self.model = marl.model.make(model, self.observation_space, self.action_space)
        
    def forward(self, x):
        x = self.model(x)
        x = Categorical(F.softmax(x, dim=-1))
        return x

    def __call__(self, state):
        pd = self.forward(state)
        return pd.sample()
    
    def load(self, filename):
        nn.Module.load(self, filename)

    def save(self, filename):
        nn.Module.save(self, filename)

    
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