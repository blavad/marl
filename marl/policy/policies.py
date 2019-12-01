import torch
import torch.nn as nn
import torch.nn.functional as F

from .policy import Policy
import numpy as np

class QValue(object):        
    def __call__(self, state, action):
        raise NotImplementedError

class QTable(QValue):
    def __init__(self, n_action, n_state):
        self.n_action = n_action
        self.n_state = n_state
        self.q_table = np.zeros((self.n_state, self.n_action))
    
    def __call__(self, state=None, action=None):
        if action is None and state is None:
            return self.q_table
        if action is None:
            return self.q_table[state, :]
        if state is None:
            return self.q_table[:, action]
        else:
            return self.q_table[state, action]

class QApprox(QValue, nn.Module):
    def __init__(self, net):
        super(QApprox, self).__init__()
        self.net = net
    
    def forward(self, x):
        x = self.net(x)
        return x

    def load_policy(self, save_name):
        self.load(save_name)
    
    def __call__(self, state, action=None):
        if action is None:
            return self.forward(state)
        else:
           return self.forward(state)[action]

class QPolicy(Policy):
    def __init__(self, q_value):
        self.q_value = q_value
       
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
    def __init__(self, net):
        super(PolicyApprox, self).__init__()
        self.net = net
        
    def forward(self, x):
        x = self.net(x)
        return x

    def __call__(self, state):
        return self.forward(state)
    
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