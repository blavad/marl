import numpy as np
import torch
import torch.nn as nn

from .model import Model

class VTable(Model):
    """
    The class of state value function for discret state space.
    
    :param obs_sp: (int) The number of possible observations
    """
    def __init__(self, obs_sp):
        self.n_obs = obs_sp
        self.value = torch.zeros((self.n_obs), dtype=torch.float64)
    
    def __call__(self, state=None):
        if state is None:
            return self.value     
        else:
            return self.value[state]
        
    @property
    def shape(self):
        return tuple([self.n_obs])


class QTable(Model):
    """
    The class of action value function for discret state and action space.
    
    :param obs_sp: (int) The number of possible observations
    :param act_sp: (int) The number of possible actions
    """
    def __init__(self, obs_sp, act_sp):
        self.n_obs = obs_sp
        self.n_actions = act_sp
        self.value = torch.zeros((self.n_obs, self.n_actions), dtype=torch.float64)
    
    @property
    def q_table(self):
        return self.value
    
    def __call__(self, state=None, action=None):
        if action is None and state is None:
            return self.value     
        if action is None:
            return self.value[state, :]
        if state is None:
            return self.value[:, action]
        else:
            return self.value[state, action]
        
    @property
    def shape(self):
        return tuple([self.n_obs] + [self.n_actions])

class MultiQTable(Model):
    """
    The class of actions value function for multi-agent with discret state and action space.
    This kind of value function is used in minimax-Q algorithm.
    
    :param obs_sp: (int) The number of possible observations
    :param act_sp: (int) The number of possible actions
    """
    def __init__(self, obs_sp, act_sp):
        self.n_obs = obs_sp
        self.n_actions = act_sp
        self.value = torch.zeros(tuple([self.n_obs] + self.n_actions), dtype=torch.float64)
    
    @property
    def q_table(self):
        return self.value
        
    @property
    def shape(self):
        return tuple([self.n_obs] + self.n_actions)
    
    def __call__(self, state=None, action=None):
        if action is None and state is None:
            return self.value.min(2).values 
        if action is None:
            return self.value.min(2).values[state, :]
        if state is None:
            return self.value.min(2).values[:, action]
        else:
            return self.value.min(2).values[state, action]
        
class ActionProb(Model):
    """
    The class of action probabilities for PHC algorithm.
    
    :param obs_sp: (int) The number of possible observations
    :param act_sp: (int) The number of possible actions
    """
    def __init__(self, obs_sp, act_sp):
        self.n_obs = obs_sp
        self.n_actions = act_sp
        self.value = torch.ones((self.n_obs, self.n_actions), dtype=torch.float64) * (1./self.n_actions)
    
    def __call__(self, state=None, action=None):
        if state is not None and torch.is_tensor(state):
            if state.dim() ==0:
                state=int(state.item())
            else:
                state =  list(state.numpy().astype(int))
        if action is not None and torch.is_tensor(action):
            if state.dim() ==0:
                action=int(action.item())
            else:
                action =  list(action.numpy().astype(int))
        if action is None and state is None:
            return self.value     
        if action is None:
            return self.value[state, :]
        if state is None:
            return self.value[:, action]
        else:
            return self.value[state, action] 
        
    @property
    def shape(self):
        return tuple([self.act_prob])
