import numpy as np
import torch
import torch.nn as nn

from .model import Model

class VTable(Model):
    """
    The class of state value function for discret state space.
    
    :param n_observations: (int) The number of possible observations
    """
    def __init__(self, n_observations):
        self.n_obs = n_observations
        self.v_table = torch.zeros((self.n_obs), dtype=torch.float64)
        
    def save(self, filename):
        torch.save(self.v_table, filename)
        
    def load(self, filename):
        self.v_table = torch.load(filename)
    
    def __call__(self, state=None):
        if state is None:
            return self.v_table     
        else:
            return self.v_table[state]
        
    @property
    def shape(self):
        return tuple([self.n_obs])


class QTable(Model):
    """
    The class of action value function for discret state and action space.
    
    :param n_observations: (int) The number of possible observations
    :param n_actions: (int) The number of possible actions
    """
    def __init__(self, n_observations, n_actions):
        self.n_obs = n_observations
        self.n_actions = n_actions
        self.q_table = torch.zeros((self.n_obs, self.n_actions), dtype=torch.float64)
        
    def save(self, filename):
        torch.save(self.q_table, filename)
        
    def load(self, filename):
        self.q_table = torch.load(filename)
    
    def __call__(self, state=None, action=None):
        if action is None and state is None:
            return self.q_table     
        if action is None:
            return self.q_table[state, :]
        if state is None:
            return self.q_table[:, action]
        else:
            return self.q_table[state, action]
        
    @property
    def shape(self):
        return tuple([self.n_obs] + [self.n_actions])

class MultiQTable(Model):
    """
    The class of actions value function for multi-agent with discret state and action space.
    This kind of value function is used in minimax-Q algorithm.
    
    :param n_observations: (int) The number of possible observations
    :param n_actions: (int) The number of possible actions
    """
    def __init__(self, n_observations, n_actions):
        self.n_obs = n_observations
        self.n_actions = n_actions
        self.q_table = torch.zeros(tuple([self.n_obs] + self.n_actions), dtype=torch.float64)
    
    def save(self, filename):
        torch.save(self.q_table, filename)
        
    def load(self, filename):
        self.q_table = torch.load(filename)
        
    @property
    def shape(self):
        return tuple([self.n_obs] + self.n_actions)
    
    def __call__(self, state=None, action=None):
        if action is None and state is None:
            return self.q_table.min(2).values 
        if action is None:
            return self.q_table.min(2).values[state, :]
        if state is None:
            return self.q_table.min(2).values[:, action]
        else:
            return self.q_table.min(2).values[state, action]