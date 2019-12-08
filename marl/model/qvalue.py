import numpy as np
import torch
import torch.nn as nn

from .model import Model

class QTable(Model):
    def __init__(self, observation_space, action_space):
        self.n_actions = action_space.n
        self.n_obs = observation_space.n
        self.q_table = torch.zeros((self.n_obs, self.n_actions), dtype=torch.float64)
    
    def __call__(self, state=None, action=None):
        if torch.is_tensor(state) and state.dim()>0:
            state=state[0]
        if torch.is_tensor(action) and action.dim()>0:
            action=action[0]
        if action is None and state is None:
            return self.q_table     
        if action is None:
            return self.q_table[state, :].view(-1, self.n_actions)
        if state is None:
            return self.q_table[:, action].view(-1, self.n_obs)
        else:
            return self.q_table[state, action]