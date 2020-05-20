import math
import torch
import torch.nn as nn
from torch.nn.functional import softmax

from . import ExplorationProcess

class UCB1(ExplorationProcess):
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.count = [1 for _ in range(n_actions)]
        self.t = sum(self.count)

    def reset(self, t=None):
        """ Reinitialize the state of the process """
        self.count = [1 for _ in range(self.n_actions)]         
        self.t = sum(self.count)
        
    def update(self, t):
        self.t = self.t + 1
            
    def __call__(self, policy, observation):
        q_value = policy.model(observation)
        ucb1_value = q_value + q_value.max().item() * torch.tensor([math.sqrt(2*math.log(self.t)/i) for i in self.count])
        action = ucb1_value.argmax().item()
        self.count[action] += 1
        return action