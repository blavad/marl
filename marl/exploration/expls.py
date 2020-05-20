import math
import torch
import torch.nn as nn
from torch.nn.functional import softmax

from . import ExplorationProcess

class UCB1(ExplorationProcess):
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.count = [1 for _ in range(n_actions)]

    def reset(self, t=None):
        """ Reinitialize the state of the process """
        self.count = [1 for _ in range(self.n_actions)]         
        
    def update(self, t):
        pass
            
    def __call__(self, policy, observation):
        value = policy.model(observation) + torch.tensor([math.sqrt(2*math.log(100)/i) for i in self.count])
        action = value.argmax().item()
        self.count[action] += 1
        return action