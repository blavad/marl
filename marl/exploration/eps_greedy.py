from . import ExplorationProcess
import random

class EpsGreedy(ExplorationProcess):
    def __init__(self, eps_deb=1.0, eps_fin=0.1):
        self.eps_deb = eps_deb
        self.eps_fin = eps_fin
        self.eps = self.eps_deb
        
    def __call__(self, policy, observation):
        if random.random() < self.eps:
            return 
        else :
            return policy(observation)
        
    def init(self):
        self.eps = self.eps_deb
        
    def __str__(self):
        return "{}-greedy exploration".format(self.eps)