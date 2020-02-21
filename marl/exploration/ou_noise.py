from . import ExplorationProcess
import random
import numpy as np
import copy

class OUNoise(ExplorationProcess):
    """
    The Ornstein-Uhlenbeck process.
    
    :param size: (float) The number of variables to add noise
    :param seed: (float) The seed
    :param mu: (float) The drift term 
    :param theta: (float) The amount of keeping previous state
    :param sigma: (float) The amount of noise
    """
    
    def __init__(self, size, dt=0.01, mu=0., theta=0.15, sigma=0.2):
        self.size = size
        self.dt = dt
        self.mu = mu 
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self, t=None):
        """ Reinitialize the state of the process """
        self.state =  self.mu * np.ones(self.size)
        
    def update(self, t):
        self.reset()
        
    def __call__(self, policy, observation):
        return np.clip(policy(observation) + self.sample(), policy.low, policy.high)
        
    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.random.randn(len(x)) * np.sqrt(self.dt)
        self.state = x + dx
        return self.state