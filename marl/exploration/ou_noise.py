from . import ExplorationProcess
import random
import numpy as np
import copy

class OUNoise(ExplorationProcess):
    """
    The Ornstein-Uhlenbeck process.
    
    :param size: (float) The initial amount of exploration to process
    :param seed: (float) The final amount of exploration to process
    :param mu: (float) The percentage of time before starting exploration (default: 0.1)
    :param theta: (float) The percentage of time before starting exploration (default: 0.1)
    :param sigma: (float) The percentage of time before starting exploration (default: 0.1)
    """
    
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self, t=None):
        """ Reinitialize the state of the process """
        self.state = copy.copy(self.mu)
        
    def __call__(self, policy, observation):
        return policy(observation) + self.sample()
        
    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state