from . import ExplorationProcess
import random
import numpy as np
import copy

class OUNoise(ExplorationProcess):
    """
    The epsilon-greedy exploration class
    
    :param eps_deb: (float) The initial amount of exploration to process
    :param eps_fin: (float) The final amount of exploration to process
    :param deb_expl: (flaot) The percentage of time before starting exploration (default: 0.1)
    :param deb_expl: (flaot) The percentage of time before starting exploration (default: 0.1)
    """
    
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
    
    def update(self, t):
        if t > self.init_expl_step:
            self.eps = max(self.eps_fin, self.eps - (self.eps_deb-self.eps_fin)/(self.final_expl_step-self.init_expl_step))
        
    def __call__(self, policy, observation):
        if random.random() < self.eps:
            return policy.action_space.sample()
        else :
            return policy(observation)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state