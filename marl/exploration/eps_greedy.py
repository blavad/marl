from . import ExplorationProcess
import random

class EpsGreedy(ExplorationProcess):
    """
    The epsilon-greedy exploration class
    
    :param eps_deb: (float) The initial amount of exploration to process
    :param eps_fin: (float) The final amount of exploration to process
    :param deb_expl: (float) The percentage of time before starting exploration (default: 0.1)
    :param deb_expl: (float) The percentage of time before starting exploration (default: 0.1)
    """
    def __init__(self, eps_deb=1.0, eps_fin=0.1, deb_expl=0.1, fin_expl=0.9):
        self.eps_deb = eps_deb
        self.eps_fin = eps_fin
        self.eps = self.eps_deb
        if fin_expl < deb_expl:
            raise ValueError("'deb_expl' must be lower than 'fin_expl'")
        self.deb_expl = deb_expl
        self.fin_expl = fin_expl
    
    def reset(self, training_duration):
        """ Reinitialize some parameters  """
        self.eps = self.eps_deb
        self.init_expl_step = int(self.deb_expl * training_duration)
        self.final_expl_step = int(self.fin_expl * training_duration)
    
    def update(self, t):
        """ Update epsilon linearly """   
        if t > self.init_expl_step:
            self.eps = max(self.eps_fin, self.eps_deb - (t-self.init_expl_step)*(self.eps_deb-self.eps_fin)/(self.final_expl_step-self.init_expl_step))
        
    def __call__(self, policy, observation):
        """ Choose an action according to the policy and the exploration rate """   
        if random.random() < self.eps:
            return policy.action_space.sample()
        else :
            return policy(observation)