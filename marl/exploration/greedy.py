from .eps_greedy import EpsGreedy

class Greedy(EpsGreedy):
    """
    The Greedy process
    
    :param eps_deb: (float) The intial amount of exploration 
    :param eps_fin: (float) The final amount of exploration 
    """        
    def __init__(self):
        super(Greedy, self).__init__(eps_deb=0.0, eps_fin=0.0)
        
    def __call__(self, policy, observation):
        return policy(observation)
        
        