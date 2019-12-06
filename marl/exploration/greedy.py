from .eps_greedy import EpsGreedy

class Greedy(EpsGreedy):        
    def __init__(self):
        super(Greedy, self).__init__(eps_deb=0.0, eps_fin=0.0)
        
    def __call__(self, policy, observation):
        return policy(observation)
        
        