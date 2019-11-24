from . import ExplorationProcess

class Greedy(ExplorationProcess):        
    
    def __call__(self, policy, observation):
        return policy(observation)
    
    def __str__(self):
        return "Greedy Exploration"