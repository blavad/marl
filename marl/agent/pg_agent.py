from . import TrainableAgent

class PGAgent(TrainableAgent):
    
    def __init__(self):
        pass
    
    def process_experience(self, obs, act, rew, new_obs, done, terminal):
        raise NotImplementedError
        
    def update_model(self):
        raise NotImplementedError
        
    def action(self, observation):
        raise NotImplementedError
        
    def save_model(self, save_name):
        raise NotImplementedError
        
    def load_model(self, save_name):
        raise NotImplementedError