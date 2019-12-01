from . import TrainableAgent

class MADDPGAgent(TrainableAgent):
    
    def __init__(self):
        pass
    
    def store_experience(self,*args, **kwargs):
        raise NotImplementedError()
        
    def update_model(self):
        raise NotImplementedError()
        
    def action(self, observation):
        raise NotImplementedError()
        
    def save_model(self, save_name):
        raise NotImplementedError()
        
    def load_model(self, save_name):
        raise NotImplementedError()