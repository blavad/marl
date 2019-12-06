from .agent import TrainableAgent

class MARL(TrainableAgent):
    
    def __init__(self, agents, env, name='marl'):
        self.agents = agents
        self.env = env
        self.name = name
        
    def store_experience(self, *args, **kwargs):
        for ag in self.agents:
            ag.store_experience(args, kwargs)
            
    def update_model(self, t):
        for ag in self.agents:
            if isinstance(ag, TrainableAgent):
                ag.update_model(t)
    
    def update_exploration(self, t):
        for ag in self.agents:
            if isinstance(ag, TrainableAgent):
                ag.exploration_process.update(t)
        
    def action(self, observation):
        return [ag.action(obs) for ag, obs in zip(self.agents, observation)]
        
    def save_model(self, save_name):
        for ag in self.agents:
            if isinstance(ag, TrainableAgent):
                ag.save_model("{}-{}".format(save_name, ag.name))
        
    def load_model(self, save_name):
        for ag in self.agents:
            if isinstance(ag, TrainableAgent):
                ag.load_model(save_name)
                