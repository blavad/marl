from .agent import TrainableAgent, Agent

class MARL(TrainableAgent):
    
    def __init__(self, agents_list, name='marl'):
        self.agents = agents_list
        self.name = name
        
    def store_experience(self, *args):
        TrainableAgent.store_experience(self, *args)
        observation, action, reward, next_observation, done = args
        for i, ag in enumerate(self.agents):
            if isinstance(ag, TrainableAgent):
                ag.store_experience(observation[i], action[i], reward[i], next_observation[i], done[i])
            
    def update_model(self, t):
        for ag in self.agents:
                ag.update_model(t)
    
    def update_exploration(self, t):
        TrainableAgent.update_exploration(self, t)        
        for ag in self.agents:
            if isinstance(ag, TrainableAgent):
                ag.exploration.update(t)
        
    def action(self, observation):
        return [ag.action(obs) for ag, obs in zip(self.agents, observation)]
        
    def greedy_action(self, observation):
        return [Agent.action(ag, observation) for ag, obs in zip(self.agents, observation)]
    
    def save_model(self, save_name):
        for ag in self.agents:
            if isinstance(ag, TrainableAgent):
                ag.save_model("{}-{}".format(save_name, ag.name))
        
    def load_model(self, save_name):
        for ag in self.agents:
            if isinstance(ag, TrainableAgent):
                ag.load_model(save_name)
                
    def append(self, agent):
        self.agents.append(agent)
        
    def get_by_name(self, name):
        for ag in self.agents:
            if ag.name == name:
                return ag
        return None