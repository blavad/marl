from marl.exploration import ExplorationProcess
from marl.exploration.eps_greedy import EpsGreedy

from marl.policy.policy import Policy

class Agent(object):
    def __init__(self, policy):
        self.policy = Policy.make(policy)
        
    def load_model(self, save_name):
        self.policy.load(save_name)
        
    def action(self, observation):
        return self.policy(observation)
        
class TrainableAgent(Agent):
    
    agents = {}
    
    def __init__(self, policy, experience_buffer, exploration_process):
        super(TrainableAgent, self).__init__(policy)
        self.experience_buffer = experience_buffer
        self.exploration_process = ExplorationProcess.make(exploration_process)
    
    def store_experience(self, *args, **kwargs):
        self.experience_buffer.store(args, kwargs)
        
    def update_model(self):
        self.policy.update()
        
    def action(self, observation):
        return self.exploration_process(self.policy, observation)
        
    def save_model(self, save_name):
        self.policy.save(save_name)
        
    @classmethod
    def make(cls, agent_name, *args, **kwargs):
        return TrainableAgent.agents[agent_name](*args,**kwargs)
    
    @classmethod
    def register(cls, agent_name, agent_cl):
        TrainableAgent.agents[agent_name] = agent_cl

    @classmethod
    def available(cls):
        return TrainableAgent.agents.keys()
    
class OnPolicyAgent(TrainableAgent):
    def __init__(self, policy, experience_buffer, exploration_process):
        super(OnPolicyAgent, self).__init__(policy, experience_buffer, exploration_process)
        
class OffPolicyAgent(TrainableAgent):
    def __init__(self, policy, experience_buffer, exploration_process):
        super(OffPolicyAgent, self).__init__(policy, experience_buffer, exploration_process)
        self.local_policy = Policy.make(policy)
    
    def update_model(self):
        self.local_policy.update()
        