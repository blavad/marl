from marl.exploration import ExplorationProcess
from marl.exploration.eps_greedy import EpsGreedy
from marl.policy.policy import Policy
from marl.policy.policies import *

import torch
import torch.optim as optim
import copy 

class Agent(object):
    def __init__(self, policy, name="UnknownAgent"):
        self.observation_space = None
        self.action_space = None
        self.policy = Policy.make(policy)
        self.name = name
        
    def load_model(self, save_name):
        self.policy.load(save_name)
    
    def action(self, observation):
        return self.policy(observation)
        
class TrainableAgent(Agent):
    
    agents = {}
    
    def __init__(self, policy, experience_buffer, exploration_process, loss, optimizer, gamma=0.99, lr=0.001, beta=None , name="TrainableAgent"):
        super(TrainableAgent, self).__init__(policy)
        self.experience_buffer = experience_buffer
        self.exploration_process = ExplorationProcess.make(exploration_process)
        self.gamma = gamma
        self.lr = lr
        
        self.optimizer = optimizer(self.policy.parameters(), lr=self.lr)
        self.loss = loss
    
    def store_experience(self, *args, **kwargs):
        self.experience_buffer.store(*args, **kwargs)
        
    def update_model(self):
        self.policy.update()
        
    def action(self, observation):
        return self.exploration_process(self.policy, observation)
        
    def save_policy(self, save_name):
        self.policy.save(save_name)
        
    def save_all(self):
        pass
        
    @classmethod
    def make(cls, agent_name, *args, **kwargs):
        return TrainableAgent.agents[agent_name](*args,**kwargs)
    
    @classmethod
    def register(cls, agent_name, agent_cl):
        TrainableAgent.agents[agent_name] = agent_cl

    @classmethod
    def available(cls):
        return TrainableAgent.agents.keys()
    
class QAgent(TrainableAgent):
    def __init__(self, q_value, experience_buffer, exploration_process, gamma=0.99, lr=0.1, dueling=True, off_policy=True, name="QAgent"):
        super(QAgent, self).__init__(QPolicy(q_value), experience_buffer, exploration_process, gamma, lr, name=name)
        self.off_policy = off_policy
        self.dueling = dueling and self.off_policy
        if self.off_policy:
            self.local_policy = copy.deepcopy(self.policy)
        
    def update_model(self):
        self.optimizer.zero_grad()
        loss = self._calculate_loss()
        loss.backward()
        self.optimizer.step()  
    
    def _calculate_loss(self):
        o, a, r, o1 = self.experience_buffer.sample()
        output = self.policy.Q(o, a)
        target = self._calculate_target(o, a, r, o1)
        return self.loss(output, target)
       
    def _calculate_target(self, obs, action, rew, obs1):
        if self.off_policy:
            if self.dueling:
                amax = self.policy(obs1)
                return rew + self.gamma * self.policy.Q(obs1, amax)            
            else:
                return rew + self.gamma * torch.max(self.local_policy.Q(obs1))[0]
        else:            
            return rew + self.gamma * torch.max(self.policy.Q(obs1))[0]
    
    
    
class DQNAgent(QAgent):
    def __init__(self, q_net, experience_buffer, exploration_process="EpsGreedy", gamma=0.99, lr=0.00025, dueling=True, name="DQNAgent"):
        super(DQNAgent, self).__init__(QPolicy(QApprox(q_net)), experience_buffer, exploration_process, gamma, lr, dueling=dueling, off_policy=True, name=name)
        

class PGAgent(TrainableAgent):
    def __init__(self):
        super(PGAgent, self).__init__(policy, experience_buffer, exploration_process, gamma, lr, dueling=dueling, off_policy=True, name=name)
        
    
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
    
class ActorCriticAgent(MARL):
    def __init__(self, actor_agent, crtic_agent):
        super(MARL, self).__init__([crtic_agent, actor_agent], None)
        self.actor = self.agents[0]
        self.actor = self.agents[1]
        
    def store_experience(self, *args, **kwargs):
        for ag in self.agents:
            ag.store_experience(args, kwargs)
        
    def action(self, observation):
        return [ag.action(obs) for ag, obs in zip(self.agents, observation)]
        
    def save_model(self, save_name):
        for ag in self.agents:
            ag.save_model("{}-{}".format(save_name, ag.name))
        
    def load_model(self, save_name):
        for ag in self.agents:
            ag.load_model(save_name)