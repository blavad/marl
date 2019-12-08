import torch.optim as optim

from marl import MARL
from marl.experience import ReplayMemory

from . import DQNAgent, PGAgent
from . import TrainableAgent

class DDPGAgent(MARL):
    
    def __init__(self, critic_policy, actor_policy, env, experience_buffer, lr_critic = 0.01, lr_actor = 0.01, tau = 0.01, gamma = 0.95, num_sub_policy=2, capacity_buff = 1.e6, seed=None):
        self.lr_critic = lr_critic
        self.lr_actor = lr_actor
        self.momentum = 0.95
        self.tau = tau
        self.num_sub_policy = num_sub_policy
        self.env = env
        
        self.critic = DQNAgent(critic_policy, self.env, ReplayMemory(capacity_buff), "EpsGreedy")
        self.actor = PGAgent(actor_policy, self.env, ReplayMemory(capacity_buff), "EpsGreedy")
        
        self.agents = [self.critic, self.actor]
        
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)        

    def update_model(self):
        self.critic.update_model()
        self.actor.update_model()
        self.soft_update(self.critic.local_policy, self.critic.policy, self.tau)
        
    def action(self, observation):
        raise NotImplementedError
        
    def save_model(self, save_name):
        raise NotImplementedError
        
    def load_model(self, save_name):
        raise NotImplementedError