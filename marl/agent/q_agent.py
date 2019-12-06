from . import TrainableAgent
from ..policy import QPolicy
from ..model import QApprox

import copy
import torch
import torch.nn as nn
import torch.optim as optim

class QAgent(TrainableAgent):
    def __init__(self, q_value, obs_space, action_space, experience, exploration_process, gamma=0.99, lr=0.1, target_update_freq=10000, dueling=True, off_policy=True, name="QAgent"):
        super(QAgent, self).__init__(policy=QPolicy(q_value), observation_space=obs_space, action_space=action_space, experience=experience, exploration_process=exploration_process, gamma=gamma, lr=lr, name=name)
        self.off_policy = off_policy
        self.dueling = dueling and self.off_policy
        self.target_update_freq = target_update_freq
        
        if self.off_policy:
            self.local_policy = copy.deepcopy(self.policy)
        
    def update_model(self, t):
        batch = self.experience.sample()
        target = self._calculate_target(batch.obs, batch.action, batch.reward, batch.next_obs)
        self.policy.Q.q_table[batch.obs, batch.action] = (1.-self.lr)*self.policy.Q(batch.obs, batch.action)+self.lr * target
        if t % self.target_update_freq == 0:
            self.update_target()
       
    def _calculate_target(self, obs, action, reward, next_obs, done_flag=0):
        if self.off_policy:
            if self.dueling:
                amax = self.policy(next_obs)
                return reward + (1-done_flag)* self.gamma * self.policy.Q(next_obs, amax)            
            else:
                return reward + (1-done_flag)* self.gamma * torch.max(torch.tensor(self.local_policy.Q(next_obs)))[0]
        else:            
            return reward + (1-done_flag)* self.gamma * torch.max(torch.tensor(self.policy.Q(next_obs)))[0]
        
    def update_target(self):
        if self.off_policy:
            self.policy = copy.deepcopy(self.local_policy)
    
class DQNAgent(QAgent):
    def __init__(self, q_net, obs_space, action_space, experience, exploration_process="EpsGreedy", gamma=0.99, lr=0.00025,  batch_size=32, target_update_freq=10000, dueling=True, name="DQNAgent"):
        super(DQNAgent, self).__init__(q_value=QApprox(q_net), observation_space=obs_space, action_space=action_space, experience=experience, exploration_process=exploration_process, gamme=gamma, lr=lr, target_update_freq=target_update_freq, dueling=dueling, off_policy=True, name=name)
        self.batch_size = batch_size
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.local_policy.parameters(), lr=self.lr)

    def update_model(self, t):
        self.optimizer.zero_grad()
        batch = self.experience.sample(self.batch_size)
        target = self._calculate_target(batch.obs, batch.action, batch.reward, batch.next_obs, batch.done_flag)
        output = self.policy.Q(batch.obs, batch.action)
        loss = self.loss(output, target)
        loss.backward()
        self.optimizer.step()
        
        if t % self.target_update_freq == 0:
            self.update_target()
    