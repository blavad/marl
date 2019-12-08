import marl
from . import TrainableAgent
from ..policy import QPolicy

import copy
import torch
import torch.nn as nn
import torch.optim as optim

class QAgent(TrainableAgent):
    def __init__(self, qmodel, observation_space, action_space, experience="ReplayMemory-1", exploration="EpsGreedy", gamma=0.99, lr=0.1, batch_size=1, target_update_freq=None, name="QAgent"):
        super(QAgent, self).__init__(policy=QPolicy(model=qmodel, observation_space=observation_space, action_space=action_space), observation_space=observation_space, action_space=action_space, experience=experience, exploration=exploration, gamma=gamma, lr=lr, batch_size=batch_size, name=name)
        
        self.off_policy = target_update_freq is not None
        self.target_update_freq = target_update_freq
        
        if self.off_policy:
            self.target_policy = copy.deepcopy(self.policy)
        
    def update_model(self, t):
        if len(self.experience) < self.batch_size:
            return
        
        # Get changing policy
        if self.off_policy:
            curr_policy = self.target_policy
        else:
            curr_policy = self.policy
        
        # Get batch of experience
        batch = self.experience.sample(self.batch_size)
        
        # Compute target r_t + gamma*max_a Q(s_t+1, a)
        target_value = self.target(curr_policy.Q, batch).detach()
        # Compute current value Q(s_t, a_t)
        curr_value = self.value(batch.observation, batch.action)
        
        # print("Target : ", target_value, " - Current : ", curr_value)
        
        # Update Q values
        self.update_q(curr_value, target_value, batch)
        
        if self.off_policy and t % self.target_update_freq==0:
            self.update_target_model()
       
    def target(self, Q, batch):
        next_action_values = Q(batch.next_observation).max(1).values
        return (batch.reward + (1-batch.done_flag)* self.gamma * next_action_values).unsqueeze(1)
    
    def value(self, observation, action):
        raise NotImplementedError
    
    def update_q(self, curr_value, target_value, batch):
        raise NotImplementedError    
    
    def update_target_model(self):
        raise NotImplementedError
            
class QTableAgent(QAgent):
    def __init__(self, observation_space, action_space, exploration="EpsGreedy", gamma=0.99, lr=0.1, target_update_freq=None, name="QTableAgent"):
        super(QTableAgent, self).__init__(qmodel="QTable", observation_space=observation_space, action_space=action_space, experience="ReplayMemory-1", exploration=exploration, gamma=gamma, lr=lr, batch_size=1, target_update_freq=target_update_freq, name=name)
        
    def update_q(self, curr_value, target_value, batch):
        self.policy.Q.q_table[batch.observation, batch.action] = (1.-self.lr)*curr_value + self.lr * target_value
        
    def update_target_model(self):
        self.target_policy = copy.deepcopy(self.policy)
        
    def value(self, observation, action):
        return self.policy.Q(observation, action)
    
class DQNAgent(QAgent):
    def __init__(self, qmodel, observation_space, action_space, experience="ReplayMemory-10000", exploration="EpsGreedy", gamma=0.99, lr=0.0005,  batch_size=32, target_update_freq=1000, name="DQNAgent"):
        super(DQNAgent, self).__init__(qmodel=qmodel, observation_space=observation_space, action_space=action_space, experience=experience, exploration=exploration, gamma=gamma, lr=lr, batch_size=batch_size, target_update_freq=target_update_freq, name=name)
        self.criterion = nn.SmoothL1Loss() # Huber criterion
        self.optimizer = optim.Adam(self.policy.Q.parameters(), lr=self.lr)
        if self.off_policy:
            self.target_policy.Q.eval()
            
    def update_q(self, curr_value, target_value, batch):
        self.optimizer.zero_grad()
        loss = self.criterion(curr_value, target_value)
        loss.backward()
        self.optimizer.step()
        
    def update_target_model(self):
        self.target_policy.Q.load_state_dict(self.policy.Q.state_dict())
        
    def value(self, observation, action):
        action = action.long().unsqueeze(1)
        return self.policy.Q(observation).gather(1, action)
        