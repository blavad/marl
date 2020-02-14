import marl
from . import TrainableAgent, MATrainable
from ..policy import QPolicy
from ..model import MultiQTable
from marl.tools import gymSpace2dim 

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class QAgent(TrainableAgent):
    """
    The class of trainable agent using Qvalue-based methods
    
    :param qmodel: (Model or torch.nn.Module) The q-value model 
    :param observation_space: (gym.Spaces) The observation space
    :param action_space: (gym.Spaces) The action space
    :param experience: (Experience) The experience memory data structure
    :param exploration: (Exploration) The exploration process 
    :param gamma: (float) The training parameters
    :param lr: (float) The learning rate
    :param batch_size: (int) The size of a batch
    :param target_update_freq: (int) The update frequency of the target model  
    :param name: (str) The name of the agent      
    """
    def __init__(self, qmodel, observation_space, action_space, experience="ReplayMemory-1", exploration="EpsGreedy", gamma=0.99, lr=0.1, batch_size=1, target_update_freq=None, name="QAgent"):
        super(QAgent, self).__init__(policy=QPolicy(model=qmodel, observation_space=observation_space, action_space=action_space), observation_space=observation_space, action_space=action_space, experience=experience, exploration=exploration, gamma=gamma, lr=lr, batch_size=batch_size, name=name)
        
        self.off_policy = target_update_freq is not None
        self.target_update_freq = target_update_freq
        
        if self.off_policy:
            self.target_policy = copy.deepcopy(self.policy)
        
    def update_model(self, t):
        """
        Update the model.
        
        :param t: (int) The current timestep
        """
        if len(self.experience) < self.batch_size:
            return
        
        # Get changing policy
        if self.off_policy:
            curr_policy = self.target_policy
        else:
            curr_policy = self.policy
        
        # Get batch of experience
        if isinstance(self, MATrainable):
            ind = self.experience.sample_index(self.batch_size)
            batch = self.mas.experience.get_transition(len(self.mas.experience) - np.array(ind)-1)
        else:
            batch = self.experience.sample(self.batch_size)
        
        # print(batch)
        
        # Compute target r_t + gamma*max_a Q(s_t+1, a)
        target_value = self.target(curr_policy.Q, batch)
        
        # Compute current value Q(s_t, a_t)
        curr_value = self.value(batch.observation, batch.action)
        
        # print("Target : ", target_value, " - Current : ", curr_value)
        
        # Update Q values
        self.update_q(curr_value, target_value, batch)
        
        if self.off_policy and t % self.target_update_freq==0:
            self.update_target_model()
    
    def target(self, Q, batch):
        """
        Compute the target value.
        
        :param Q: (Model or torch.nn.Module) The model of the Q-value
        :param batch: (list) A list composed of the different information about the batch required
        """
        raise NotImplementedError
    
    def value(self, observation, action):
        """
        Compute the value.
        
        :param observation: The observation
        :param action: The action
        """
        raise NotImplementedError
    
    def update_q(self, curr_value, target_value, batch):
        """
        Update the Q value.
        
        :param curr_value: (torch.Tensor) The current value 
        :param target_value: (torch.Tensor) The target value
        :param batch: (list) A list composed of the different information about the batch required
        """
        raise NotImplementedError    
    
    def update_target_model(self):
        """
        Update the target model.
        """
        raise NotImplementedError
            
class QTableAgent(QAgent):
    """
    The class of trainable agent using  Q-table to model the  function Q 
    
    :param observation_space: (gym.Spaces) The observation space
    :param action_space: (gym.Spaces) The action space
    :param exploration: (Exploration) The exploration process 
    :param gamma: (float) The training parameters
    :param lr: (float) The learning rate
    :param target_update_freq: (int) The update frequency of the target model  
    :param name: (str) The name of the agent      
    """
    
    def __init__(self, observation_space, action_space, exploration="EpsGreedy", gamma=0.99, lr=0.1, target_update_freq=None, name="QTableAgent"):
        super(QTableAgent, self).__init__(qmodel="QTable", observation_space=observation_space, action_space=action_space, experience="ReplayMemory-1", exploration=exploration, gamma=gamma, lr=lr, batch_size=1, target_update_freq=target_update_freq, name=name)
        
    def update_q(self, curr_value, target_value, batch):
        self.policy.Q.q_table[batch.observation, batch.action] = curr_value + self.lr * (target_value - curr_value)
        
    def update_target_model(self):
        self.target_policy = copy.deepcopy(self.policy)
        
    def target(self, Q, batch):
        next_obs  = batch.next_observation
        next_action_value = Q(next_obs).max(1).values.float()
        rew = torch.from_numpy(batch.reward).float()
        not_dones = torch.from_numpy(1.-batch.done_flag).float()
        
        target_value = rew + not_dones * self.gamma * next_action_value
        return target_value
        
    def value(self, observation, action):
        return self.policy.Q(observation, action)
    
class MinimaxQAgent(QAgent, MATrainable):
    """
    The class of trainable agent using  minimax-Q-table algorithm 
    
    :param observation_space: (gym.Spaces) The observation space
    :param my_action_space: (gym.Spaces) My action space
    :param other_action_space: (gym.Spaces) The action space of the other agent
    :param index: (int) The position of the agent in the list of agent
    :param mas: (marl.agent.MAS) The multi-agent system corresponding to the agent
    :param exploration: (Exploration) The exploration process 
    :param gamma: (float) The training parameters
    :param lr: (float) The learning rate
    :param target_update_freq: (int) The update frequency of the target model  
    :param name: (str) The name of the agent      
    """
    
    def __init__(self, observation_space, my_action_space, other_action_space, index=None, mas=None, exploration="EpsGreedy", gamma=0.99, lr=0.1, target_update_freq=None, name="MinimaxQAgent"):
        QAgent.__init__(self, qmodel=MultiQTable(gymSpace2dim(observation_space), [gymSpace2dim(my_action_space), gymSpace2dim(other_action_space)]), observation_space=observation_space, action_space=my_action_space, experience="ReplayMemory-1", exploration=exploration, gamma=gamma, lr=lr, batch_size=1, target_update_freq=target_update_freq, name=name)
        MATrainable.__init__(self, mas, index)
        
    def update_q(self, curr_value, target_value, batch):
        if len(batch.action[0]) > 2:
            raise Exception("The number of agents should not exceed 2.")
        self.policy.Q.q_table[batch.observation[0][self.index], batch.action[0][self.index], batch.action[0][1-self.index]]  = curr_value + self.lr * (target_value - curr_value)
        
    def update_target_model(self):
        self.target_policy = copy.deepcopy(self.policy)
        
    def target(self, Q, joint_batch):
        next_obs  = joint_batch.next_observation.squeeze(0)[self.index]
        next_value = Q(next_obs).max()
        rew = torch.from_numpy(joint_batch.reward).squeeze(0)[self.index].float()
        not_dones = torch.from_numpy(1.-joint_batch.done_flag).squeeze(0)[self.index].float()
        target_value = rew + not_dones * self.gamma * next_value
        return target_value
        
    def value(self, observation, action):
        return self.policy.Q.q_table[observation[0][self.index], action[0][self.index], action[0][1-self.index]]
    
    
class MinimaxDQNAgent(QAgent, MATrainable):
    """
    The class of trainable agent using  minimax-DQN algorithm 
    
    :param observation_space: (gym.Spaces) The observation space
    :param my_action_space: (gym.Spaces) My action space
    :param other_action_space: (gym.Spaces) The action space of the other agent
    :param index: (int) The position of the agent in the list of agent
    :param mas: (marl.agent.MAS) The multi-agent system corresponding to the agent
    :param exploration: (Exploration) The exploration process 
    :param gamma: (float) The training parameters
    :param lr: (float) The learning rate
    :param target_update_freq: (int) The update frequency of the target model  
    :param name: (str) The name of the agent      
    """
    
    def __init__(self, qmodel, observation_space, my_action_space, other_action_space, index=None, mas=None, exploration="EpsGreedy", experience="ReplayMemory-10000", gamma=0.99, lr=0.001, batch_size=32, target_update_freq=None, name="MinimaxDQNAgent"):
        QAgent.__init__(self, qmodel=qmodel, observation_space=observation_space, action_space=my_action_space, experience=experience, exploration=exploration, gamma=gamma, lr=lr, batch_size=batch_size, target_update_freq=target_update_freq, name=name)
        MATrainable.__init__(self, mas, index)
        self.criterion = nn.SmoothL1Loss() # Huber criterion
        self.optimizer = optim.Adam(self.policy.Q.parameters(), lr=self.lr)
        if self.off_policy:
            self.target_policy.Q.eval()
        
        
    def update_q(self, curr_value, target_value, batch):
        if len(batch.action[0]) > 2:
            raise Exception("The number of agents should not exceed 2.")
        self.optimizer.zero_grad()
        loss = self.criterion(curr_value, target_value)
        loss.backward()
        self.optimizer.step()
        
    def update_target_model(self):
        self.target_policy.Q.load_state_dict(self.policy.Q.state_dict())
        
    def target(self, Q, joint_batch):
        next_obs  = torch.from_numpy(joint_batch.next_observation)[self.index].float()
        next_value = Q(next_obs).max()
        rew = torch.from_numpy(joint_batch.reward)[self.index].float()
        not_dones = torch.from_numpy(1.-joint_batch.done_flag)[self.index].float()
        target_value = (rew + not_dones * self.gamma * next_value).unsqueeze(1)
        return target_value.detach()
        
    def value(self, observation, action):
        t_action = torch.from_numpy(action).long().unsqueeze(1)
        t_observation = torch.from_numpy(observation).float()
        return self.policy.Q(t_observation).gather(1, t_action)
        
        # return self.policy.Q.q_table[observation[0][self.index], action[0][self.index], action[0][1-self.index]]


    
class DQNAgent(QAgent):
    """
    The class of trainable agent using a neural network to model the  function Q
    
    :param qmodel: (Model or torch.nn.Module) The q-value model 
    :param observation_space: (gym.Spaces) The observation space
    :param action_space: (gym.Spaces) The action space
    :param experience: (Experience) The experience memory data structure
    :param exploration: (Exploration) The exploration process 
    :param gamma: (float) The training parameters
    :param lr: (float) The learning rate
    :param batch_size: (int) The size of a batch
    :param target_update_freq: (int) The update frequency of the target model  
    :param name: (str) The name of the agent      
    """
    
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
    
    def target(self, Q, batch):
        next_obs  = torch.from_numpy(batch.next_observation).float()
        next_action_values = Q(next_obs).max(1).values.float()
        rew = torch.from_numpy(batch.reward).float()
        not_dones = torch.from_numpy(1.-batch.done_flag).float()
        target_value = (rew + not_dones * self.gamma * next_action_values).unsqueeze(1)
        return target_value.detach()
        
    def value(self, observation, action):
        t_action = torch.from_numpy(action).long().unsqueeze(1)
        t_observation = torch.from_numpy(observation).float()
        return self.policy.Q(t_observation).gather(1, t_action)
        