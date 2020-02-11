import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import marl
from .policy import Policy
from marl.tools import gymSpace2dim

class QPolicy(Policy):
    """
    The class of policies based on a Q function
    
    :param model: (Model or torch.nn.Module) The q-value model 
    :param observation_space: (gym.Spaces) The observation space
    :param action_space: (gym.Spaces) The action space
    """
    def __init__(self, model, observation_space=None, action_space=None):
        self.observation_space = observation_space
        self.action_space = action_space
            
        self.Q = marl.model.make(model, input_size=gymSpace2dim(self.observation_space), output_size=gymSpace2dim(self.action_space))
        
    def __call__(self, state):
        """
        Return an action given the state
        
        :param state: The current state
        """
    
        state = torch.tensor(state).float().unsqueeze(0)
        with torch.no_grad():
            return self.Q(state).max(1).indices.item()

    @property
    def model(self):
        return self.Q

class StochasticPolicy(Policy):
    """
    The class of stochastic policies
    
    :param model: (Model or torch.nn.Module) The model of the policy 
    :param observation_space: (gym.Spaces) The observation space
    :param action_space: (gym.Spaces) The action space
    """
    
    def __init__(self, model, observation_space=None, action_space=None):
        super(StochasticPolicy, self).__init__()
        
        self.observation_space = observation_space
        self.action_space = action_space
        
        input_dim = gymSpace2dim(self.observation_space)[0]
        output_dim = gymSpace2dim(self.action_space)
        self.model = marl.model.make(model, input_size=input_dim, output_size=output_dim)
        
    def forward(self, x):
        x = self.model(x)
        return Categorical(x)

    def __call__(self, observation):
        observation = torch.tensor(observation).float()
        with torch.no_grad():
            pd = self.forward(observation)
            return pd.sample()
        
class DeterministicPolicy(Policy):
    """
    The class of deterministic policies
    
    :param model: (Model or torch.nn.Module) The model of the policy
    :param observation_space: (gym.Spaces) The observation space
    :param action_space: (gym.Spaces) The action space
    """
    
    def __init__(self, model, observation_space=None, action_space=None):
        super(DeterministicPolicy, self).__init__()
        
        self.observation_space = observation_space
        self.action_space = action_space
        
        input_dim = gymSpace2dim(self.observation_space)[0]
        output_dim = gymSpace2dim(self.action_space)
        self.model = marl.model.make(model, input_size=input_dim, output_size=output_dim)

    def __call__(self, observation):
        observation = torch.tensor(observation).float()
        with torch.no_grad():
            action = np.array(self.model(observation))
            return np.clip(action, 0.1, 0.5)