import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import marl
from .policy import Policy, ModelBasedPolicy
from marl.tools import gymSpace2dim

class RandomPolicy(Policy):
    """
    The class of random policies
    
    :param model: (Model or torch.nn.Module) The q-value model 
    :param action_space: (gym.Spaces) The action space
    """
    def __init__(self, action_space):
        self.action_space = action_space
        
    def __call__(self, state):
        """
        Return a random action given the state
        
        :param state: (Tensor) The current state
        """  
        return self.action_space.sample()
    

class QPolicy(ModelBasedPolicy):
    """
    The class of policies based on a Q function
    
    :param model: (Model or torch.nn.Module) The q-value model 
    :param observation_space: (gym.Spaces) The observation space
    :param action_space: (gym.Spaces) The action space
    """
    def __init__(self, model, observation_space=None, action_space=None):
        self.observation_space = observation_space
        self.action_space = action_space
        
        self.model = marl.model.make(model, obs_sp=gymSpace2dim(self.observation_space), act_sp=gymSpace2dim(self.action_space))
        
        
    def __call__(self, state):
        """
        Return an action given the state
        
        :param state: (Tensor) The current state
        """  
        if isinstance(self.Q, nn.Module):
            state = torch.tensor(state).float().unsqueeze(0)
            with torch.no_grad():
                return self.Q(state).max(1).indices.item()
        else:
            return self.Q(state).max(0).indices.item()
            

    @property
    def Q(self):
        return self.model


class StochasticPolicy(ModelBasedPolicy):
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
        
        obs_dim = gymSpace2dim(self.observation_space)
        act_dim = gymSpace2dim(self.action_space)
        self.model = marl.model.make(model, obs_sp=obs_dim, act_sp=act_dim)
        
    def forward(self, x):
        x = self.model(x)
        pd = Categorical(x)
        return pd

    def __call__(self, observation):
        observation = torch.tensor(observation).float()
        with torch.no_grad():
            pd = self.forward(observation)
            return pd.sample().item()
        
class DeterministicPolicy(ModelBasedPolicy):
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
        
        self.low = self.action_space.low[0] if isinstance(self.action_space, gym.spaces.Box) else 0
        self.high = self.action_space.high[0] if isinstance(self.action_space, gym.spaces.Box) else 1
        
        obs_dim = gymSpace2dim(self.observation_space)
        act_dim = gymSpace2dim(self.action_space)
        self.model = marl.model.make(model, obs_sp=obs_dim, act_sp=act_dim)

    def __call__(self, observation):
        observation = torch.tensor(observation).float()
        with torch.no_grad():
            action = np.array(self.model(observation))
            return np.clip(action, self.low, self.high)