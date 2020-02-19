import marl
from marl.agent import TrainableAgent, DQNAgent, QTableAgent, MATrainable
from marl.experience import ReplayMemory
from marl.policy import StochasticPolicy

import torch
import torch.optim as optim
import numpy as np

import copy

class PGAgent(TrainableAgent):
    """
    The class of generic trainable agent using policy-based methods
    
    :param critic: (QAgent) The critic agent 
    :param actor_policy: (Policy) The policy for the actor
    :param actor_model: (Model or nn.Module) The model for the actor
    :param observation_space: (gym.Spaces) The observation space
    :param action_space: (gym.Spaces) The action space
    :param experience: (Experience) The experience memory data structure
    :param exploration: (Exploration) The exploration process 
    :param lr_actor: (float) The learning rate for the actor
    :param lr_critic: (float) The learning rate for the critic
    :param gamma: (float) The training parameters
    :param batch_size: (int) The size of a batch
    :param target_update_freq: (int) The update frequency of the target model  
    :param name: (str) The name of the agent      
    """
    def __init__(self, critic, actor_policy, observation_space, action_space, actor_model=None, experience="ReplayMemory-1000", exploration="EpsGreedy", lr_actor=0.001, gamma=0.95, batch_size=32, target_update_freq=None, name="PGAgent"):
        TrainableAgent.__init__(self, policy=actor_policy, model=actor_model, observation_space=observation_space, action_space=action_space, experience=experience, exploration=exploration, lr=lr_actor, gamma=gamma, batch_size=batch_size, name=name)
        
        self.critic = critic

        # Init target networks
        self.use_target_net = target_update_freq is not None
        self.target_update_freq = target_update_freq
        
        if self.use_target_net:
            self.target_policy = copy.deepcopy(self.policy)
            self.target_policy.model.eval()

    @property
    def lr_actor(self):
        return self.lr
    
    
    @property
    def lr_critic(self):
        return self.critic.lr
    
    def update_model(self, t):
        """
        Update the model.
        
        :param t: (int) The current timestep
        """
        if len(self.experience) < self.batch_size:
            return
        
        # Get batch of experience
        if isinstance(self, MATrainable):
            ind = self.experience.sample_index(self.batch_size)
            batch = self.mas.experience.get_transition(len(self.mas.experience) - np.array(ind)-1)
        else:
            batch = self.experience.sample(self.batch_size)
        
        # Update the critic model
        self.critic.update_model(t)
        # Update the actor model
        self.update_actor(batch)
        
        if self.use_target_net and t % self.target_update_freq==0:
            self.update_target_policy()

    def update_target_policy(self):
        """
        Update the target policy.
        """
        raise NotImplementedError
    
    def update_actor(self, batch):
        """
        Update the actor.
        """
        raise NotImplementedError
    

class DeepACAgent(PGAgent):
    """
    Deep Actor-Critic Agent class. The critic is train following DQN algorithm and the policy is represented by a neural network with a softmax output.
    
    :param critic_model: (nn.Module) The critic's model 
    :param actor_model: (Model or nn.Module) The model for the actor
    :param observation_space: (gym.Spaces) The observation space
    :param action_space: (gym.Spaces) The action space
    :param experience: (Experience) The experience memory data structure
    :param exploration: (Exploration) The exploration process 
    :param lr_actor: (float) The learning rate for the actor
    :param lr_critic: (float) The learning rate for the critic
    :param gamma: (float) The training parameters
    :param batch_size: (int) The size of a batch
    :param target_update_freq: (int) The update frequency of the target model  
    :param name: (str) The name of the agent      
    """
    def __init__(self, critic_model, actor_model, observation_space, action_space, experience="ReplayMemory-1000", exploration="EpsGreedy", lr_actor=0.001, lr_critic=0.001, gamma=0.95, batch_size=32, target_update_freq=None, name="DeepACAgent"):
        PGAgent.__init__(self, critic=None, actor_policy="StochasticPolicy", actor_model=actor_model, observation_space=observation_space, action_space=action_space, experience=experience, exploration=exploration, lr_actor=lr_actor, gamma=gamma, batch_size=batch_size, name=name)

        self.critic = DQNAgent(qmodel=critic_model, observation_space=observation_space, action_space=action_space, experience=self.experience, gamma=self.gamma, lr=lr_critic, batch_size=self.batch_size, target_update_freq=self.target_update_freq)
        self.actor_optimizer = optim.Adam(self.policy.model.parameters(), lr=self.lr)

    def update_target_policy(self):
        self.target_policy.model.load_state_dict(self.policy.model.state_dict())

    def update_actor(self, batch):
        obs = torch.from_numpy(batch.observation).float()
        act = torch.from_numpy(batch.action)
        
        # Calcul actor loss
        self.actor_optimizer.zero_grad()
        pd = self.policy.forward(obs)
        log_prob = pd.log_prob(act)
        gae = self.critic.policy.Q(obs).gather(1,act.unsqueeze(1)).detach()
        
        actor_loss = -(log_prob * gae).mean()
        
        actor_loss.backward()
        self.actor_optimizer.step()
    

class PHC(PGAgent):
    """
    Policy Hill Climbing Agent's class. The critic is train following standard Q-learning algorithm.
    
    :param observation_space: (gym.Spaces) The observation space
    :param action_space: (gym.Spaces) The action space
    :param exploration: (Exploration) The exploration process 
    :param delta: (float) The learning rate for the actor
    :param lr_critic: (float) The learning rate for the critic
    :param gamma: (float) The training parameters
    :param target_update_freq: (int) The update frequency of the target model  
    :param name: (str) The name of the agent      
    """
    def __init__(self, observation_space, action_space, exploration="EpsGreedy", delta=0.01, lr_critic=0.01, gamma=0.95, target_update_freq=None, name="PHCAgent"):
        PGAgent.__init__(self, critic=None, actor_policy="StochasticPolicy", actor_model="ActionProb", observation_space=observation_space, action_space=action_space, experience="ReplayMemory-1", exploration=exploration, lr_actor=delta, gamma=gamma, batch_size=1, name=name)

        self.critic = QTableAgent(observation_space=observation_space, action_space=action_space, gamma=self.gamma, lr=lr_critic, target_update_freq=self.target_update_freq)
        self.critic.experience = self.experience

    def update_target_policy(self):
        self.target_policy = copy.deepcopy(self.policy)
        
    @property
    def delta(self):
        return self.lr_actor

    def update_actor(self, batch):
        n_a = self.policy.model.n_actions
        obs = batch.observation
        max_a = self.critic.greedy_action(obs[0]) 
        
        delta_sa = torch.Tensor([min(self.policy.model(obs, a), self.delta/(n_a-1)) for a in range(n_a)])
        delta_sa[max_a] = 0.
        for a in range(n_a):
            self.policy.model.value[obs,a] = max(self.policy.model.value[obs,a] - delta_sa[a],0)
        self.policy.model.value[obs,max_a] += sum(delta_sa).item()
        
# class DDPGAgent(TrainableAgent):
    
#     def __init__(self, critic_policy, actor_policy, experience_buffer, lr_critic = 0.01, lr_actor = 0.01, tau = 0.01, gamma = 0.95, num_sub_policy=2, capacity_buff = 1.e6, seed=None):
#         self.lr_critic = lr_critic
#         self.lr_actor = lr_actor
#         self.momentum = 0.95
#         self.tau = tau
#         self.num_sub_policy = num_sub_policy
#         self.env = env
        
#         self.critic = DQNAgent(critic_policy, ReplayMemory(capacity_buff), "EpsGreedy")
#         self.actor = PGAgent(actor_policy, ReplayMemory(capacity_buff), "EpsGreedy")
        
#         self.agents = [self.critic, self.actor]
        
#     def soft_update(self, local_model, target_model, tau):
#         for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
#             target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)        

#     def update_model(self, t):
#         self.critic.update_model(t)
#         self.actor.update_model(t)
#         self.soft_update(self.critic.local_policy, self.critic.policy, self.tau)