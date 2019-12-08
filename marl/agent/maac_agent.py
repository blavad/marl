import marl
from marl import MARL
from marl.agent import QAgent, TrainableAgent
from marl.policy import PolicyApprox, DeterministicPolicyApprox

import torch
import torch.nn as nn
import torch.optim as optim
import copy


class MAPGAgent(TrainableAgent):
    def __init__(self, critic_model, actor_policy, observation_space, action_space, index, marl=None, experience="ReplayMemory-1000", exploration="EpsGreedy", lr_actor=0.001, lr_critic=0.001, gamma=0.95, batch_size=32, tau=0.01, use_target_net=False, name="MAACAgent"):
        super(MAPGAgent, self).__init__(policy=actor_policy, observation_space=observation_space, action_space=action_space, experience=experience, exploration=exploration, lr=lr_actor, gamma=gamma, batch_size=batch_size, name=name)
        
        self.tau = tau
        self.index = index
        self.marl = marl
        
        # Actor model
        self.actor_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        # Critic model
        self.critic_model = marl.model.make(critic_model)
        self.critic_criterion = nn.MSELoss() # Huber criterion
        self.lr_critic = lr_critic
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=self.lr_critic)
        
        # Init target networks
        self.use_target_net = use_target_net
        if self.use_target_net:
            self.target_critic = copy.deepcopy(self.critic_model)
            self.target_critic.eval()
            
            # !!! Unused part
            self.target_policy = copy.deepcopy(self.policy)
            self.target_policy.eval()

            
    def set_marl(self, marl):
        self.marl = marl
    
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def update_model(self, t):
        # Get batches
        ind = self.experience.sample_index(self.batch_size)
        self.global_batch = self.marl.experience.get_transition(ind)
        self.local_batch = self.experience.get_transition(ind)
    
        # Get changing policy
        self.curr_critic = self.target_critic if self.use_target_net else self.critic_model
        
        self.update_critic()
        self.update_actor()
        
        # Compute 
        if self.use_target_net:
            self.soft_update(self.policy.model, self.target_policy.model, self.tau)
            self.soft_update(self.critic_model, self.target_critic, self.tau)
            
    def target(self, model, batch):
        next_actions = self.marl.greedy_action(batch.next_observation)
        inputs = torch.cat((batch.next_observation, next_actions), dim=1)
        next_action_value = model(inputs)
        return (batch.reward + (1-batch.done_flag)* self.gamma * next_action_value)
    
    def update_critic(self):
        self.critic_optimizer.zero_grad()
        # Calcul critic loss
        target_value = self.target(self.curr_critic, self.global_batch).detach()
        curr_value = self.critic_model(self.global_batch.observation, self.global_batch.action)
        loss = self.critic_criterion.loss(curr_value, target_value)
        
        # Update params
        loss.backward()
        self.critic_optimizer.step()
        
    def update_actor(self):
        self.actor_optimizer.zero_grad()
        # Calcul actor loss
        pd = self.policy.forward(self.local_batch.observation)
        log_prob = pd.log_prob(self.local_batch.action) #.unsqueeze(0)
        gae = self.critic_model(self.global_batch.observation, self.global_batch.action).detach()
        actor_loss = -(log_prob * gae).mean()
        actor_loss.backward()
        self.actor_optimizer.step()
        
class MAACAgent(MAPGAgent):
    def __init__(self, critic_model, actor_model, observation_space, action_space, index, marl=None, experience="ReplayMemory-1000", exploration="EpsGreedy", lr_actor=0.001, lr_critic=0.001, gamma=0.95, batch_size=32, tau=0.01, use_target_net=False, name="MAACAgent"):
        super(MAACAgent, self).__init__(policy=PolicyApprox(actor_model), observation_space=observation_space, action_space=action_space, experience=experience, exploration=exploration, lr=lr_actor, gamma=gamma, batch_size=batch_size, name=name)

class MADDPGAgent(MAPGAgent):
    def __init__(self, critic_model, actor_model, observation_space, action_space, index, marl=None, experience="ReplayMemory-1000", exploration="EpsGreedy", lr_actor=0.001, lr_critic=0.001, gamma=0.95, batch_size=32, tau=0.01, use_target_net=False, name="MAACAgent"):
        super(MADDPGAgent, self).__init__(policy=DeterministicPolicyApprox(actor_model), observation_space=observation_space, action_space=action_space, experience=experience, exploration=exploration, lr=lr_actor, gamma=gamma, batch_size=batch_size, name=name)
    
    def update_actor(self):
        self.actor_optimizer.zero_grad()
        # Calcul actor loss
        my_action_pred = self.policy.model(self.local_batch.observation)
        actions = self.global_batch.action
        actions[self.index] = my_action_pred
        actor_loss = -self.critic_model(self.global_batch.observation, actions).mean()

        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()