from marl import MARL
from marl.agent import TrainableAgent, DQNAgent
from marl.experience import ReplayMemory
from marl.policy import StochasticPolicy

import torch
import torch.optim as optim

class PGAgent(TrainableAgent):
    def __init__(self, model, experience="ReplayMemory-1000", exploration="EpsGreedy", gamma=0.99, lr=0.1, dueling=True, off_policy=True, name="QAgent"):
        super(PGAgent, self).__init__(policy=StochasticPolicy(model), experience=experience, exploration=exploration, gamme=gamma, lr=lr, off_policy=True, name=name)
        # self.gae = gae # general advantage estimation --> value function
        self.optimizer = optim.Adam(self.policy.model.parameters(), lr=self.lr)
    
    def update_model(self, t, gae):
        batch = self.experience.sample(self.batch_size)
        self.optimizer.zero_grad()
        
        pd = self.policy.forward(batch.observation)
        log_prob = pd.log_prob(batch.action) #.unsqueeze(0)
        loss = -(log_prob * gae).mean()
        
        loss.backward()
        self.optimizer.step()
        
    def save_model(self, save_name):
        raise NotImplementedError
        
    def load_model(self, save_name):
        raise NotImplementedError


class ActorCriticAgent(TrainableAgent):
    def __init__(self, critic_agent, actor_agent, experience=None, exploration_process="EpsGreedy", lr_actor=0.001, lr_critic=0.001, gamma=0.99, off_policy=True, name="ACAgent"):
        super(ActorCriticAgent, self).__init__([critic_agent, actor_agent])
        self.name = name
        self.experience = experience
        
        self.critic = critic_agent
        self.actor = actor_agent
        
        self.policy = actor_agent.policy
        
    def save_model(self, save_name):
        for ag in self.agents:
            ag.save_model("{}-{}".format(save_name, ag.name))
        
    def load_model(self, save_name):
        for ag in self.agents:
            ag.load_model(save_name)
            
    def update_model(self, t):
        # Update critic
        batch = self.critic_agent.experience.sample(self.batch_size)
        
        self.critic_agent.optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        
        target = self.critic_agent.target(batch).detach()
        curr_value = self.critic_agent.policy.Q(batch.observation, batch.action)
        
        loss = self.critic_agent.loss(curr_value, target)
        loss.backward()
        self.critic_agent.optimizer.step()
        
        if t % self.target_update_freq == 0:
            self.update_target()

        # Update actor
        batch_actor = self.actor_agent.experience.sample(self.batch_size)
        
        
        
        pd = self.actor_agent.forward(o)
        log_prob = pd.log_prob(a) #.unsqueeze(0)
        self.actor_loss = -(log_prob * self.critic_agent.Q(o, a).detach()).mean()
        
        self.actor_loss.backward()
        self.actor_agent.optimizer.step()

    
    
class DDPGAgent(MARL):
    
    def __init__(self, critic_policy, actor_policy, experience_buffer, lr_critic = 0.01, lr_actor = 0.01, tau = 0.01, gamma = 0.95, num_sub_policy=2, capacity_buff = 1.e6, seed=None):
        self.lr_critic = lr_critic
        self.lr_actor = lr_actor
        self.momentum = 0.95
        self.tau = tau
        self.num_sub_policy = num_sub_policy
        self.env = env
        
        self.critic = DQNAgent(critic_policy, ReplayMemory(capacity_buff), "EpsGreedy")
        self.actor = PGAgent(actor_policy, ReplayMemory(capacity_buff), "EpsGreedy")
        
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