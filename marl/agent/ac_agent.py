import marl
from marl import MARL
from marl.agent import QAgent, TrainableAgent

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
        