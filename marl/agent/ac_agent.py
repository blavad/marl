import marl
from marl import MARL
from marl.agent import QAgent, TrainableAgent

class ActorCriticAgent(MARL):
    def __init__(self, critic_agent, actor_agent, env, experience, exploration_process="EpsGreedy", lr_actor=0.001, lr_critic=0.001, gamma=0.99, dueling=True, off_policy=True, name="ACAgent"):
        super(ActorCriticAgent, self).__init__([critic_agent, actor_agent], env)
        self.name = name
        self.experience = experience
        self.critic = critic_agent
        self.actor = marl.agent.make(actor_agent, env=self.env)
        
    def action(self, observation):
        return [ag.action(obs) for ag, obs in zip(self.agents, observation)]
        
    def save_model(self, save_name):
        for ag in self.agents:
            ag.save_model("{}-{}".format(save_name, ag.name))
        
    def load_model(self, save_name):
        for ag in self.agents:
            ag.load_model(save_name)
            
    def update_model(self, t):
        # Update critic
        o, a, r, o1, done = self.critic_agent.experience.sample(self.batch_size)
        
        target = self.critic_agent._calculate_target(o, a, r, o1, done)
        output = self.critic_agent.policy.Q(o, a)
        
        loss = self.critic_agent.loss(output, target)
        
        self.critic_agent.optimizer.zero_grad()
        loss.backward()
        self.critic_agent.optimizer.step()
        
        if t % self.target_update_freq == 0:
            self.update_target()

        # Update actor
        o, a, r, o1, done = self.actor_agent.experience.sample(self.batch_size)
        
        pd = self.actor_agent.forward(o)
        log_prob = pd.log_prob(a) #.unsqueeze(0)
        self.actor_loss = -(log_prob * self.critic_agent.Q(o, a).detach()).mean()
        
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_agent.optimizer.step()
        

class ActorCriticAgent2(MARL):
    def __init__(self, critic_agent, actor_agent, env, name="ACAgent"):
        super(ActorCriticAgent2, self).__init__([critic_agent, actor_agent], env, name=name)
        
    def action(self, observation):
        return self.agents[1].action(observation)