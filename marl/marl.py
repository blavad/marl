from agent import TrainableAgent

class MARL(TrainableAgent):
    
    def __init__(self, agents, env):
        self.agents = agents
        self.env = env
        
    def store_experience(self, *args, **kwargs):
        for ag in self.agents:
            ag.store_experience(args, kwargs)
            
    def update_model(self):
        for ag in self.agents:
            ag.update_model()
        
    def action(self, observation):
        return [ag.action(obs) for ag, obs in zip(self.agents, observation)]
        
    def save_model(self, save_name):
        for ag in self.agents:
            ag.save_model("{}-{}".format(save_name, ag.name))
        
    def load_model(self, save_name):
        for ag in self.agents:
            ag.load_model(save_name)
    
    def learn(self, nb_timesteps):
        timestep = 0
        while timestep < nb_timesteps:
            obs = self.env.reset()
            done = False
            while not done:
                action = self.action(obs)
                obs2, rew, done, _ = self.env.step(action)
                self.store_experience(obs, rew, action, obs2)
                self.update_model()
                obs = obs2