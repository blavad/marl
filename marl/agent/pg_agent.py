from . import TrainableAgent

class PGAgent(TrainableAgent):
    def __init__(self, policy, gae, env, experience_buffer, exploration_process, gamma=0.99, lr=0.1, dueling=True, off_policy=True, name="QAgent"):
        super(PGAgent, self).__init__(policy=policy, env=env, experience_buffer=experience_buffer, exploration_process=exploration_process, gamme=gamma, lr=lr, dueling=dueling, off_policy=True, name=name)
        self.gae = gae # general advantage estimation --> value function
    
    def process_experience(self, obs, act, rew, new_obs, done, terminal):
        raise NotImplementedError
        
    def update_model(self):
        raise NotImplementedError
        
    def action(self, observation):
        raise NotImplementedError
        
    def save_model(self, save_name):
        raise NotImplementedError
        
    def load_model(self, save_name):
        raise NotImplementedError