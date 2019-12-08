from . import TrainableAgent
from marl.policy import PolicyApprox

import torch
import torch.optim as optim

class PGAgent(TrainableAgent):
    def __init__(self, model, experience="ReplayMemory-1000", exploration="EpsGreedy", gamma=0.99, lr=0.1, dueling=True, off_policy=True, name="QAgent"):
        super(PGAgent, self).__init__(policy=PolicyApprox(model), experience=experience, exploration=exploration, gamme=gamma, lr=lr, off_policy=True, name=name)
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