import marl
from marl.tools import ClassSpec, _std_repr

from marl.policy.policy import Policy
from marl.exploration import ExplorationProcess

import torch
import torch.optim as optim
import numpy as np

class Agent(object):
    
    agents = {}
    
    def __init__(self, policy, name="UnknownAgent"):
        self.name = name
        self.policy = Policy.make(policy)
        
    def action(self, observation):
        return self.policy(observation)
    
    def __repr__(self):
        return _std_repr(self)
    
    @classmethod
    def make(cls, id, **kwargs):
        if isinstance(id, cls):
            return id
        else:
            return Agent.agents[id].make(**kwargs)
    
    @classmethod
    def register(cls, id, entry_point, **kwargs):
        if id in Agent.agents.keys():
            raise Exception('Cannot re-register id: {}'.format(id))
        Agent.agents[id] = ClassSpec(id, entry_point, **kwargs)
        
    @classmethod
    def available(cls):
        return Agent.agents.keys()
        
class TrainableAgent(Agent):    
    def __init__(self, policy, observation_space, action_space, experience="ReplayMemory-10000", exploration_process="EpsGreedy", gamma=0.99, lr=0.001, name="TrainableAgent"):
        self.name = name
        
        self.observation_space = observation_space
        self.action_space = action_space
        
        # Create policy, exploration and experience
        self.policy = marl.policy.make(policy, observation_space=self.observation_space, action_space=self.action_space)
        self.experience = marl.experience.make(experience)
        self.exploration_process = marl.exploration.make(exploration_process)
        
        self.gamma = gamma
        self.lr = lr
        
        # self.optimizer = optimizer(self.policy.parameters(), lr=self.lr)
        # self.loss = loss
    
    def store_experience(self, *args):
        self.experience.push(*args)
        
    def update_model(self, t):
        raise NotImplementedError
    
    def update_exploration(self, t):
        self.exploration_process.update(t)
        
    def action(self, observation):
        return self.exploration_process(self, observation)
        
    def greedy_action(self, observation):
        return Agent.action(self, observation)
        
    def save_policy(self, save_name):
        self.policy.save(save_name)
        
    def save_all(self):
        pass
    
    def learn(self, env, nb_timesteps, test_freq=1000, freq_save=1000):
        timestep = 0
        episode = 0
        self.exploration_process.reset(nb_timesteps)
        while timestep < nb_timesteps:
            episode +=1
            obs = env.reset()
            done = False
            while not done:
                action = self.action(obs)
                obs2, rew, done, _ = env.step(action)
                self.store_experience(obs, action, rew, obs2, done)
                self.update_model(timestep)
                obs = obs2
                timestep+=1
                self.update_exploration(timestep)
                
                if timestep % test_freq == 0:
                    _, m_rews, std_rews = self.test(env, 10, max_num_step=300, render=False)
                    print("Step {}/{} ({} episodes) --- Mean rewards {} -- Dev rewards {}".format(timestep, nb_timesteps, episode, m_rews, std_rews))
                    break
                
    def test(self, env, nb_episodes=1, max_num_step=100, render=True):
        rewards = []
        for _ in range(nb_episodes):
            observation = env.reset()
            sum_r = 0
            done = False
            for step in range(max_num_step):
                if render:
                    env.render()
                action = self.greedy_action(observation)
                observation, reward, done, _ = env.step(action)
                sum_r += reward
                if done:
                    break
            rewards.append(sum_r)
        if render:
            env.close()
        return rewards, np.mean(rewards), np.std(rewards)
        
def register(id, entry_point, **kwargs):
    Agent.register(id, entry_point, **kwargs)
    
def make(id, **kwargs):
    return Agent.make(id, **kwargs)
    
def available():
    return Agent.available()