import marl
from marl.tools import ClassSpec, _std_repr, is_done

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
        return np.array(self.policy(observation))
    
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
    
    counter = 0
     
    def __init__(self, policy, observation_space=None, action_space=None, model=None, experience="ReplayMemory-10000", exploration="EpsGreedy", gamma=0.99, lr=0.001, batch_size=32, name="TrainableAgent"):
        TrainableAgent.counter +=1
        
        self.id = TrainableAgent.counter
        self.name = name
        
        self.lr = lr
        self.batch_size = batch_size
        self.gamma = gamma
        
        # Create policy, exploration and experience
        self.policy = marl.policy.make(policy, model=model, observation_space=observation_space, action_space=action_space)
        self.experience = marl.experience.make(experience)
        self.exploration = marl.exploration.make(exploration)
        
        assert self.experience.capacity > self.batch_size
    
    @property
    def observation_space(self):
        return self.policy.observation_space
    
    @property
    def action_space(self):
        return self.policy.action_space
    
    def store_experience(self, *args):
        self.experience.push(*args)
        
    def update_model(self, t):
        raise NotImplementedError
    
    def reset_exploration(self, nb_timesteps):
        self.exploration.reset(nb_timesteps)
    
    def update_exploration(self, t):
        self.exploration.update(t)
        
    def action(self, observation):
        return np.array(self.exploration(self.policy, observation))
        
    def greedy_action(self, observation):
        return Agent.action(self, observation)
        
    def save_policy(self, filename='', timestep=None):
        filename_tmp = "{}-{}".format(filename, self.name) if filename is not '' else "{}".format(self.name)
        filename_tmp = "{}".format(filename_tmp) if timestep is None else "{}-{}".format(filename_tmp, timestep)
        self.policy.model.save(filename_tmp)
        
    def save_all(self):
        pass
    
    def learn(self, env, nb_timesteps, max_num_step=100, test_freq=1000, save_freq=1000):
        timestep = 0
        episode = 0
        self.reset_exploration(nb_timesteps)
        while timestep < nb_timesteps:
            episode +=1
            obs = env.reset()
            done = False
            for _ in range(timestep, timestep + max_num_step):
                action = self.action(obs)
                obs2, rew, done, _ = env.step(action)
                self.store_experience(obs, action, rew, obs2, done)
                self.update_model(timestep)
                obs = obs2
                timestep+=1
                self.update_exploration(timestep)
                
                # Test the model
                if timestep % test_freq == 0:
                    _, m_rews, std_rews = self.test(env, 100, max_num_step=max_num_step, render=False)
                    print("Step {}/{} ({} episodes) --- Mean rewards {} -- Dev rewards {}".format(timestep, nb_timesteps, episode, m_rews, std_rews))
                    break
                
                # Save the model
                if timestep % save_freq == 0:
                    print("Step {}/{} --- Save Model".format(timestep, nb_timesteps))
                    self.save_policy(filename=self.name, timestep=timestep)
                    
                if is_done(done):
                    break
                    
                
    def test(self, env, nb_episodes=1, max_num_step=200, render=True):
        rewards = []
        for _ in range(nb_episodes):
            observation = env.reset()
            done = False
            for step in range(max_num_step):
                if render:
                    env.render()
                action = self.greedy_action(observation)
                observation, reward, done, _ = env.step(action)
                sum_r = reward if step==0 else sum_r + reward
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