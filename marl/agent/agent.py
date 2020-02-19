import marl
from marl.tools import ClassSpec, _std_repr, is_done

from marl.policy.policy import Policy
from marl.exploration import ExplorationProcess

import os
import time
import torch
import torch.optim as optim
import numpy as np

class Agent(object):
    """
    The class of generic agent.
    
    :param policy: (Policy) The policy of the agent
    :param name: (str) The name of the agent      
    """
    
    agents = {}
    
    counter = 0
    
    def __init__(self, policy, name="UnknownAgent"):
        Agent.counter +=1
        
        self.id = Agent.counter
        
        self.name = name
        self.policy = Policy.make(policy)
        
    def action(self, observation):
        """
        Return the action given an observation  
        :param observation: The observation
        """
        return self.policy(observation)
    
    def greedy_action(self, observation):
        """
        Return the greedy action given an observation  
        :param observation: The observation
        """
        return Agent.action(self, observation)
    
    def test(self, env, nb_episodes=1, max_num_step=200, render=True, time_laps=0.):
        """
        Test a model.
        
        :param env: (Gym) The environment
        :param nb_episodes: (int) The number of episodes to test
        :param max_num_step: (int) The maximum number a step before stopping an episode
        :param render: (bool) Whether to visualize the test or not (using render function of the environment)
        """
        rewards = np.array([])
        for episode in range(nb_episodes):
            observation = env.reset()
            done = False
            if render:
                env.render()
                time.sleep(time_laps)
            for step in range(max_num_step):
                action = self.greedy_action(observation)
                observation, reward, done, _ = env.step(action)
                sum_r = np.array(reward) if step==0 else np.add(sum_r, reward)
                if render:
                    env.render()
                    time.sleep(time_laps)
                if is_done(done):
                    break
            rewards = np.array([sum_r/step]) if episode==0 else np.append(rewards, [sum_r/step], axis=0)
        if render:
            env.close()
        return rewards, rewards.mean(axis=0), rewards.std(axis=0)
    
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
    """
    The class of trainable agent.
    
    :param policy: (Policy) The policy 
    :param observation_space: (gym.Spaces) The observation space
    :param action_space: (gym.Spaces) The action space
    :param experience: (Experience) The experience memory data structure
    :param exploration: (Exploration) The exploration process 
    :param lr: (float) The learning rate
    :param gamma, batch_size: (float) The training parameters
    :param name: (str) The name of the agent      
    """
         
    def __init__(self, policy, observation_space=None, action_space=None, model=None, experience="ReplayMemory-10000", exploration="EpsGreedy", gamma=0.99, lr=0.001, batch_size=32, name="TrainableAgent"):
        Agent.__init__(self, policy=marl.policy.make(policy, model=model, observation_space=observation_space, action_space=action_space), name=name)
        
        self.lr = lr
        self.batch_size = batch_size
        self.gamma = gamma
        
        # Create policy, exploration and experience
        self.experience = marl.experience.make(experience)
        self.exploration = marl.exploration.make(exploration)
        
        assert self.experience.capacity >= self.batch_size
    
    @property
    def observation_space(self):
        return self.policy.observation_space
    
    @property
    def action_space(self):
        return self.policy.action_space
    
    def store_experience(self, *args):
        """
        Store a transition in the experience buffer.
        """
        self.experience.push(*args)
        
    def update_model(self, t):
        """
        Update the model.
        """
        raise NotImplementedError
    
    def reset_exploration(self, nb_timesteps):
        """
        Reset the exploration process. 
        """
        self.exploration.reset(nb_timesteps)
    
    def update_exploration(self, t):
        """
        Update the exploration process.
        """
        self.exploration.update(t)
        
    def action(self, observation):
        """
        Return an action given an observation (action in selected according to the exploration process).
        
        :param observation: The observation
        """
        return self.exploration(self.policy, observation)
        
    def save_policy(self, folder='.',filename='', timestep=None):
        """
        Save the policy in a file called '<filename>-<agent_name>-<timestep>'.
        
        :param filename: (str) A specific name for the file (ex: 'test2')
        :param timestep: (int) The current timestep  
        """
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename_tmp = "{}-{}".format(filename, self.name) if filename is not '' else "{}".format(self.name)
        filename_tmp = "{}".format(filename_tmp) if timestep is None else "{}-{}".format(filename_tmp, timestep)
        
        filename_tmp = os.path.join(folder, filename_tmp)
        self.policy.save(filename_tmp)
        
    def save_all(self):
        pass
    
    def learn(self, env, nb_timesteps, max_num_step=100, test_freq=1000, save_freq=1000, save_folder="models", render=False, time_laps=0.):
        """
        Start the learning part.
        
        :param env: (Gym) The environment
        :param nb_timesteps: (int) The total duration (in number of steps)
        :param max_num_step: (int) The maximum number a step before stopping episode
        :param test_freq: (int) The frequency of testing model
        :param save_freq: (int) The frequency of saving model
        """
        print("#> Start learning process !")
        timestep = 0
        episode = 0
        self.reset_exploration(nb_timesteps)
        while timestep < nb_timesteps:
            self.update_exploration(timestep)
            episode +=1
            obs = env.reset()
            done = False
            if render:
                env.render()
                time.sleep(time_laps)
            for _ in range(timestep, timestep + max_num_step):
                action = self.action(obs)
                obs2, rew, done, _ = env.step(action)
                self.store_experience(obs, action, rew, obs2, done)
                self.update_model(timestep)
                obs = obs2
                timestep+=1
                if render:
                    env.render()
                    time.sleep(time_laps)
            
                # Save the model
                if timestep % save_freq == 0:
                    print("Step {}/{} --- Save Model".format(timestep, nb_timesteps))
                    self.save_policy(timestep=timestep, folder=save_folder)
                    
                # Test the model
                if timestep % test_freq == 0:
                    _, m_rews, std_rews = self.test(env, 100, max_num_step=max_num_step, render=False)
                    print("Step {}/{} ({} episodes) --- Mean rewards {} -- Dev rewards {}\n".format(timestep, nb_timesteps, episode, m_rews, std_rews))
                    break
                
                if is_done(done):
                    break
        print("#> End of learning process !")
    

class MATrainable(object):
    def __init__(self, mas, index):    
        self.mas = mas
        self.index = index
    
    def set_mas(self, mas):
        self.mas = mas
        for ind, ag in enumerate(self.mas.agents):
            if ag.id == self.id:
                self.index = ind

        
def register(id, entry_point, **kwargs):
    Agent.register(id, entry_point, **kwargs)
    
def make(id, **kwargs):
    return Agent.make(id, **kwargs)
    
def available():
    return Agent.available()

