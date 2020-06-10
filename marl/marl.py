import os
import marl
from .agent import TrainableAgent, Agent
from torch.utils.tensorboard import SummaryWriter

class MAS(object):
    """
    The class of multi-agent "system".
    
    :param agents_list: (list) The list of agents in the MAS
    :param name: (str) The name of the system
    """
    
    def __init__(self, agents_list=[], name="mas"):
        self.name = name
        self.agents = agents_list
        
    def append(self, agent):
        """
        Add an agent to the system.

        :param agent: (Agent) The agents to be added
        """
        self.agents.append(agent)          
    
    def action(self, observation):
        """
        Return the joint action.

        :param observation: The joint observation
        """
        return [ag.greedy_action(ag, obs) for ag, obs in zip(self.agents, observation)]    
    
    def get_by_name(self, name):
        for ag in self.agents:
            if ag.name == name:
                return ag
        return None
    
    def get_by_id(self, id):
        for ag in self.agents:
            if ag.id == id:
                return ag
        return None
        
    def __len__(self):
        return len(self.agents)

class MARL(TrainableAgent, MAS):
    """
    The class for a multi-agent reinforcement learning.
    
    :param agents_list: (list) The list of agents in the MARL model
    :param name: (str) The name of the system
    """
    def __init__(self, agents_list=[], name='marl', log_dir="logs"):
        MAS.__init__(self, agents_list=agents_list, name=name)
        # self.experience = marl.experience.make("ReplayMemory", capacity=10000)
        
        self.log_dir = log_dir
        self.init_writer(log_dir)
        
    def reset(self):
        for  ag in self.agents:
            ag.reset()
            
    def init_writer(self, log_dir):
        log_path = os.path.join(log_dir, self.name)
        self.writer = SummaryWriter(log_path)
        for ag in self.agents:
            if isinstance(ag, TrainableAgent):
                ag.init_writer(log_path)
        
    def store_experience(self, *args):
        # TrainableAgent.store_experience(self, *args)
        observation, action, reward, next_observation, done = args
        for i, ag in enumerate(self.agents):
            if isinstance(ag, TrainableAgent):
                ag.store_experience(observation[i], action[i], reward[i], next_observation[i], done[i])
            
    def update_model(self, t):
        # TrainableAgent.update_model(self, t)        
        for ag in self.agents:
            if isinstance(ag, TrainableAgent):
                ag.update_model(t)
    
    def reset_exploration(self, nb_timesteps):
        # TrainableAgent.update_exploration(self, nb_timesteps)        
        for ag in self.agents:
            if isinstance(ag, TrainableAgent):
                ag.reset_exploration(nb_timesteps)
    
    def update_exploration(self, t):
        # TrainableAgent.update_exploration(self, t)        
        for ag in self.agents:
            if isinstance(ag, TrainableAgent):
                ag.exploration.update(t)
        
    def action(self, observation):
        return [ag.action(obs) for ag, obs in zip(self.agents, observation)]
        
    def greedy_action(self, observation):
        return [ag.greedy_action(obs) for ag, obs in zip(self.agents, observation)]
    
    def save_policy(self, folder='.', filename='', timestep=None):
        """
        Save the policy in a file called '<filename>-<agent_name>-<timestep>'.
        
        :param folder: (str) The path to the directory where to save the model(s)
        :param filename: (str) A specific name for the file (ex: 'test2')
        :param timestep: (int) The current timestep  
        """
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename_tmp = "{}-{}".format(filename, self.name) if filename is not '' else "{}".format(self.name)
        for ag in self.agents:
            if isinstance(ag, TrainableAgent):
                ag.save_policy(folder=folder, filename=filename_tmp, timestep=timestep)
        
    def get_best_rew(rew1, rew2):
        for ind, ag in enumerate(self.agents):
            rew1[ind] = ag.get_best_rew(rew1[ind], rew2[ind])
        return rew1
        
    def save_policy_if_best(self, best_rew, rew, folder='.', filename=''):
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename_tmp = "{}-{}".format(filename, self.name) if filename is not '' else "{}".format(self.name)
        for ind, ag in enumerate(self.agents):
            if isinstance(ag, TrainableAgent):
                best_rew[ind] = ag.save_policy_if_best(best_rew[ind], rew[ind], folder=folder, filename=filename_tmp)
            else:
                best_rew[ind] = ag.get_best_rew(best_rew[ind], rew[ind])
        return best_rew
            
    def worst_rew(self):
        best_rew = []
        for ag in self.agents:
            best_rew += [ag.worst_rew()]
        return best_rew
                
    def load_model(self, filename):
        for ag in self.agents:
            if isinstance(ag, TrainableAgent):
                ag.load_model(filename)
                
    def training_log(self, verbose):
        log = ""
        if verbose >= 2:
            for ag in self.agents:
                if isinstance(ag, TrainableAgent):
                    log += ag.training_log(verbose)
                else:
                    log += "#> {}\n".format(ag.name)
        return log