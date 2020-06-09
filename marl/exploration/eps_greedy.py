from . import ExplorationProcess
import random
from marl.tools import _sub_child_lines

class EpsExplProcess(ExplorationProcess):
    """
    A generic class for exploration processes based on randomly choosing action with probability epsilon
    
    :param eps_deb: (float) The initial amount of exploration to process
    :param eps_fin: (float) The final amount of exploration to process
    :param deb_expl: (float) The percentage of time before starting exploration (default: 0.1)
    :param deb_expl: (float) The percentage of time before starting exploration (default: 0.1)
    """
    def __init__(self, eps_deb=1.0, eps_fin=0.1, deb_expl=0.1, fin_expl=0.9):
        self.eps_deb = eps_deb
        self.eps_fin = eps_fin
        self.eps = self.eps_deb
        if fin_expl < deb_expl:
            raise ValueError("'deb_expl' must be lower than 'fin_expl'")
        self.deb_expl = deb_expl
        self.fin_expl = fin_expl
    
    def reset(self, training_duration):
        """ Reinitialize some parameters """
        self.eps = self.eps_deb
        self.init_expl_step = int(self.deb_expl * training_duration)
        self.final_expl_step = int(self.fin_expl * training_duration)
    
    def update(self, t):
        """ Update epsilon linearly """   
        if t > self.init_expl_step:
            if self.eps_deb >= self.eps_fin:
                self.eps = max(self.eps_fin, self.eps_deb - (t-self.init_expl_step)*(self.eps_deb-self.eps_fin)/(self.final_expl_step-self.init_expl_step))
            else:
                self.eps = min(self.eps_fin, self.eps_deb - (t-self.init_expl_step)*(self.eps_deb-self.eps_fin)/(self.final_expl_step-self.init_expl_step))
                
    def expl_action(self, policy, observation):
        raise NotImplementedError
    
    def greedy_action(self, policy, observation):
        raise NotImplementedError
        
    def __call__(self, policy, observation):
        """ Choose an action according to the policy and the exploration rate """   
        
        greedy_action = self.greedy_action(policy, observation)
        random_action = self.expl_action(policy, observation)
        
        if random.random() < self.eps:
            return random_action
        else :
            return greedy_action

class EpsGreedy(EpsExplProcess):
    """
    The epsilon-greedy exploration class
    
    :param eps_deb: (float) The initial amount of exploration to process
    :param eps_fin: (float) The final amount of exploration to process
    :param deb_expl: (float) The percentage of time before starting exploration (default: 0.1)
    :param fin_expl: (float) The percentage of time after what we stop decreasing exploration (default: 0.9)
    """
    def __init__(self, eps_deb=1.0, eps_fin=0.1, deb_expl=0.1, fin_expl=0.9):
        super().__init__(eps_deb=eps_deb, eps_fin=eps_fin, deb_expl=deb_expl, fin_expl=fin_expl)
    
    def expl_action(self, policy, observation):
        return policy.random_action(observation)#action_space.sample()
    
    def greedy_action(self, policy, observation):
        return policy(observation)

class Greedy(EpsGreedy):
    """
    The Greedy process. The agent will take everytime the greedy action.
    """        
    def __init__(self):
        super(Greedy, self).__init__(eps_deb=0.0, eps_fin=0.0)
        
    def __call__(self, policy, observation):
        return policy(observation)

class EpsSoftmax(EpsGreedy):
    def greedy_action(self, policy, observation):
        return policy.softmax_action(observation)
        

class Softmax(EpsSoftmax):
     def __init__(self):
            super().__init__(eps_deb=0.0, eps_fin=0.0)
                

class EpsExpert(EpsGreedy):
    """
    The epsilon-expert process consists in taking an action randomly with probability epsilon and following Expert policy otherwise.
    
    :param expert: (Agent) The expert agent 
    :param eps_deb: (float) The intial amount of exploration 
    :param eps_fin: (float) The final amount of exploration 
    :param deb_expl: (float) The percentage of time before starting exploration (default: 0.1)
    :param fin_expl: (float) The percentage of time after what we stop decreasing exploration (default: 0.9)
    """  
    def __init__(self, expert, eps_deb=1.0, eps_fin=0.1, deb_expl=0.1, fin_expl=0.9):
        super().__init__(eps_deb=eps_deb, eps_fin=eps_fin, deb_expl=deb_expl, fin_expl=fin_expl)
        self.expert = expert
        
    def greedy_action(self, policy, observation):
        return self.expert.greedy_action(observation)
        
    def __repr__(self):
        return '{}<{}>({})'.format(self.__class__.__name__, self.expert.__class__.__name__, _sub_child_lines(self, separator=" ", exclude=["expert"]))
        
        
class Expert(EpsExpert):
    """
    In this case, actions will only be taken thanks to Expert policy.
    
    :param expert: (Agent) The expert agent 
    :param eps_deb: (float) The intial amount of exploration 
    :param eps_fin: (float) The final amount of exploration 
    """        
    def __init__(self, expert):
        super().__init__(expert=expert, eps_deb=0.0, eps_fin=0.0)
        
class HierarchicalEpsGreedy(EpsGreedy):
    """
    The HierarchicalEpsGreedy process consists in taking an action sampled from another exploration process with probability epsilon and following greedy policy otherwise.
    
    :param sub_expl: (marl.exploration.ExplorationProcess) The sub-exploration process 
    :param eps_deb: (float) The intial amount of exploration 
    :param eps_fin: (float) The final amount of exploration 
    :param deb_expl: (float) The percentage of time before starting exploration (default: 0.1)
    :param fin_expl: (float) The percentage of time after what we stop decreasing exploration (default: 0.9)
    """ 
    def __init__(self, sub_expl, eps_deb=1.0, eps_fin=0.1, deb_expl=0.1, fin_expl=0.9):
        super().__init__(eps_deb=eps_deb, eps_fin=eps_fin, deb_expl=deb_expl, fin_expl=fin_expl)
        self.sub_expl = sub_expl
    
    def reset(self, training_duration):
        """ Reinitialize some parameters """
        super().reset(training_duration)
        self.sub_expl.reset(training_duration)
        
    def update(self, t):
        """ Update epsilon linearly """   
        super().update(t)
        self.sub_expl.update(t)
        
    def expl_action(self, policy, observation):
        return self.sub_expl(policy, observation)
        
    def __repr__(self):
        return '{}<{}>({})'.format(self.__class__.__name__, self.sub_expl, _sub_child_lines(self, separator=" ", exclude=["sub_expl"]))
        

class EpsExpertEpsGreedy(HierarchicalEpsGreedy):
    """
    The EpsExpertEpsGreedy process is a hierarchical epsilon-greedy process consisting in taking an action sampled from EpsExpert process with probability epsilon and following greedy policy otherwise.
    
    :param expert: (Agent) The expert agent 
    :param eps_deb: (float) The intial amount of exploration 
    :param eps_fin: (float) The final amount of exploration 
    :param deb_expl: (float) The percentage of time before starting exploration (default: 0.1)
    :param fin_expl: (float) The percentage of time after what we stop decreasing exploration (default: 0.9)
    """ 
    def __init__(self, expert, epsG_deb=1.0, epsG_fin=0.1, debG_expl=0.1, finG_expl=0.9, epsE_deb=0.0, epsE_fin=1.):
        super().__init__(sub_expl=EpsExpert(expert, eps_deb=epsE_deb, eps_fin=epsE_fin, deb_expl=0., fin_expl=1.), eps_deb=epsG_deb, eps_fin=epsG_fin, deb_expl=debG_expl, fin_expl=finG_expl)