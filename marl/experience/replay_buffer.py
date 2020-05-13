import torch
import random
import pickle
import numpy as np
from collections import deque
from collections import namedtuple

from .sumtree import SumTree
from . import Experience
from ..tools import seq2unique_transition


transition_type = {
    "FFTransition" : ['observation', 'action', 'reward', 'next_observation', 'done_flag'],
    "RNNTransition" : ['observation','h0', 'action', 'reward', 'done_flag', 'seq_len']
}

transition_tuple = {
    "FFTransition": namedtuple('FFTransition', field_names=transition_type["FFTransition"]),
    "RNNTransition": namedtuple('RNNTransition', field_names=transition_type["RNNTransition"])
}

class ReplayMemory(Experience):
    def __init__(self, capacity, burn_in_frames=None, transition_type="FFTransition"):
        
        assert transition_type in transition_tuple.keys(), "Transition type not valid (must be in {})".format(transition_tuple.keys)
        
        self.capacity = capacity
        self.burn_in_frames = capacity//12 if burn_in_frames is None else burn_in_frames
        
        assert self.burn_in_frames < capacity
        
        self.memory = []
        self.position = 0
        self.transition_type = transition_type

    def push(self, *transition):
        assert len(transition_tuple[self.transition_type]._fields)==len(transition) , "Invalid number of arguments"
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition_tuple[self.transition_type](*transition)
        self.position = (self.position + 1) % self.capacity


    def push_tr(self, tr):
        assert len(transition_tuple[self.transition_type]._fields)==len(tr._fields) , "Invalid number of transition values : {} given instead of {} required".format(len(tr._fields), len(transition_tuple[self.transition_type]._fields))
        
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = tr
        self.position = (self.position + 1) % self.capacity 

    def sample(self, batch_size=1):
        assert batch_size <= len(self), "Batch size > Memory length"
        _sample = random.sample(self.memory, batch_size)
        return seq2unique_transition(_sample)

    def __len__(self):
        return len(self.memory)
    
    def __repr__(self):
        return 'ReplayMemory<{}>({}/{})'.format(self.transition_type, len(self), self.capacity)
    
    def get_transition(self, index):
        _sample = []
        for ind in index:
            _sample.append(self.memory[ind])
        _sample = list(zip(*_sample))
        sample_arr = [np.asarray(s) for s in _sample]
        # sample_arr = [torch.from_numpy(np.asarray(s)).float() for s in _sample]
        return transition_tuple[self.transition_type](*sample_arr)
    
    def sample_index(self, batch_size):
        assert batch_size <= len(self)
        return np.random.randint(len(self), size=batch_size)
    
    def as_dict(self, n=None):
        tr_dict = []
        for tr in self.memory:
            tr_dict.append(dict(tr._asdict()))
        return tr_dict if n is None else tr_dict[0:-1:len(self)//(n-1)]
    
    def save(self, n=None, filename="logs/experience.pickle"):
        with open(filename, 'wb') as f:
            pickle.dump(self.as_dict(n), f)


class PrioritizedReplayMemory(Experience):
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity, burn_in_frames=None, alpha=0.6, beta=0.4, eps=1e-6, transition_type="FFTransition"):
        
        assert transition_type in transition_tuple.keys(), "Transition type not valid (must be in {})".format(transition_tuple.keys)
        
        self.tree = SumTree(capacity)
        self.burn_in_frames = capacity//12 if burn_in_frames is None else burn_in_frames
        
        assert self.burn_in_frames < capacity
            
        # self.seed = seed
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        
        self.transition_type = transition_type
        
    @property
    def capacity(self):
        return self.tree.capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.eps) ** self.alpha
        
    def push(self, error, transition):
        assert len(transition_tuple[self.transition_type]._fields)==len(transition._fields) , "Invalid number of transition values : {} given instead of {} required".format(len(transition._fields), len(transition_tuple[self.transition_type]._fields))
        
        p = self._get_priority(error)
        self.tree.add(p, transition)        

    def sample(self, batch_size=1):
        sample = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            sample.append(data)
            idxs.append(idx)
        
        batch = seq2unique_transition(sample)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def get_current_transition(self):
        return self.transition(**self.current_transition)

    def update(self, index, error):
        priority = self._get_priority(error)
        priority = np.array(priority)
        index = np.array(index)
        for idx, p in list(zip(index, priority)):
            self.tree.update(idx, p)

    def __repr__(self):
        return 'PrioritizedReplayMemory<{}>({}/{})'.format(self.transition_type, len(self.tree),self.tree.capacity)
    
    def __len__(self):
        return len(self.tree)
    
    
    def as_dict(self, n=None):
        assert n>0
        n = max(len(self)//4, 1) if n is None else n
        
        tr_dict = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            tr_dict.append({"p":p, "transition":dict(data._asdict()), "index":idx})
        return tr_dict
    
    
    def save(self, n=None, filename="logs/experience.pickle"):
        with open(filename, 'wb') as f:
            pickle.dump(self.as_dict(n), f)
    
    # def push_error(self, error):
    #     # assert len(transition_tuple[self.transition_type]._fields)==len(transition) , "Invalid number of transition values : {} given instead of {} required".format(len(transition), len(self.transition._fields))
        
    #     p = self._get_priority(error)
    #     tr = self.get_current_transition()
    #     self.tree.add(p, tr)
        
    #     self.current_transition = self.none_transition_dict()
        
    # def push_transition(self, observation, action, reward, next_observation, done_flag):
    #     self.push_to_current_transition('observation', observation)
    #     self.push_to_current_transition('action', action)
    #     self.push_to_current_transition('reward', reward)
    #     self.push_to_current_transition('done_flag', done_flag)
    #     self.push_to_current_transition('next_observation', next_observation)
        
    # def push_to_current_transition(self, key, val):
    #     assert key in self.transition._fields, "'{}' is not a transition's field"
    #     self.current_transition[key] = val