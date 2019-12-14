import torch
import random
import numpy as np
from collections import deque
from collections import namedtuple

from . import Experience

class ReplayMemory(Experience):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.field_names = ['observation', 'action', 'reward', 'next_observation', 'done_flag']
        self.transition = namedtuple('Transition', field_names=self.field_names)

    def push(self, *args):
        if len(self.memory) <= self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size=1):
        assert batch_size <= len(self)
        _sample = random.sample(self.memory, batch_size)
        _sample = list(zip(*_sample))
        sample_arr = [np.asarray(s) for s in _sample]
        # sample_arr = [torch.from_numpy(np.asarray(s)).float() for s in _sample]
        return self.transition(*sample_arr)

    def __len__(self):
        return len(self.memory)
    
    def __repr__(self):
        return 'ReplayMemory({}/{})'.format(len(self),self.capacity)
    
    def get_transition(self, index):
        _sample = []
        for ind in index:
            _sample.append(self.memory[ind])
        _sample = list(zip(*_sample))
        sample_arr = [np.asarray(s) for s in _sample]
        # sample_arr = [torch.from_numpy(np.asarray(s)).float() for s in _sample]
        return self.transition(*sample_arr)
    
    def sample_index(self, batch_size):
        assert batch_size <= len(self)
        return np.random.randint(len(self), size=batch_size)        
        

    
class ExpReplay:
    def __init__(self, memory_size=5000, burn_in=1000):
        self.memory_size = memory_size
        self.burn_in = burn_in

    def burn_in_capacity(self):
        return len(self) / self.burn_in