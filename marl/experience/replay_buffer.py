import random
from collections import namedtuple
from . import Experience
from collections import deque
import numpy as np

Transition = namedtuple('Transition',
                        ('obs', 'action', 'next_obs', 'reward', 'done_flag'))

class ReplayMemory(Experience):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size=1):
        _sample = random.sample(self.memory, batch_size)
        _sample = list(zip(*_sample))
        sample_arr = [np.asarray(s) for s in _sample]
        return Transition(*sample_arr)

    def __len__(self):
        return len(self.memory)
    
class ExpReplay:
    def __init__(self, memory_size=5000, burn_in=1000):
        self.memory_size = memory_size
        self.burn_in = burn_in

    def burn_in_capacity(self):
        return len(self) / self.burn_in
        
class ExpReplayBuffer(ExpReplay):
    
    def __init__(self, memory_size=5000, burn_in=1000):
        ExpReplay.__init__(self, memory_size, burn_in)
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.Buffer = namedtuple('Buffer', 
            field_names=['state', 'action', 'reward', 'done', 'next_state'])
        self.replay_memory = deque(maxlen=memory_size)

    def sample(self, batch_size=32):
        samples = np.random.choice(len(self.replay_memory), batch_size, 
                                   replace=False)
        batch = list(zip(*[self.replay_memory[i] for i in samples]))
        return [np.asarray(b) for b in batch]

    def append(self, state, action, reward, next_state, done):
        self.replay_memory.append(
            self.Buffer(state, action, reward, next_state, done))
        
    def __len__(self):
        return len(self.replay_memory)