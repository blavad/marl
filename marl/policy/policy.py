import torch
import torch.nn as nn

from marl.tools import ClassSpec, _std_repr

class Policy(object):
    policy = {}
    
    def __init__(self, action_space):
        self.action_space = action_space
    
    def __call__(self, state):
        raise NotImplementedError
    
    def __repr__(self):
        return _std_repr(self)
    
    def random_action(self, observation=None):
        return self.action_space.sample()
    
    @classmethod
    def make(cls, id, **kwargs):
        if isinstance(id, cls):
            return id
        else:
            return Policy.policy[id].make(**kwargs)
    
    @classmethod
    def register(cls, id, entry_point, **kwargs):
        if id in Policy.policy.keys():
            raise Exception('Cannot re-register id: {}'.format(id))
        Policy.policy[id] = ClassSpec(id, entry_point, **kwargs)
        
    @classmethod
    def available(cls):
        return Policy.policy.keys()
    
class ModelBasedPolicy(Policy):
    
    def __init__(self, model):
        self.model = model
    
    def load(self, filename):
        if isinstance(self.model, nn.Module):
            self.model.load_state_dict(torch.load(filename))
        else:
            self.model.load(filename=filename)

    def save(self, filename):
        if isinstance(self.model, nn.Module):
            torch.save(self.model.state_dict(), filename)
        else:
            self.model.save(filename=filename)

def register(id, entry_point, **kwargs):
    Policy.register(id, entry_point, **kwargs)
    
def make(id, **kwargs):
    return Policy.make(id, **kwargs)
    
def available():
    return Policy.available()