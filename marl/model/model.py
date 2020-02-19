from marl.tools import ClassSpec, _std_repr
import torch
import torch.nn as nn

class Model(object):
    model = {}
    
    def save(self, filename):
        torch.save(self.value, filename)
        
    def load(self, filename):
        self.value = torch.load(filename)
    
    def __repr__(self):
        return _std_repr(self)
    
    @classmethod
    def make(cls, id, **kwargs):
        if isinstance(id, cls) or isinstance(id, nn.Module):
            return id
        elif isinstance(id, str):
            return Model.model[id].make(**kwargs)
        else:
            return id(**kwargs)
            
    
    @classmethod
    def register(cls, id, entry_point, **kwargs):
        if id in Model.model.keys():
            raise Exception('Cannot re-register id: {}'.format(id))
        Model.model[id] = ClassSpec(id, entry_point, **kwargs)
        
    @classmethod
    def available(cls):
        return Model.model.keys()

def register(id, entry_point, **kwargs):
    Model.register(id, entry_point, **kwargs)
    
def make(id, **kwargs):
    return Model.make(id, **kwargs)
    
def available():
    return Model.available()