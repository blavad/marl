from marl.tools import ClassSpec, _std_repr

class Experience(object):
    experience = {}
    
    def push(self, *args):
        raise NotImplementedError
    
    def sample(self, batch_siz=1):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError
    
    def __repr__(self):
        return _std_repr(self)
    
    @classmethod
    def make(cls, id, **kwargs):
        if isinstance(id, cls):
            return id
        else:
            return Experience.experience[id].make(**kwargs)
    
    @classmethod
    def register(cls, id, entry_point, **kwargs):
        if id in Experience.experience.keys():
            raise Exception('Cannot re-register id: {}'.format(id))
        Experience.experience[id] = ClassSpec(id, entry_point, **kwargs)
        
    @classmethod
    def available(cls):
        return Experience.experience.keys()
    

def register(id, entry_point, **kwargs):
    Experience.register(id, entry_point, **kwargs)
    
def make(id, **kwargs):
    return Experience.make(id, **kwargs)
    
def available():
    return Experience.available()