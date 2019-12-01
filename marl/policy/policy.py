import importlib

class PolicySpec(object):
    def __init__(self, id, entry_point, **kwargs):
        self.id = id
        self.entry_point = entry_point
        self._kwargs = {} if kwargs is None else kwargs
        
    def make(self, **kwargs):
        if self.entry_point is None:
            raise Exception('Attempting to make deprecated policy {}.'.format(self.id))
        _kwargs = self._kwargs.copy()
        _kwargs.update(kwargs)
        if callable(self.entry_point):
            expl = self.entry_point(**_kwargs)
        else:
            cls = load(self.entry_point)
            expl = cls(**_kwargs)
        return expl

class Policy(object):
    policy = {}
    
    def __call__(self, state):
        raise NotImplementedError
    
    def load(self, filename):
        raise NotImplementedError

    def save(self, filename):
        raise NotImplementedError


    
    @classmethod
    def make(cls, id, *args, **kwargs):
        if isinstance(id, cls):
            return id
        else:
            return Policy.policy[id].make(**kwargs)
    
    @classmethod
    def register(cls, id, entry_point, **kwargs):
        if id in Policy.policy.keys():
            raise Exception('Cannot re-register id: {}'.format(id))
        Policy.policy[id] = PolicySpec(id, entry_point, **kwargs)
        
    @classmethod
    def available(cls):
        return Policy.policy.keys()

def load(name):
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


def register(cls, id, entry_point, **kwargs):
    Policy.register(id, entry_point, **kwargs)
    
def make(id, **kwargs):
    return Policy.make(id, **kwargs)
    
def available():
    return Policy.available()