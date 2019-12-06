from marl.tools import ClassSpec, _std_repr
        
class ExplorationProcess(object):
    process = {}
    
    def reset(self, training_duration):
        raise NotImplementedError
        
    def update(self, t):
        raise NotImplementedError
    
    def __call__(self):
        raise NotImplementedError
    
    def __repr__(self):
        return _std_repr(self)
    
    @classmethod
    def make(cls, id, *args, **kwargs):
        if isinstance(id, cls):
            return id
        else:
            return ExplorationProcess.process[id].make(**kwargs)
    
    @classmethod
    def register(cls, id, entry_point, **kwargs):
        if id in ExplorationProcess.process.keys():
            raise Exception('Cannot re-register id: {}'.format(id))
        ExplorationProcess.process[id] = ClassSpec(id, entry_point, **kwargs)
        
    @classmethod
    def available(cls):
        return ExplorationProcess.process.keys()
    
def register(id, entry_point, **kwargs):
    ExplorationProcess.register(id, entry_point, **kwargs)
    
def make(id, **kwargs):
    return ExplorationProcess.make(id, **kwargs)
    
def available():
    return ExplorationProcess.available()
