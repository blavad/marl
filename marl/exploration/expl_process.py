from marl.tools import ClassSpec, _std_repr, _inline_std_repr
        
class ExplorationProcess(object):
    """
    The generic exploration class
    """
    
    process = {}
    
    def reset(self, training_duration):
        """ 
        Intialize some additional values and reset the others 
        
        :param training_duration: (int) Number of timesteps while training
        """    
        raise NotImplementedError
        
    def update(self, t):
        """ 
        If required update exploration parameters
        
        :param t: (int) The current timestep
        """
        pass
    
    def __call__(self):
        raise NotImplementedError
    
    def __repr__(self):
        return _inline_std_repr(self)
    
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
