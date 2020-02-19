import importlib
import gym
import numpy as np

def _addindent(s_, numSpaces):
    s = s_.split('\n')
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s

def _std_repr(obj):
    child_lines = []
    for key, value in obj.__dict__.items():
        mod_str = repr(value)
        mod_str = _addindent(mod_str, 2)
        child_lines.append('# ' + key + ': ' + mod_str)
    lines = child_lines
    main_str = obj.__class__.__name__ + '('
    if lines:
        main_str += '\n  ' + '\n  '.join(lines) + '\n'
    main_str += ')'
    return main_str

def gymSpace2dim(gym_space):
    if isinstance(gym_space, gym.spaces.Discrete):
        return gym_space.n
    if isinstance(gym_space, gym.spaces.Box):
        l_sp = list(gym_space.shape)
        return l_sp[0] if len(l_sp) <2 else l_sp
    
def super_cat(obs, act):
    if type(obs[0]) is not np.ndarray and type(obs[0]) is not list and len(obs.shape) <=1 and len(act.shape) <=1:
        concat = [obs, act]
    else:
        concat = [super_cat(o, a) for o, a in zip(obs,act)]
    return np.concatenate(concat)

def is_done(done):
    if type(done) is bool:
        return done
    elif type(done) is  list:
        done = [is_done(d) for d in done]
        return all(done)

def load(name):
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn

class ClassSpec(object):
    def __init__(self, id, entry_point, **kwargs):
        self.id = id
        self.entry_point = entry_point
        self._kwargs = {} if kwargs is None else kwargs
        
    def make(self, **kwargs):
        if self.entry_point is None:
            raise Exception('Attempting to make deprecated class {}.'.format(self.id))
        _kwargs = self._kwargs.copy()
        _kwargs.update(kwargs)
        if callable(self.entry_point):
            expl = self.entry_point(**_kwargs)
        else:
            cls = load(self.entry_point)
            expl = cls(**_kwargs)
        return expl
    
