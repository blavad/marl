import importlib
import gym
import numpy as np
import torch
import logging

#### Function for simpler susage ####

def _addindent(s_, numSpaces):
    s = s_.split('\n')
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s

def _std_child_lines(obj, separator='\n'):
    child_lines = []
    for key, value in obj.__dict__.items():
        mod_str = repr(value) 
        mod_str = _addindent(mod_str, 2)
        child_lines.append('# ' + key + ': ' + mod_str)
    child_str = ''
    if child_lines:
        child_str +=  '{} '.format(separator).join(child_lines) 
    return child_str

def _sub_child_lines(obj, separator='\n', exclude=[]):
    child_lines = []
    for key, value in obj.__dict__.items():
        if key not in exclude:
            mod_str = repr(value) 
            mod_str = _addindent(mod_str, 2)
            child_lines.append('# ' + key + ': ' + mod_str)
    child_str = ''
    if child_lines:
        child_str +=  '{} '.format(separator).join(child_lines) 
    return child_str

def _std_repr(obj, separator='\n'):
    chil_str = _std_child_lines(obj, separator)
    main_str = obj.__class__.__name__
    if chil_str is not '':
        main_str += '({}'.format(separator)
        main_str +=  chil_str
        main_str += '{})'.format(separator)
    return main_str

def _inline_std_repr(obj):
    return _std_repr(obj, separator=' ')

def gymSpace2dim(gym_space):
    if isinstance(gym_space, gym.spaces.Discrete):
        return gym_space.n
    if isinstance(gym_space, gym.spaces.Box):
        l_sp = list(gym_space.shape)
        return l_sp[0] if len(l_sp) <2 else l_sp
    

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
    
def reset_logging():
    import logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

##### Tools for data preprocessing ##### 

def super_cat(obs, act):
    if type(obs[0]) is not np.ndarray and type(obs[0]) is not list and len(obs.shape) <=1 and len(act.shape) <=1:
        concat = [obs, act]
    else:
        concat = [super_cat(o, a) for o, a in zip(obs,act)]
    return np.concatenate(concat)

def seq2unique_dict(seq_dict):
    new_dict = {}
    for key in seq_dict[0].keys():
        new_dict[key] = []
    for sub_dict in seq_dict:
        for key in sub_dict.keys():
            new_dict[key].append(sub_dict[key])
    return new_dict

def seq2unique_transition(seq_transition):
    try:
        dict_transition = {}
        transition_class = seq_transition[0].__class__
        fields_ = seq_transition[0]._fields
        # Init value of dict to void
        for field in fields_:
            val = getattr(seq_transition[0], field)
            if isinstance(val, dict):
                dict_transition[field] = {}
                for key in val.keys():
                    dict_transition[field][key] = []
            else:
                dict_transition[field] = []
        # Add elements in dict of full transition
        for tr in seq_transition:
            for field in fields_:
                if isinstance(dict_transition[field], dict):
                    for key in dict_transition[field].keys():
                        dict_transition[field][key].append(getattr(tr, field)[key])    
                else:
                    dict_transition[field].append(getattr(tr, field))
        return transition_class(**dict_transition)
    except AttributeError:
        logging.warning("\n\n\n!!!!!!! Attribute Error !!!!!!!!!")
        logging.warning(seq_transition)

def zeros_like(var):
    zero_var = None
    if isinstance(var, dict):
        zero_var = {}
        for key, val in var.items():
            zero_var[key] = zeros_like(val)
    if isinstance(var, list):
        zero_var = [zeros_like(val) for val in var]
    if isinstance(var, torch.Tensor):
        zero_var = torch.zeros_like(var)
    if isinstance(var, float) or isinstance(var, int):
        zero_var = 0
    if isinstance(var, str):
        zero_var = ''
    if isinstance(var, bool):
        zero_var = False
    if zero_var is None:
        zero_var = None
    
    # assert zero_var is not None, "Erreur type nonreconnu ({}): {}".format(type(var), var)
    return zero_var 

def ones_like(var):
    return v_like(var, value=1.)

def v_like(var, value=0):
    new_var = None
    if isinstance(var, dict):
        new_var = {}
        for key, val in val.items():
            new_var[key] = v_like(val, value)
    if isinstance(var, list):
        new_var = [v_like(val, value) for val in var]
    if isinstance(var,torch.Tensor):
        new_var = torch.full_like(var, value)
    if isinstance(var, float) or isinstance(var, int):
        new_var = value
    if isinstance(var, bool):
        if value == 0:
            new_var = False
        if value == 1:
            new_var = True
        else :
            raise TypeError("bool doesn't match with value ", value)
        
    assert new_var is not None, "Erreur type non reconnu ({}): {}".format(type(var), var)
    return new_var 

            
def pad_like(transition):
    dict_transition = {}
    if transition.__class__.__name__ != "FFTransition":
        raise NotImplementedError
    observation = zeros_like(transition.observation)
    action = zeros_like(transition.action)
    reward = zeros_like(transition.reward)
    done_flag = ones_like(transition.done_flag)
    next_observation = zeros_like(transition.next_observation)
    return transition.__class__(observation=observation,
                                action=action,
                                reward=reward,
                                done_flag=done_flag,
                                next_observation=next_observation)

    
def is_done(done):
    if type(done) is bool:
        return done
    elif type(done) is list:
        done = [is_done(d) for d in done]
        return all(done)