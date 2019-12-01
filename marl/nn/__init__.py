import torch
import torch.nn as nn
import torch.nn.functional as F

class MlpNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=[64,64], hidden_activ=nn.ReLU):
        super(MlpNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.h_activ = hidden_activ
        
        in_size = hidden_size[-1] if len(hidden_size) > 0 else self.input_size
        
        self.feature_extractor = self._build_module(hidden_size)
        self.output_layer = nn.Linear(in_size, self.output_size)
    
    def _build_module(self, h_size):
        in_size = self.input_size
        modules = []
        for n_units in h_size:
            modules.append(nn.Linear(in_size, n_units))
            modules.append(self.h_activ())
            in_size = n_units
        return nn.Sequential(*modules)
         
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.output_layer(x)
        return x