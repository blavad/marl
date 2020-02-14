import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class FootCnn(nn.Module):
    def __init__(self, shape, my_actions):
        super(FootCnn, self).__init__()
        self.c, self.h, self.w = shape
        self.a = my_actions
        self.seq = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(self.c, 32, 3),
            nn.ReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(32,16,3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.h*self.w*16, 350),
            nn.Linear(350, self.a)
        )
        self.a2 = my_actions
            
    def forward(self, x):
        x = self.seq(x)
        return x