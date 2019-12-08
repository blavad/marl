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
    

class MlpNet2(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_1=64, hidden_layer_2=64, hidden_activ=nn.ReLU):
        super(MlpNet2, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.h_activ = hidden_activ
        
        self.lin1 = nn.Linear(self.input_size, hidden_layer_1)
        self.lin2 = nn.Linear(hidden_layer_1, hidden_layer_2)
        self.output_layer = nn.Linear(hidden_layer_2, self.output_size)
         
    def forward(self, x):
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.output_layer(x)
        return x
    
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features