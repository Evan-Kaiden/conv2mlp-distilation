import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class convModel(nn.Module):
    def __init__(self, in_channels, num_classes=100):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)


    def forward(self, x):
        layer_output = []
        x = self.pool1(F.relu(self.conv1(x)))
        layer_output.append(x.flatten(1))    
        x = self.pool2(F.relu(self.conv2(x)))
        layer_output.append(x.flatten(1))      
        x = self.pool3(F.relu(self.conv3(x)))
        layer_output.append(x.flatten(1))      
        x = self.flatten(x) 
        x = F.relu(self.fc1(x))
        layer_output.append(x.flatten(1))      
        x = self.fc2(x)   
        layer_output.append(x)          
        return F.log_softmax(x, dim=1), layer_output

class linearModel(nn.Module):
    def __init__(self, convmodel, input_shape=(1, 3, 32, 32)):
        super().__init__()

        self.layers = nn.ModuleList()

        dummy_input = torch.zeros(input_shape)
        x = dummy_input

        for layer in convmodel.children():
            if isinstance(layer,(nn.Linear, nn.Conv2d)):
                in_features = x.flatten(1).shape[1]
            

            try:
                x = layer(x)
            except Exception:
                x = x.flatten(1)
                x = layer(x)

            if isinstance(layer, (nn.Linear, nn.MaxPool2d)):
                out_features = x.flatten(1).shape[1]
                self.layers.append(nn.Linear(in_features, out_features))


    def forward(self, x):
        x = x.flatten(1)
        layer_output = []
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
            layer_output.append(x)
        
        x = self.layers[-1](x)
        layer_output.append(x)
        return F.log_softmax(x, dim=1), layer_output
