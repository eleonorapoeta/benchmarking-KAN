import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicNet(nn.Module):
    
    def __init__(self, num_features, num_classes, num_intermediate_nodes):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.layers = 0
        
        scale = 20
        self.lin1 = torch.nn.Linear(self.num_features,  num_intermediate_nodes)        
        self.lin2 = torch.nn.Linear(num_intermediate_nodes, self.num_classes)
        self.drop = torch.nn.Dropout(0.5)
        
    def forward(self, xin):
        self.layers = 0
        
        x = F.silu(self.lin1(xin))
        self.layers += 1

        x = self.drop(x)
        
        x = F.silu(self.lin2(x))
        self.layers += 1
        return x
      