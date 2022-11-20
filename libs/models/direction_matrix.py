import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class DirectionMatrix(nn.Module):
    def __init__(self, shift_dim, input_dim=None, out_dim=None, inner_dim=512,
                 bias=True, w_plus = False, num_layers = 14, initialization = 'normal'):
        super(DirectionMatrix, self).__init__()
        self.shift_dim = shift_dim
        self.input_dim = input_dim if input_dim is not None else np.product(shift_dim)
        self.out_dim = out_dim if out_dim is not None else np.product(shift_dim)
        self.w_plus = w_plus
        self.num_layers = num_layers

        if self.w_plus:
            print("Linear Direction matrix-A in w+ space: input dimension {}, output dimension {}, shift dimension {} ".format(self.input_dim, 
            self.out_dim, self.shift_dim))
        else:
            print("Linear Direction matrix-A type : input dimension {}, output dimension {}, shift dimension {} ".format(self.input_dim,
            self.out_dim, self.shift_dim))

        if self.w_plus:
            out_dim = self.out_dim * num_layers
        else:
            out_dim = self.out_dim

        self.linear = nn.Linear(self.input_dim, out_dim, bias=bias)
        self.linear.weight.data = torch.zeros_like(self.linear.weight.data)
          
        if initialization == 'normal':
            torch.nn.init.normal_(self.linear.weight, mean=0.0, std=0.03)
        if not self.w_plus and initialization == 'eye':
            min_dim = int(min(self.input_dim, out_dim))      
            self.linear.weight.data[:min_dim, :min_dim] = torch.eye(min_dim)
        if self.w_plus and initialization == 'eye':
            min_dim = int(min(self.input_dim, out_dim))      
            for layer_cnt in range(num_layers):
                self.linear.weight.data[layer_cnt*self.out_dim:(layer_cnt*self.out_dim + min_dim), :min_dim] = torch.eye(min_dim)
            
    def forward(self, input):
        input = input.view([-1, self.input_dim])

        out  = self.linear(input)
        if self.w_plus:
            out = out.view(len(input), self.num_layers, self.shift_dim)   
       
        return out
