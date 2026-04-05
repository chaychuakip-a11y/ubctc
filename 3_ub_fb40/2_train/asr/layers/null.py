import torch
import torch.jit as jit
import torch.nn as nn

class NullModule(nn.Module):
    def __init__(self):
        super(NullModule, self).__init__()


    def forward(self, x):
        return x