# pcli2 2020 Jan.
import torch
import torch.jit as jit
import torch.nn as nn


class ConcatFrLayer(nn.Module):
    def __init__(self, nmod):
        super(ConcatFrLayer, self).__init__()
        self.nmod = nmod


    def forward(self, data):
        data = data.permute(0,1,3,2)
        data = data.reshape(data.size(0),data.size(1),-1,data.size(3) * self.nmod)
        data = data.permute(0,1,3,2)
        data = data.reshape(data.size(0),-1,1,data.size(3)).contiguous()
        return data
