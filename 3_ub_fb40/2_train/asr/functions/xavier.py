import torch.nn as nn
import math

def xavier(x, p=6):
    shape = x.shape
    count = 1
    for s in shape:
        count = count * s
    fan_in = count / shape[0]
    scale = math.sqrt(p/fan_in)
    nn.init.uniform_(x.data, -scale, scale)