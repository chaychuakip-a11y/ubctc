# pcli2 2020 Jan.

import torch
import torch.jit as jit
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
import os
from ..c import *
from torch.nn import Parameter
from torch.utils import cpp_extension
from torch.utils.cpp_extension import load
from ..data import clip_mask
from typing import Dict, Tuple

class LSTMPCell(nn.Module):
    def __init__(self, input_dim, cell_num, out_dim, clip_threshold=1.0):
        super(LSTMPCell, self).__init__()
        self.clip_threshold = float(clip_threshold)

        self.weight_x = nn.Parameter(torch.zeros(4 * cell_num, input_dim))
        self.weight_r = nn.Parameter(torch.zeros(4 * cell_num, out_dim))
        self.weight_bias = nn.Parameter(torch.zeros(1, 4 * cell_num))
        self.weight_p = nn.Parameter(torch.zeros(out_dim, cell_num))

        init.uniform_(self.weight_x.data, -0.05, 0.05)
        init.uniform_(self.weight_r.data, -0.05, 0.05)
        init.zeros_(self.weight_bias.data)
        init.uniform_(self.weight_p.data, -0.05, 0.05)


    def forward(self, data, c_1, r_1):
        c, r = lstmpcell(data, self.weight_x, self.weight_r, self.weight_bias, self.weight_p, c_1, r_1, self.clip_threshold)
        return c, r


class LSTMP(nn.Module):
    def __init__(self, input_dim, cell_num, out_dim, clip_threshold=1.0):
        super(LSTMP, self).__init__()
        self.clip_threshold = float(clip_threshold)

        self.weight_x = nn.Parameter(torch.zeros(4 * cell_num, input_dim))
        self.weight_r = nn.Parameter(torch.zeros(4 * cell_num, out_dim))
        self.weight_bias = nn.Parameter(torch.zeros(1, 4 * cell_num))
        self.weight_p = nn.Parameter(torch.zeros(out_dim, cell_num))

        init.uniform_(self.weight_x.data, -0.05, 0.05)
        init.uniform_(self.weight_r.data, -0.05, 0.05)
        init.zeros_(self.weight_bias.data)
        init.uniform_(self.weight_p.data, -0.05, 0.05)


    def forward(self, data, mask):
        # mask = clip_mask(mask, data.size(0), 0)
        output = lstmp(data, self.weight_x, self.weight_r,
                           self.weight_bias, self.weight_p, self.clip_threshold, mask)
        return output


    def step(self, data, c_1, r_1) -> Tuple[torch.Tensor, torch.Tensor]:
        return lstmpcell(data, self.weight_x, self.weight_r, self.weight_bias, self.weight_p, c_1, r_1, self.clip_threshold)



