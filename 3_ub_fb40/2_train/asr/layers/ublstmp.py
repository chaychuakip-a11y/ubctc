# pcli2 2020 Jan.
# ublstmp python binding
import torch
import torch.jit as jit
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
import os
from typing import Dict
from torch.nn import Parameter
from torch.utils import cpp_extension
from torch.utils.cpp_extension import load
from ..data import clip_mask
from ..c import ublstmp

class UBLSTMP(nn.Module):
    def __init__(self, input_dim, cell_num_fwd, out_dim_fwd, cell_num_bwd, out_dim_bwd, fwd_step, bwd_block, clip_threshold=1.0):
        super(UBLSTMP, self).__init__()

        self.fwd_step = fwd_step
        self.bwd_block = bwd_block
        self.clip_threshold = float(clip_threshold)

        self.weight_x_fwd = nn.Parameter(torch.zeros(4 * cell_num_fwd, input_dim))
        self.weight_r_fwd = nn.Parameter(torch.zeros(4 * cell_num_fwd, out_dim_fwd))
        self.weight_bias_fwd = nn.Parameter(torch.zeros(1, 4 * cell_num_fwd))
        self.weight_p_fwd = nn.Parameter(torch.zeros(out_dim_fwd, cell_num_fwd))

        self.weight_x_bwd = nn.Parameter(torch.zeros(4 * cell_num_bwd, input_dim))
        self.weight_r_bwd = nn.Parameter(torch.zeros(4 * cell_num_bwd, out_dim_bwd))
        self.weight_bias_bwd = nn.Parameter(torch.zeros(1, 4 * cell_num_bwd))
        self.weight_p_bwd = nn.Parameter(torch.zeros(out_dim_bwd, cell_num_bwd))

        init.uniform_(self.weight_x_fwd.data, -0.05, 0.05)
        init.uniform_(self.weight_r_fwd.data, -0.05, 0.05)
        init.zeros_(self.weight_bias_fwd.data)
        init.uniform_(self.weight_p_fwd.data, -0.05, 0.05)

        init.uniform_(self.weight_x_bwd.data, -0.05, 0.05)
        init.uniform_(self.weight_r_bwd.data, -0.05, 0.05)
        init.zeros_(self.weight_bias_bwd.data)
        init.uniform_(self.weight_p_bwd.data, -0.05, 0.05)


    def forward(self, data, mask):
        # mask = clip_mask(mask, data.size(0), 0)
        output = ublstmp(data, self.weight_x_fwd, self.weight_r_fwd, self.weight_bias_fwd, self.weight_p_fwd,
                         self.weight_x_bwd, self.weight_r_bwd, self.weight_bias_bwd, self.weight_p_bwd,
                         self.fwd_step, self.bwd_block, self.clip_threshold, mask)
        return output
