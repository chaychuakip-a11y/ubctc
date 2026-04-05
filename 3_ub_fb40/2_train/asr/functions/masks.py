import torch
import torch.nn as nn
import torch.jit as jit
import os
from typing import Dict
from torch.utils.cpp_extension import load
from ..data import clip_mask
from ..c import *


# class selfatt_enc_dec_mask_function(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx, x, mask, label_mask, forward_val, backward_val):
#         x = selfatt_enc_dec_mask_(x, mask, label_mask, forward_val)
#         ctx.save_for_backward(*[mask, label_mask])
#         ctx.backward_val = backward_val
#         return x

#     @staticmethod
#     def backward(ctx, d_r):
#         d_r = d_r.contiguous()
#         mask, label_mask = ctx.saved_variables
#         backward_val = ctx.backward_val
#         d_r = selfatt_enc_dec_mask_(d_r, mask, label_mask, backward_val)
#         return d_r, None, None, None, None


# class selfatt_dec_dec_mask_function(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, label_mask, forward_val, backward_val):
#         x = selfatt_dec_dec_mask_(x, label_mask, forward_val)
#         ctx.save_for_backward(*[label_mask])
#         ctx.backward_val = backward_val
#         return x

#     @staticmethod
#     def backward(ctx, d_r):
#         d_r = d_r.contiguous()
#         backward_val = ctx.backward_val
#         label_mask, = ctx.saved_variables
#         d_r = selfatt_dec_dec_mask_(d_r, label_mask, backward_val)
#         return d_r, None, None, None



# def mha_mask(x: torch.Tensor, mask: torch.Tensor, window_width: int, forward_val: float,
#              backward_val: float) -> torch.Tensor:
#     x = x.contiguous()
#     mask = clip_mask(mask, x.size(2), 1)
#     return mha_mask(x, window_width, mask, forward_val, backward_val)


# @jit.ignore
# def selfatt_enc_dec_mask(x: torch.Tensor, mask: torch.Tensor, label_mask: torch.Tensor, forward_val: float, backward_val: float):
#     x = x.contiguous()
#     mask = clip_mask(mask, x.size(2), 1)
#     return selfatt_enc_dec_mask_function.apply(x, mask, label_mask, forward_val, backward_val)


# @jit.ignore
# def selfatt_dec_dec_mask(x: torch.Tensor, label_mask: torch.Tensor, forward_val: float, backward_val: float):
#     x = x.contiguous()
#     return selfatt_dec_dec_mask_function.apply(x, label_mask, forward_val, backward_val)
