import torch
import os


dll_path = os.path.abspath(__file__)
dll_path = os.path.dirname(os.path.dirname(dll_path))
dll_path = os.path.join(dll_path, "c.so")


torch.ops.load_library(dll_path)

mocha_energy = torch.ops.c.mocha_energy
cumprod_1mp = torch.ops.c.cumprod_1mp
cumsum_adp = torch.ops.c.cumsum_adp
window_cumsum_alpha_sigmoid = torch.ops.c.window_cumsum_alpha_sigmoid
window_cumsum_exp_alpha = torch.ops.c.window_cumsum_exp_alpha
mocha_context = torch.ops.c.mocha_context
lstmp = torch.ops.c.lstmp
lstmpcell = torch.ops.c.lstmpcell
ublstmp = torch.ops.c.ublstmp
sum_hard_attention = torch.ops.c.sum_hard_attention
mask = torch.ops.c.mask
mha_mask = torch.ops.c.mha_mask


__all__ = ["mocha_energy", "cumprod_1mp", "cumsum_adp", "window_cumsum_alpha_sigmoid", "window_cumsum_exp_alpha", "mocha_context", "lstmp", "lstmpcell", "ublstmp", 
           "sum_hard_attention", "mask", "mha_mask"]