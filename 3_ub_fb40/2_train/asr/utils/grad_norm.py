# pcli2 2020 Jan, modified from pytorch source code
import torch
import math


def clip_grad_norm_(parameters, ClipGradient, ClipGradient2, Discount):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    ssqrt = 0.0
    ssqrt = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2)
    clip_coef = 1.0
    if ssqrt > ClipGradient:
        if ssqrt > ClipGradient2:
            clip_coef = Discount * ClipGradient / ssqrt
        else:
            slope = (1 - Discount) * ClipGradient / (ClipGradient2 - ClipGradient)
            clip_coef = (ClipGradient - (ssqrt - ClipGradient) * slope) / ssqrt
            

    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef)



def clip_grad_norm(parameters, ClipGradient, ClipGradient2, Discount):
    return clip_grad_norm_(parameters, ClipGradient, ClipGradient2, Discount)
