# by pcli2
import torch
import torch.jit as jit

@jit.ignore
def clip_mask(mask: torch.Tensor, clip_length: int, dim: int):
    assert mask.shape[dim] % clip_length == 0, "mask dim {0} must be a integer multiple of clip_length {1}".format(mask.shape[dim], clip_length)
    nmod = int(mask.shape[dim] / clip_length)
    indices = torch.Tensor([i*nmod for i in range(clip_length)]).long()
    indices = indices.to(mask.device)
    cliped_mask = torch.index_select(mask, dim, indices).contiguous()
    return cliped_mask


@jit.ignore
def cnn2rnn(x: torch.Tensor) -> torch.Tensor:
    assert len(x.shape) == 4 and x.shape[2] == 1, "x must be 4-dim tensor and dim 2 must be 1"
    x = x.squeeze(2)
    x = x.permute(2, 0, 1)
    x = x.contiguous()
    return x


@jit.ignore
def rnn2cnn(x: torch.Tensor) -> torch.Tensor:
    assert len(x.shape) == 3, "x must be 3-dim tensor"
    x = x.permute(1, 2, 0)
    x = x.unsqueeze(2)
    x = x.contiguous()
    return x
