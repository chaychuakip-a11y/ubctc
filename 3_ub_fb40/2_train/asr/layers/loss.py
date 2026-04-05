import torch
import torch.nn as nn
import torch.jit as jit
from torch.jit import Final


class CeLoss(nn.Module):
    smooth: Final[float]

    def __init__(self, smooth=0.95):
        super(CeLoss, self).__init__()
        self.smooth = smooth


    def forward(self, x: torch.Tensor, att_label: torch.Tensor) -> torch.Tensor:
        lprob, target = self.getlabel(x, att_label)
        smooth_loss = self.get_smooth_loss(lprob, target)
        return smooth_loss


    @jit.ignore
    def getlabel(self, x, att_label):
        lprob = nn.functional.log_softmax(x, dim=1)
        num_classes = x.shape[1]
        ori_att_label = att_label.clone()
        ori_att_label = ori_att_label.flatten()
        ori_att_label[ori_att_label < 0] = -1
        ori_att_label = ori_att_label.long()
        target = ori_att_label
        target = target.unsqueeze(-1)
        _, new_target = torch.broadcast_tensors(lprob, target)
        remove_pad_mask = new_target.ne(-1)
        lprob = lprob[remove_pad_mask]
        lprob = lprob.reshape((-1, num_classes))
        target = target[target != -1]
        target = target.unsqueeze(-1)
        return lprob, target


    @jit.ignore
    def get_smooth_loss(self, lprob, target):
        valid = lprob.shape[0]
        num_classes = lprob.shape[1]
        ####################    
        smooth_label = torch.ones_like(lprob) * (1 - self.smooth) / (num_classes - 1)
        smooth_label[range(smooth_label.shape[0]), target.reshape(-1, ).long()] = self.smooth
        smooth_loss = -(smooth_label * lprob).sum() / valid
        return smooth_loss


class CTCLoss(jit.ScriptModule):
    def __init__(self, blankID):
        super(CTCLoss, self).__init__()
        self.ctcLoss = nn.CTCLoss(blank=blankID, reduction='mean', zero_infinity=True)

    def forward(self, x, ctc_list, x_len , ctc_len):
        ctc_loss = self.ctcLoss(x, ctc_list, x_len, ctc_len)
        return ctc_loss
