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


class CodaWeightedCTCLoss(nn.Module):
    """
    CTC loss with prior boost on coda senones.

    Mechanism: add a positive log-prob bias to coda dimensions before CTC
    alignment. This is equivalent to a Bayesian prior favoring coda emission.
    Re-normalized after boost so log_probs remain valid log-probabilities.

    Args:
        blankID: index of CTC blank in the vocabulary
        coda_mask: torch.BoolTensor of shape [vocab_size], True at coda senones
        coda_boost: log-space additive boost for coda dims (e.g., 1.1 ≈ log(3))
                    coda_boost=0 reduces to standard CTC loss.
    """
    def __init__(self, blankID, coda_mask, coda_boost=1.1):
        super(CodaWeightedCTCLoss, self).__init__()
        self.ctcLoss = nn.CTCLoss(blank=blankID, reduction='mean', zero_infinity=True)
        self.coda_boost = float(coda_boost)
        # Pre-compute boost vector once: coda dims get +coda_boost, others get 0
        boost_vec = torch.zeros_like(coda_mask, dtype=torch.float32)
        boost_vec[coda_mask] = self.coda_boost
        # persistent=False: not saved in state_dict (avoids pytorch2mat / checkpoint
        # compatibility issues), but still auto-moved to GPU via .cuda() / .to()
        self.register_buffer('boost_vec', boost_vec, persistent=False)

    def forward(self, x, ctc_list, x_len, ctc_len):
        # x: [T, B, V] log-probs (already log_softmax applied in net_ubctc)
        # Add boost broadcast over T,B
        boosted = x + self.boost_vec
        # Re-normalize across vocab dim to keep proper log-probs
        boosted = boosted - boosted.logsumexp(dim=-1, keepdim=True)
        return self.ctcLoss(boosted, ctc_list, x_len, ctc_len)
