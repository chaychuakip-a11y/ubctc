import torch
import torch.nn as nn
import torch.nn.functional as F
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


class CodaWeightedCeLoss(nn.Module):
    """
    Frame-level cross-entropy loss with extra weight on coda senones.

    Use case: HMM-DNN hybrid auxiliary supervision. Each frame has a
    senone target (from forced alignment / pfile celabel). Coda senones
    (ll, mm, ng, kg-coda for Korean digits) get a multiplicative weight
    on their per-frame CE loss, giving the model stronger gradient signal
    on the under-learned coda dimensions.

    This is a clean alternative to CodaWeightedCTCLoss — it does NOT
    interfere with CTC alignment, just provides direct frame-level
    supervision in parallel.

    Args:
        coda_mask: bool tensor [vocab_size], True at coda senone IDs
        coda_weight: multiplier on per-frame loss when target is coda
                     (default 3.0)
        ignore_index: padding label value to ignore (default -1)
    """
    def __init__(self, coda_mask, coda_weight=3.0, ignore_index=-1):
        super(CodaWeightedCeLoss, self).__init__()
        self.coda_weight = float(coda_weight)
        self.ignore_index = ignore_index
        self.register_buffer('coda_mask', coda_mask, persistent=False)

    def forward(self, logits, target):
        """
        logits: [N, V] raw logits (before softmax)
        target: any shape, will be flattened to [N]; values in [0, V)
                or == ignore_index (-1) for padding
        """
        target_flat = target.reshape(-1).long()

        # Per-frame CE; ignore_index frames contribute 0
        ce_per_frame = F.cross_entropy(
            logits, target_flat,
            ignore_index=self.ignore_index,
            reduction='none'
        )

        # Build per-frame weight (coda_weight at coda frames, 1.0 elsewhere)
        with torch.no_grad():
            valid_mask = (target_flat != self.ignore_index)
            safe_target = target_flat.clamp(min=0)  # avoid -1 indexing
            is_coda = self.coda_mask[safe_target]
            weights = torch.where(
                is_coda,
                torch.tensor(self.coda_weight, device=logits.device, dtype=logits.dtype),
                torch.tensor(1.0, device=logits.device, dtype=logits.dtype)
            )
            weights = weights * valid_mask.to(logits.dtype)

        # Weighted mean over valid frames
        denom = weights.sum().clamp(min=1.0)
        return (ce_per_frame * weights).sum() / denom
