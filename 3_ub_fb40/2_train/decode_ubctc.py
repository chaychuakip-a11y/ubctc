#!/usr/bin/env python3
"""
Prototype CTC decoder for UBCTC network.

Network path:
    input (b, feat_dim, 1, T)
    -> ConcatFrLayer(nmod=4)   : frame-stack 4 frames -> (b, feat_dim*4, 1, T//4)
    -> cnn2rnn                 : (T//4, b, feat_dim*4)
    -> LSTMP(160, 1024, 256)
    -> LSTMP(256, 1024, 256)
    -> UBLSTMP(256, 512,150, 512,150, fwd_step=4, bwd_block=6)  -> 300-dim
    -> rnn2cnn
    -> Conv2d(300->1024) -> LeakyReLU -> Conv2d(1024->256)
    -> classification: Conv2d(256->9004)
    -> CTC decode  (blank = 9003)

Usage:
    python decode_ubctc.py \\
        --model   /path/to/checkpoint.pt \\
        --feat    /path/to/utt.npy \\
        [--dict   /path/to/tokens.txt] \\
        [--mode   greedy|beam] \\
        [--beam   10] \\
        [--blank  9003] \\
        [--gpu    0]

Feature file (.npy):
    Shape (T, feat_dim) float32, e.g. 40-dim log-filterbank.
    T will be zero-padded to the nearest multiple of 4 automatically.

Dictionary file (optional):
    One token per line: "<token> <id>"  (tab or space separated).
"""

import sys
import os
import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F

# Ensure the training directory is on sys.path so we can import net_ubctc and asr.*
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from net_ubctc import Ubctc
from asr.data import clip_mask, cnn2rnn, rnn2cnn


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device) -> Ubctc:
    """Load Ubctc model from a checkpoint file."""
    model = Ubctc()
    raw = torch.load(checkpoint_path, map_location='cpu')

    # Support several common checkpoint layouts
    if isinstance(raw, dict):
        if 'model' in raw:
            state = raw['model']
        elif 'state_dict' in raw:
            state = raw['state_dict']
        elif 'net' in raw:
            state = raw['net']
        else:
            state = raw
    else:
        state = raw  # assume it is already an OrderedDict

    # Strip a "module." prefix that appears when saved from DataParallel
    cleaned = {}
    for k, v in state.items():
        cleaned[k.replace('module.', '', 1)] = v

    model.load_state_dict(cleaned, strict=True)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_input(feat: np.ndarray, device: torch.device):
    """
    Convert a (T, feat_dim) numpy array to the CNN-format tensor expected by
    the Ubctc encoder, and build the corresponding rnn_mask.

    ConcatFrLayer(nmod=4) requires T to be a multiple of 4; we zero-pad if not.

    Returns:
        x        : (1, feat_dim, 1, T_padded)   float32 tensor
        rnn_mask : (T_padded, 1, 1)              float32 tensor  (1=valid, 0=pad)
        T_orig   : original (un-padded) time length
    """
    T_orig, feat_dim = feat.shape
    nmod = 4  # must match ConcatFrLayer(4) in Encoder

    # Pad T to the nearest multiple of nmod
    T_pad = math.ceil(T_orig / nmod) * nmod
    if T_pad > T_orig:
        pad = np.zeros((T_pad - T_orig, feat_dim), dtype=feat.dtype)
        feat = np.concatenate([feat, pad], axis=0)

    x = torch.from_numpy(feat).float()           # (T_pad, feat_dim)
    x = x.T.unsqueeze(0).unsqueeze(2)            # (1, feat_dim, 1, T_pad)

    # Build mask: 1 for valid frames, 0 for padding
    rnn_mask = torch.zeros(T_pad, 1, 1)
    rnn_mask[:T_orig, :, :] = 1.0

    return x.to(device), rnn_mask.to(device), T_orig


# ---------------------------------------------------------------------------
# CTC decoders
# ---------------------------------------------------------------------------

def ctc_greedy_decode(log_probs: torch.Tensor, blank_id: int):
    """
    Greedy CTC decode: argmax per frame → strip blank → collapse repeats.

    Args:
        log_probs : (T, num_class) tensor
        blank_id  : index of the blank label

    Returns:
        List[int] of decoded token IDs
    """
    ids = torch.argmax(log_probs, dim=1).tolist()
    decoded = []
    prev = -1
    for idx in ids:
        if idx == blank_id:
            prev = blank_id
            continue
        if idx != prev:
            decoded.append(idx)
        prev = idx
    return decoded


def _logsumexp2(a: float, b: float) -> float:
    """Numerically-stable log(exp(a) + exp(b))."""
    NEG_INF = float('-inf')
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def ctc_prefix_beam_decode(log_probs: torch.Tensor, blank_id: int, beam_size: int = 10):
    """
    CTC prefix beam search.

    State:  dict{ prefix_tuple -> (log_prob_b, log_prob_nb) }
      log_prob_b  : log probability of this prefix ending with a blank
      log_prob_nb : log probability of this prefix ending with a non-blank

    Args:
        log_probs : (T, num_class) tensor
        blank_id  : index of the blank label
        beam_size : number of active hypotheses

    Returns:
        List[int] of decoded token IDs (best hypothesis)
    """
    NEG_INF = float('-inf')
    lp = log_probs.cpu().float().numpy()          # (T, num_class)
    T, num_class = lp.shape

    # {prefix: (prob_b, prob_nb)}
    beams = {(): (0.0, NEG_INF)}                  # empty prefix starts with prob_b=0

    for t in range(T):
        lp_t = lp[t]
        # Candidate non-blank ids sorted by probability (prune search space)
        top_ids = np.argsort(lp_t)[::-1][:max(beam_size * 2, 30)]

        new_beams: dict = {}

        def _add(prefix, pb, pnb):
            if prefix not in new_beams:
                new_beams[prefix] = (NEG_INF, NEG_INF)
            ob, onb = new_beams[prefix]
            new_beams[prefix] = (_logsumexp2(ob, pb), _logsumexp2(onb, pnb))

        for prefix, (pb, pnb) in beams.items():
            p_total = _logsumexp2(pb, pnb)

            # Extend with blank → same prefix, prob_b updates
            _add(prefix, p_total + lp_t[blank_id], NEG_INF)

            # Extend with each non-blank symbol
            for c in top_ids:
                if c == blank_id:
                    continue
                lp_c = float(lp_t[c])
                new_prefix = prefix + (int(c),)

                if len(prefix) > 0 and prefix[-1] == c:
                    # Repeating last label: can only come from a blank-ending path
                    _add(new_prefix, NEG_INF, pb + lp_c)
                else:
                    _add(new_prefix, NEG_INF, p_total + lp_c)

        # Prune to beam_size by total log-prob
        scored = sorted(new_beams.items(),
                        key=lambda kv: _logsumexp2(kv[1][0], kv[1][1]),
                        reverse=True)
        beams = dict(scored[:beam_size])

    # Pick the best prefix
    best_prefix = max(beams, key=lambda p: _logsumexp2(beams[p][0], beams[p][1]))
    return list(best_prefix)


# ---------------------------------------------------------------------------
# Main decode function
# ---------------------------------------------------------------------------

@torch.no_grad()
def decode_utterance(model: Ubctc, feat: np.ndarray, device: torch.device,
                     decode_mode: str = 'greedy',
                     beam_size: int = 10,
                     blank_id: int = 9003):
    """
    Run encoder → classification → CTC decode for one utterance.

    Args:
        model       : loaded Ubctc in eval mode
        feat        : (T, feat_dim) numpy array
        device      : torch device
        decode_mode : 'greedy' or 'beam'
        beam_size   : beam size (only used when decode_mode='beam')
        blank_id    : CTC blank label id (default 9003 = num_class-1)

    Returns:
        hyp       : List[int] decoded token ids
        log_probs : (T', num_class) tensor (frame-level log-probabilities)
    """
    x, rnn_mask, T_orig = prepare_input(feat, device)
    # meta dict mirrors what the training dataloader provides
    meta = {
        "rnn_mask": rnn_mask,
        # att_label not used by encoder; provide a dummy to avoid KeyError
        "att_label": torch.zeros(1, 1, dtype=torch.long, device=device),
    }

    # Clip mask to match the time-downsampled encoder output
    # (same operation that Ubctc.forward does before calling encoder)
    enc = model.encoder
    meta["rnn_mask"] = clip_mask(meta["rnn_mask"], enc.concat_fr.nmod, 0)

    # --- Encoder forward ---
    enc_out = enc(x, meta)               # (1, 256, 1, T')

    # --- Flatten batch dimension for classification (matches Ubctc.forward) ---
    b, d, f, t = enc_out.shape
    enc_out_flat = enc_out.permute((2, 1, 3, 0)).reshape((1, d, 1, t * b))

    # --- Classification head ---
    logit = model.classification(enc_out_flat)   # (1, 9004, 1, T')
    logit = logit.squeeze().permute((1, 0))       # (T', 9004)

    log_probs = F.log_softmax(logit, dim=-1)      # (T', 9004)

    # --- CTC decode ---
    if decode_mode == 'greedy':
        hyp = ctc_greedy_decode(log_probs, blank_id)
    elif decode_mode == 'beam':
        hyp = ctc_prefix_beam_decode(log_probs, blank_id, beam_size)
    else:
        raise ValueError(f"Unknown decode mode: {decode_mode!r}. Choose 'greedy' or 'beam'.")

    return hyp, log_probs


# ---------------------------------------------------------------------------
# Dictionary helpers
# ---------------------------------------------------------------------------

def load_dict(dict_path: str):
    """
    Load a token dictionary file.

    Expected format (one token per line):
        <token> <integer_id>

    Returns:
        dict{ id -> token_string }
    """
    id2tok = {}
    with open(dict_path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                print(f'[warn] dict line {line_no} skipped (unexpected format): {line!r}',
                      file=sys.stderr)
                continue
            tok, idx = parts[0], int(parts[-1])
            id2tok[idx] = tok
    return id2tok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Prototype CTC decoder for UBCTC network',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model', required=True,
                        help='Path to model checkpoint (.pt)')
    parser.add_argument('--feat', required=True,
                        help='Path to feature file (.npy), shape (T, feat_dim)')
    parser.add_argument('--dict', default=None,
                        help='Token dictionary file ("token id" per line). '
                             'If omitted, raw integer IDs are printed.')
    parser.add_argument('--mode', choices=['greedy', 'beam'], default='greedy',
                        help='CTC decode mode')
    parser.add_argument('--beam', type=int, default=10,
                        help='Beam size for beam-search mode')
    parser.add_argument('--blank', type=int, default=9003,
                        help='Blank label ID  (num_class - 1)')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU device id (-1 for CPU)')
    args = parser.parse_args()

    # --- Device ---
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f'[decode] device : {device}')

    # --- Model ---
    print(f'[decode] loading model : {args.model}')
    model = load_model(args.model, device)
    print(f'[decode] model loaded  (params: '
          f'{sum(p.numel() for p in model.parameters()):,})')

    # --- Dictionary ---
    id2tok = None
    if args.dict:
        id2tok = load_dict(args.dict)
        print(f'[decode] dictionary    : {len(id2tok)} tokens  ({args.dict})')

    # --- Features ---
    print(f'[decode] loading feat  : {args.feat}')
    feat = np.load(args.feat).astype(np.float32)
    if feat.ndim == 1:
        feat = feat[:, np.newaxis]
    print(f'[decode] feat shape    : {feat.shape}  (T={feat.shape[0]}, dim={feat.shape[1]})')

    # --- Decode ---
    mode_str = args.mode + (f' (beam={args.beam})' if args.mode == 'beam' else '')
    print(f'[decode] mode          : {mode_str}')

    hyp, log_probs = decode_utterance(
        model, feat, device,
        decode_mode=args.mode,
        beam_size=args.beam,
        blank_id=args.blank,
    )

    # --- Output ---
    print()
    print(f'hypothesis IDs   : {hyp}')
    if id2tok:
        tokens = [id2tok.get(i, f'<unk:{i}>') for i in hyp]
        print(f'hypothesis text  : {" ".join(tokens)}')
    print(f'decoded length   : {len(hyp)} tokens  |  encoder frames: {log_probs.shape[0]}')


if __name__ == '__main__':
    main()
