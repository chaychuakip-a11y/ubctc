#!/usr/bin/env python3
"""
从 pfile 直接解码, 绕过 WAV 特征提取, 用于验证模型是否正常.

Usage:
    python decode_pfile.py \\
        --model   /path/to/model.iter13.part4 \\
        --fea     /path/to/fea.pfile.0 \\
        --norm    /path/to/fea.norm \\
        [--lab    /path/to/lab.pfile.0]   # 有标签则计算准确率 \\
        [--nsent  100]                    # 只解前 N 句, 默认全部 \\
        [--blank  9003] \\
        [--gpu    0]
"""

import sys
import os
import struct
import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from net_ubctc import Ubctc
from asr.data.pfile_reader import Pfileinfo, Normfile


# ─────────────────────────────────────────────────────────────
# 读单条句子
# ─────────────────────────────────────────────────────────────

def read_one_sent(pfileinfo: Pfileinfo, fp, sent_idx: int) -> np.ndarray:
    """
    从已打开的二进制文件句柄读取第 sent_idx 条句子的特征.
    返回 (T, D) float32 numpy array.
    """
    seqid, start_frame, num_frames = pfileinfo.seq_info[sent_idx]

    if pfileinfo.data_format[2] == 'f':
        dtype = np.dtype("float32")
    else:
        dtype = np.dtype("int32")

    frame_len = pfileinfo.frame_length
    offset = pfileinfo.header_size + frame_len * dtype.itemsize * start_frame
    fp.seek(offset)
    raw = fp.read(frame_len * dtype.itemsize * num_frames)
    arr = np.frombuffer(raw, dtype=dtype).copy()
    arr = arr.byteswap()
    arr = arr.reshape(num_frames, frame_len)
    arr = arr[:, pfileinfo.real_data_start:]   # 去掉 seqid/frameid 列
    return arr.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# 模型加载
# ─────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: torch.device) -> Ubctc:
    model = Ubctc()
    raw = torch.load(ckpt_path, map_location='cpu')
    if isinstance(raw, dict):
        state = raw.get('model', raw.get('state_dict', raw.get('net', raw)))
    else:
        state = raw
    state = {k.replace('module.', '', 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


# ─────────────────────────────────────────────────────────────
# Tensor 准备  (对齐 PfileDataset.__pad_nmod / rnn_mask 构建)
# ─────────────────────────────────────────────────────────────

def feat_to_tensor(feat: np.ndarray, device: torch.device):
    """
    feat: (T, D) float32
    返回 x (1,D,1,T_pad), rnn_mask (T_pad,1,1), T_orig
    """
    T_orig, D = feat.shape
    nmod = 4   # ConcatFrLayer(4)
    T_pad = math.ceil(T_orig / nmod) * nmod

    if T_pad > T_orig:
        pad = np.zeros((T_pad - T_orig, D), dtype=np.float32)
        feat = np.concatenate([feat, pad], axis=0)

    # (T,D) -> (D,T) -> (1,D,1,T)
    x = torch.from_numpy(feat).float().T.unsqueeze(0).unsqueeze(2)

    # rnn_mask: (T_pad, 1, 1)  —— 与 PfileDataset 保持一致
    rnn_mask = torch.zeros(T_pad, 1, 1, dtype=torch.float32)
    rnn_mask[:T_orig] = 1.0

    return x.to(device), rnn_mask.to(device), T_orig


# ─────────────────────────────────────────────────────────────
# CTC greedy
# ─────────────────────────────────────────────────────────────

def ctc_greedy(log_probs: torch.Tensor, blank: int):
    ids = torch.argmax(log_probs, dim=1).tolist()
    out, prev = [], -1
    for i in ids:
        if i == blank:
            prev = blank
            continue
        if i != prev:
            out.append(i)
        prev = i
    return out


# ─────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='pfile 直接解码 (跳过 WAV 特征提取)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--model',  required=True)
    p.add_argument('--fea',    required=True, help='fea.pfile.X')
    p.add_argument('--norm',   required=True, help='fea.norm')
    p.add_argument('--lab',    default=None,  help='lab.pfile.X (可选, 计算acc)')
    p.add_argument('--nsent',  type=int, default=0, help='解前N句, 0=全部')
    p.add_argument('--blank',  type=int, default=9003)
    p.add_argument('--gpu',    type=int, default=-1)
    args = p.parse_args()

    device = torch.device(
        f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu'
    )
    print(f'[decode] device : {device}')

    # 模型
    model = load_model(args.model, device)
    print(f'[decode] model  : {args.model}')
    print(f'[decode]          {sum(v.numel() for v in model.parameters()):,} params')

    # norm
    norm = Normfile(args.norm)
    mean = torch.from_numpy(norm.mean).to(device).reshape(1, -1, 1, 1)  # (1,D,1,1)
    var  = torch.from_numpy(norm.var ).to(device).reshape(1, -1, 1, 1)

    # pfile
    fea_info = Pfileinfo(args.fea)
    fea_fp   = open(args.fea, 'rb')
    lab_info = Pfileinfo(args.lab) if args.lab else None
    lab_fp   = open(args.lab, 'rb') if args.lab else None

    nsent = args.nsent if args.nsent > 0 else fea_info.num_sentences
    nsent = min(nsent, fea_info.num_sentences)
    print(f'[decode] fea    : {args.fea}  ({fea_info.num_sentences} sents, dim={fea_info.dim_features})')
    print(f'[decode] 解码前 {nsent} 句')
    print('=' * 60)

    n_correct = n_total = 0

    for i in range(nsent):
        feat = read_one_sent(fea_info, fea_fp, i)   # (T, D)

        # 归一化 (与训练完全一致)
        x, rnn_mask, T_orig = feat_to_tensor(feat, device)
        x = (x - mean) * var

        meta = {
            "rnn_mask" : rnn_mask,
            "att_label": torch.zeros(1, 1, dtype=torch.long, device=device),
        }

        with torch.no_grad():
            _, logit = model(x, meta)   # (T', 9004)

        if logit.dim() == 3:
            logit = logit.squeeze(0)
        if logit.shape[0] == 9004:
            logit = logit.transpose(0, 1)

        log_probs = F.log_softmax(logit, dim=-1)
        hyp = ctc_greedy(log_probs, args.blank)

        # 读参考标签
        ref = None
        if lab_info is not None:
            lab = read_one_sent(lab_info, lab_fp, i)   # (T, 1) int32
            ref = lab[:, 0].astype(int).tolist()
            ref = [r for r in ref if r >= 0]           # 去掉 padding(-1)

        # 打印
        top5 = torch.topk(log_probs.mean(0), 5)
        top5_ids   = top5.indices.tolist()
        top5_probs = top5.values.tolist()
        print(f'[{i:4d}] T={T_orig:4d}  hyp={hyp[:20]}')
        print(f'       top5_avg_logp: {list(zip(top5_ids, [f"{v:.2f}" for v in top5_probs]))}')
        if ref is not None:
            match = (hyp == ref)
            n_total   += 1
            n_correct += int(match)
            print(f'       ref={ref[:20]}  {"OK" if match else "ERR"}')
        print()

    fea_fp.close()
    if lab_fp: lab_fp.close()

    print('=' * 60)
    if n_total > 0:
        print(f'[summary] SentAcc = {n_correct}/{n_total} = {n_correct/n_total*100:.1f}%')


if __name__ == '__main__':
    main()
