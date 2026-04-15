#!/usr/bin/env python3
"""
End-to-end decoder: WAV audio -> phone / word sequence

Feature pipeline (对齐 9_fea_merge.pl / HTK config.fea.16K_offCMN_PowerFB40):

    WAV (16kHz)
      │
      │  raw_fea:  25ms Hamming 窗, 10ms 帧移
      │            功率谱 → 40维 mel 滤波器组 → log
      ▼
    fb40  (T, 40)   ← 静态特征, 无 Δ/ΔΔ
      │
      │  offCMN:  整句均值归零  (offline per-utterance CMN)
      ▼
    cmvn-normalized fb40
      │
      │  fea.norm (可选, 内网文件):  y = (x - global_mean) * inv_std
      ▼
    UBCTC Encoder
      ConcatFrLayer(4) → LSTMP×2 → UBLSTMP → Conv2d×2
      │
      ▼
    Conv2d(256→9004) + log_softmax
      │
      ▼
    CTC greedy / prefix-beam decode
      │
      ▼
    phone / word 序列

说明:
  - 1_dnnfa.pl 用的是 fb72 (24维+Δ+ΔΔ), 那是生成 phone 对齐标注用的,
    不是 UBCTC 的训练特征.
  - UBCTC 训练特征由 9_fea_merge.pl 提取: raw_fea config3(fb40), 无 Δ/ΔΔ.
  - offCMN = 每条语音整句减均值, 不依赖 fea.norm, 本地可复现.
  - fea.norm 存于内网, 不可用时跳过全局归一化 (识别率会下降).

Usage:
    python decode_wav.py \\
        --model  /path/to/checkpoint.pt \\
        --wav    /path/to/audio.wav \\
        [--norm  /path/to/lib_fb40/fea.norm] \\
        [--dict  /path/to/tokens.txt] \\
        [--mode  greedy|beam]  [--beam 10] \\
        [--gpu   0]

字典文件格式 (每行):
    <token>  <整数id>

fea.norm 格式 (QN 格式):
    vec <N>
    <mean_0>        ← 全局均值
    ...
    vec <N>
    <inv_std_0>     ← 1 / sqrt(var), 即逆标准差
    ...
"""

import sys
import os
import math
import argparse

import numpy as np
import torch
import torch.nn.functional as F

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from net_ubctc import Ubctc
from asr.data import clip_mask


# ─────────────────────────────────────────────────────────────
# 1.  特征提取  (对齐 config.fea.16K_offCMN_PowerFB40)
# ─────────────────────────────────────────────────────────────

def extract_fb40(wav_path: str, sr: int = 16000) -> np.ndarray:
    """
    提取 40维 log mel-filterbank 特征, 对齐 HTK config.fea.16K_offCMN_PowerFB40:

        - 采样率: 16kHz (16K)
        - 窗函数: Hamming, 25ms (400 samples @ 16kHz)
        - 帧移:   10ms  (160 samples @ 16kHz)
        - 特征:   功率谱 → 40维 mel 滤波器组 → log  (PowerFB40)
        - 无 Δ/ΔΔ

    offCMN 步骤在 apply_utt_cmn() 中单独完成.

    Returns:
        fbank: (T, 40) float32 numpy array
    """
    WIN_SAMPLES = int(sr * 0.025)   # 25ms
    HOP_SAMPLES = int(sr * 0.010)   # 10ms
    N_MELS = 40

    # ── 读取音频 ──
    try:
        import torchaudio
        wav, orig_sr = torchaudio.load(wav_path)           # (C, N)
        if orig_sr != sr:
            wav = torchaudio.functional.resample(wav, orig_sr, sr)
        wav = wav.mean(dim=0)                               # mono (N,)

        # 功率谱 mel 滤波器组 (HTK 默认用 Hamming)
        mel_tf = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=WIN_SAMPLES,
            win_length=WIN_SAMPLES,
            hop_length=HOP_SAMPLES,
            n_mels=N_MELS,
            window_fn=torch.hamming_window,   # ← Hamming, 对齐 HTK
            power=2.0,                        # ← 功率谱 (PowerFB)
            norm=None,
            mel_scale='htk',                  # ← HTK mel 刻度
        )
        mel = mel_tf(wav)                                   # (40, T)
        fbank = torch.log(mel + 1e-7).T.numpy().astype(np.float32)  # (T, 40)

    except ImportError:
        try:
            import librosa
        except ImportError:
            raise ImportError(
                "需要 torchaudio 或 librosa:\n"
                "  pip install torchaudio\n"
                "  pip install librosa soundfile"
            )
        wav, orig_sr = librosa.load(wav_path, sr=sr, mono=True)
        mel = librosa.feature.melspectrogram(
            y=wav, sr=sr,
            n_fft=WIN_SAMPLES, win_length=WIN_SAMPLES, hop_length=HOP_SAMPLES,
            n_mels=N_MELS,
            window='hamming',   # ← Hamming
            power=2.0,          # ← 功率谱
            htk=True,           # ← HTK mel 刻度
        )
        fbank = np.log(mel + 1e-7).T.astype(np.float32)    # (T, 40)

    return fbank


def apply_utt_cmn(fbank: np.ndarray) -> np.ndarray:
    """
    offCMN: 整句均值归零 (offline per-utterance Cepstral Mean Normalization).

    对应 HTK config 中 'offCMN': 每个特征维度减去该维度在整句上的均值.
    注意: 这里只减均值, 不除标准差 (CMN 非 CMVN).

    Args:
        fbank: (T, D) float32

    Returns:
        (T, D) float32, 每列均值为 0
    """
    return fbank - fbank.mean(axis=0, keepdims=True)


# ─────────────────────────────────────────────────────────────
# 2.  全局归一化  (fea.norm, 内网文件, 可选)
# ─────────────────────────────────────────────────────────────

def load_fea_norm(norm_path: str):
    """
    解析 QN 格式的 fea.norm 文件.

    格式:
        vec <N>
        <mean_0>
        ...
        vec <N>
        <inv_std_0>    ← 1/sqrt(var)
        ...

    Returns:
        mean    : (N,) float32
        inv_std : (N,) float32
    """
    means, istds = [], []
    cur = None
    with open(norm_path, 'r') as fh:
        for line in fh:
            line = line.strip()
            if line.startswith('vec'):
                cur = means if not means else istds
            elif line and cur is not None:
                cur.append(float(line))
    return np.array(means, dtype=np.float32), np.array(istds, dtype=np.float32)


def apply_global_norm(fbank: np.ndarray,
                      mean: np.ndarray, inv_std: np.ndarray) -> np.ndarray:
    """
    全局均值-方差归一化: y = (x - mean) * inv_std

    fea.norm 可能是在 LFR-4 拼帧后的 160维特征上计算的.
    若 norm_dim=160, feat_dim=40, 取前 40 维近似使用.
    """
    feat_dim = fbank.shape[1]
    norm_dim = mean.shape[0]

    if norm_dim == feat_dim:
        return (fbank - mean) * inv_std

    if norm_dim > feat_dim and norm_dim % feat_dim == 0:
        print(f'[norm] norm_dim={norm_dim} > feat_dim={feat_dim}; '
              f'使用前 {feat_dim} 维 (LFR norm -> frame-level approx)')
        return (fbank - mean[:feat_dim]) * inv_std[:feat_dim]

    print(f'[warn] norm_dim={norm_dim} 与 feat_dim={feat_dim} 不匹配, 跳过全局归一化')
    return fbank


# ─────────────────────────────────────────────────────────────
# 3.  Tensor 准备
# ─────────────────────────────────────────────────────────────

def feat_to_tensor(feat: np.ndarray, device: torch.device):
    """
    将 (T, D) numpy array 转换为编码器输入格式.

    ConcatFrLayer(nmod=4) 要求 T 是 4 的整数倍, 自动零填充.

    Returns:
        x        : (1, D, 1, T_pad)   CNN 格式
        rnn_mask : (T_pad, 1, 1)      1=有效帧, 0=填充帧
        T_orig   : 原始帧数 (未填充)
    """
    T_orig, D = feat.shape
    T_pad = math.ceil(T_orig / 4) * 4

    if T_pad > T_orig:
        pad = np.zeros((T_pad - T_orig, D), dtype=np.float32)
        feat = np.concatenate([feat, pad], axis=0)

    x = torch.from_numpy(feat).float().T.unsqueeze(0).unsqueeze(2)  # (1, D, 1, T_pad)
    rnn_mask = torch.zeros(T_pad, 1, 1, dtype=torch.float32)
    rnn_mask[:T_orig] = 1.0

    return x.to(device), rnn_mask.to(device), T_orig


# ─────────────────────────────────────────────────────────────
# 4.  模型加载
# ─────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: torch.device) -> Ubctc:
    model = Ubctc()
    raw = torch.load(ckpt_path, map_location='cpu')
    if isinstance(raw, dict):
        state = raw.get('model', raw.get('state_dict', raw.get('net', raw)))
    else:
        state = raw
    # 去掉 DataParallel 的 'module.' 前缀
    state = {k.replace('module.', '', 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


# ─────────────────────────────────────────────────────────────
# 5.  CTC 解码
# ─────────────────────────────────────────────────────────────

def ctc_greedy(log_probs: torch.Tensor, blank: int):
    """Greedy CTC: argmax → 去 blank → 去连续重复."""
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


def _lse(a: float, b: float) -> float:
    """log(exp(a) + exp(b)), 数值稳定."""
    NEG_INF = float('-inf')
    if a == NEG_INF: return b
    if b == NEG_INF: return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def ctc_beam(log_probs: torch.Tensor, blank: int, beam_size: int = 10):
    """CTC prefix beam search."""
    NEG_INF = float('-inf')
    lp = log_probs.cpu().float().numpy()   # (T, V)
    T  = lp.shape[0]

    beams = {(): (0.0, NEG_INF)}           # prefix -> (log_pb, log_pnb)

    for t in range(T):
        lp_t = lp[t]
        top  = np.argsort(lp_t)[::-1][:max(beam_size * 2, 30)]
        new: dict = {}

        def add(prefix, pb, pnb):
            ob, onb = new.get(prefix, (NEG_INF, NEG_INF))
            new[prefix] = (_lse(ob, pb), _lse(onb, pnb))

        for prefix, (pb, pnb) in beams.items():
            p_tot = _lse(pb, pnb)
            # 延伸 blank → 同 prefix, 更新 prob_b
            add(prefix, p_tot + lp_t[blank], NEG_INF)
            # 延伸非 blank
            for c in top:
                if c == blank:
                    continue
                new_p = prefix + (int(c),)
                if prefix and prefix[-1] == c:
                    # 重复 label: 只能来自以 blank 结尾的路径
                    add(new_p, NEG_INF, pb + lp_t[c])
                else:
                    add(new_p, NEG_INF, p_tot + lp_t[c])

        beams = dict(
            sorted(new.items(),
                   key=lambda kv: _lse(kv[1][0], kv[1][1]),
                   reverse=True)[:beam_size]
        )

    return list(max(beams, key=lambda p: _lse(beams[p][0], beams[p][1])))


# ─────────────────────────────────────────────────────────────
# 6.  完整推理函数
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def decode(model: Ubctc, feat: np.ndarray, device: torch.device,
           mode: str = 'greedy', beam_size: int = 10, blank: int = 9003):
    """
    一条语音的完整解码:
      feat (T,40) -> encoder -> classification -> CTC decode -> token IDs

    Returns:
        hyp       : List[int]  解码结果 token ID 序列
        log_probs : (T', 9004) tensor  帧级 log 概率
    """
    x, rnn_mask, _ = feat_to_tensor(feat, device)
    meta = {
        "rnn_mask" : rnn_mask,
        "att_label": torch.zeros(1, 1, dtype=torch.long, device=device),
    }

    enc = model.encoder
    meta["rnn_mask"] = clip_mask(meta["rnn_mask"], enc.concat_fr.nmod, 0)

    enc_out = enc(x, meta)                              # (1, 256, 1, T')
    b, d, f, t = enc_out.shape
    flat   = enc_out.permute(2, 1, 3, 0).reshape(1, d, 1, t * b)
    logit  = model.classification(flat).squeeze().permute(1, 0)   # (T', 9004)
    log_probs = F.log_softmax(logit, dim=-1)

    hyp = ctc_greedy(log_probs, blank) if mode == 'greedy' \
          else ctc_beam(log_probs, blank, beam_size)
    return hyp, log_probs


# ─────────────────────────────────────────────────────────────
# 7.  State ID 字典  (id -> triphone state 标签)
# ─────────────────────────────────────────────────────────────

def load_state_dict(path: str) -> dict:
    """
    加载 state-label 字典, 返回 {id -> triphone_label 字符串}.

    格式 (每行):
        <triphone_label>  <整数id>
    例:
        sil 0
        a-b+c 1
        a-b+c_s2 2
    """
    id2lab = {}
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) >= 2:
                id2lab[int(parts[-1])] = parts[0]
    return id2lab


# ─────────────────────────────────────────────────────────────
# 8.  Triphone state -> phone 转换
# ─────────────────────────────────────────────────────────────

def triphone_to_phone(label: str) -> str:
    """
    从 triphone state 标签中提取中心 phone.

    HTK triphone 格式:  left-CENTER+right  或  left-CENTER+right_sN
    本函数提取 CENTER 部分, 若解析失败则返回原始标签.

    示例:
        'a-b+c'      -> 'b'
        'a-b+c_s2'   -> 'b'
        'sil'        -> 'sil'
        'sil_s2'     -> 'sil'
        'sp'         -> 'sp'
    """
    # 去掉 HTK state 后缀 _sN 或 [N]
    label = label.split('_s')[0].split('[')[0]

    if '-' in label and '+' in label:
        # left-CENTER+right
        center = label.split('-', 1)[1].split('+', 1)[0]
        return center
    elif '-' in label:
        # left-CENTER (无右上下文)
        return label.split('-', 1)[1]
    elif '+' in label:
        # CENTER+right (无左上下文)
        return label.split('+', 1)[0]
    else:
        # 单音素 (sil, sp, monophone)
        return label


def state_ids_to_phones(hyp_ids: list, id2lab: dict) -> list:
    """
    将 CTC 解码的 state ID 序列转换为去重后的 phone 序列.

    步骤:
      1. state ID -> triphone label (via id2lab)
      2. triphone label -> center phone (via triphone_to_phone)
      3. 合并连续相同 phone (CTC 已去重, 此处处理跨 state 的同 phone)
      4. 过滤静音 phone (sil, sp)

    Args:
        hyp_ids : CTC 解码输出的 state ID 列表
        id2lab  : {id -> triphone_label} 字典

    Returns:
        phones  : List[str]  phone 序列 (已去重, 已过滤静音)
    """
    SILENCE = {'sil', 'sp', 'SIL', 'SP', 'silence'}

    phones, prev = [], None
    for sid in hyp_ids:
        label = id2lab.get(sid, f'<unk:{sid}>')
        phone = triphone_to_phone(label)
        if phone in SILENCE:
            prev = phone
            continue
        if phone != prev:
            phones.append(phone)
        prev = phone
    return phones


# ─────────────────────────────────────────────────────────────
# 9.  发音词典  (word -> phone sequence)  &  phone->word 匹配
# ─────────────────────────────────────────────────────────────

def load_lexicon(path: str) -> dict:
    """
    加载发音词典, 返回 {word -> [phone_list, ...]}.

    格式 (每行, word 与 phones 之间用 Tab 分隔):
        word<TAB>ph1 ph2 ph3

    同一个词可以有多个发音 (多行), 全部保留.

    Returns:
        lex : {word_str -> list of phone-tuple}
    """
    lex: dict = {}
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.rstrip('\n')
            if not line or line.startswith('#'):
                continue
            if '\t' in line:
                word, pron = line.split('\t', 1)
            else:
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
                word, pron = parts
            phones = tuple(pron.strip().split())
            lex.setdefault(word, []).append(phones)
    return lex


def _build_phone2words(lex: dict) -> dict:
    """
    构建反向索引: phone_tuple -> [word, ...].
    用于 phone 序列 → word 序列的贪婪匹配.
    """
    p2w: dict = {}
    for word, prons in lex.items():
        for pron in prons:
            p2w.setdefault(pron, []).append(word)
    return p2w


def phones_to_words(phones: list, lex: dict) -> list:
    """
    贪婪最长匹配: 将 phone 序列转换为 word 序列.

    从左到右, 每次尝试最长能匹配到词典的 phone 子串.
    若某段无法匹配, 以 <ph1+ph2+...> 形式输出原始 phones.

    Args:
        phones : List[str]  phone 序列
        lex    : {word -> [phone_tuple, ...]}

    Returns:
        words  : List[str]  word 序列
    """
    p2w = _build_phone2words(lex)
    words = []
    i = 0
    # 最长 phone 串长度 (搜索上界)
    max_pron_len = max((len(p) for prons in lex.values() for p in prons), default=1)

    while i < len(phones):
        matched = False
        # 从最长到最短尝试匹配
        for length in range(min(max_pron_len, len(phones) - i), 0, -1):
            chunk = tuple(phones[i:i + length])
            if chunk in p2w:
                words.append(p2w[chunk][0])   # 取第一个匹配词
                i += length
                matched = True
                break
        if not matched:
            # 无法匹配, 输出原始 phone 并跳过
            words.append(f'<{phones[i]}>')
            i += 1

    return words


# ─────────────────────────────────────────────────────────────
# 10.  MLF 读取  (HTK Master Label File)
# ─────────────────────────────────────────────────────────────

def load_mlf(mlf_path: str, level: str = 'phone') -> dict:
    """
    解析 HTK MLF 文件, 返回 {utt_id -> label_list}.

    支持两种常见格式:
      带时间戳: "start end label [word]"  (单位 100ns)
      无时间戳: "label"

    Args:
        mlf_path : MLF 文件路径
        level    : 'phone' 直接返回 label 列
                   'word'  返回第4列 word (若存在), 否则用 label 列

    Returns:
        {utt_id: [label, ...]}
        utt_id 取自 MLF 中 "*/utt_id.lab" 的 utt_id 部分 (无后缀)
    """
    SILENCE = {'sil', 'sp', 'SIL', 'SP', 'silence', 'silb', 'sile'}
    result = {}
    cur_id = None
    cur_labels = []

    with open(mlf_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.rstrip('\n').strip()
            if not line or line == '#!MLF!#':
                continue
            if line.startswith('"'):
                # 新语句头: "*/utt_id.lab" 或 "utt_id.lab"
                if cur_id is not None:
                    result[cur_id] = cur_labels
                # 提取 utt_id: 去掉引号、路径前缀、.lab 后缀
                name = line.strip('"').replace('\\', '/').split('/')[-1]
                cur_id = os.path.splitext(name)[0]
                cur_labels = []
            elif line == '.':
                # 语句结束
                if cur_id is not None:
                    result[cur_id] = cur_labels
                cur_id = None
                cur_labels = []
            else:
                parts = line.split()
                # 判断格式: 带时间戳时 parts[0] 是纯数字
                if len(parts) >= 3 and parts[0].lstrip('-').isdigit():
                    # 格式: start end label [word ...]
                    label = parts[2]
                    word  = parts[3] if len(parts) >= 4 else label
                elif len(parts) >= 1:
                    # 格式: label
                    label = parts[0]
                    word  = label
                else:
                    continue

                token = word if level == 'word' else label
                if token not in SILENCE:
                    cur_labels.append(token)

    # 文件末尾没有 '.' 结束符时也保存
    if cur_id is not None and cur_labels:
        result[cur_id] = cur_labels

    return result


# ─────────────────────────────────────────────────────────────
# 11.  编辑距离  &  错误率计算
# ─────────────────────────────────────────────────────────────

def edit_distance(ref: list, hyp: list) -> int:
    """
    计算两个序列之间的 Levenshtein 编辑距离.
    (对应 asr/layers/acc.py 中的 edit_dist)
    """
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            temp = dp[j]
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]


def error_stats(ref: list, hyp: list) -> dict:
    """
    返回单条语音的对齐统计量 (用于累加全局 PER/WER).

    Returns:
        {'dist': int, 'ref_len': int}
    """
    return {'dist': edit_distance(ref, hyp), 'ref_len': len(ref)}


def format_error_rate(total_dist: int, total_ref: int) -> str:
    if total_ref == 0:
        return 'N/A (ref 为空)'
    rate = total_dist / total_ref * 100
    return f'{rate:.2f}%  ({total_dist}/{total_ref})'


# ─────────────────────────────────────────────────────────────
# 12.  单条语音推理  (供批量循环调用)
# ─────────────────────────────────────────────────────────────

def decode_one(wav_path: str, model: Ubctc, device: torch.device,
               args, norm_params, id2lab, lex):
    """
    对单条 wav 文件完成特征提取 + CTC 解码 + 后处理.

    Returns:
        phones : List[str]  phone 序列 (需要 --phone)
        words  : List[str]  word  序列 (需要 --lex),  否则为 None
        hyp_ids: List[int]  原始 state ID 序列
    """
    fbank = extract_fb40(wav_path, sr=args.sr)
    fbank = apply_utt_cmn(fbank)

    if norm_params is not None:
        mean, inv_std = norm_params
        fbank = apply_global_norm(fbank, mean, inv_std)

    hyp_ids, _ = decode(
        model, fbank, device,
        mode=args.mode, beam_size=args.beam, blank=args.blank,
    )

    phones = state_ids_to_phones(hyp_ids, id2lab) if (args.phone and id2lab) else []
    words  = phones_to_words(phones, lex)          if (phones and lex)        else None

    return phones, words, hyp_ids


# ─────────────────────────────────────────────────────────────
# 13.  命令行入口
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='WAV -> phone/word  via UBCTC (fb40 offCMN + CTC)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── 输入 ──
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument('--wav',     help='单条输入音频 (.wav)')
    grp.add_argument('--wav_dir', help='批量输入: 文件夹路径, 递归搜索 *.wav')

    # ── 模型 & 资源 ──
    p.add_argument('--model', required=True, help='UBCTC checkpoint (.pt)')
    p.add_argument('--norm',  default=None,
                   help='全局归一化文件 lib_fb40/fea.norm (内网, 可选)')
    p.add_argument('--dict',  default=None,
                   help='state-label 字典 ("triphone_label id" 每行)')
    p.add_argument('--lex',   default=None,
                   help='发音词典 ("word<TAB>ph1 ph2 ..." 每行)')

    # ── 解码参数 ──
    p.add_argument('--mode',  choices=['greedy', 'beam'], default='greedy')
    p.add_argument('--beam',  type=int, default=10)
    p.add_argument('--blank', type=int, default=9003)
    p.add_argument('--sr',    type=int, default=16000)
    p.add_argument('--gpu',   type=int, default=-1)

    # ── 后处理 ──
    p.add_argument('--phone', action='store_true',
                   help='triphone state -> center phone (需要 --dict)')

    # ── MLF 对比 ──
    p.add_argument('--mlf',   default=None,
                   help='参考标注 MLF 文件 (HTK 格式). '
                        '提供后计算 PER (--phone) 或 WER (--lex).')
    p.add_argument('--mlf_level', choices=['phone', 'word'], default='phone',
                   help='MLF 中读取 phone 列还是 word 列进行对比 (默认 phone)')

    # ── 输出 ──
    p.add_argument('--output', default=None,
                   help='将逐句解码结果保存到文件 (可选)')

    args = p.parse_args()

    # ── 参数联动检查 ──
    if args.lex and not args.phone:
        print('[warn] --lex 需要 --phone, 自动开启')
        args.phone = True
    if args.phone and not args.dict:
        p.error('--phone 需要 --dict')
    if args.mlf and not (args.phone or args.dict):
        p.error('--mlf 对比需要至少提供 --dict (用于 phone 级对比)')

    # ── 设备 ──
    device = torch.device(
        f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu'
    )
    print(f'[decode] device   : {device}')

    # ── 模型 ──
    print(f'[decode] model    : {args.model}')
    model = load_model(args.model, device)
    print(f'[decode]            {sum(v.numel() for v in model.parameters()):,} parameters')

    # ── 资源文件 ──
    norm_params = None
    if args.norm:
        if os.path.isfile(args.norm):
            norm_params = load_fea_norm(args.norm)
            print(f'[decode] fea.norm : {args.norm}')
        else:
            print(f'[warn] fea.norm 不存在, 跳过: {args.norm}')

    id2lab = load_state_dict(args.dict) if args.dict else None
    if id2lab:
        print(f'[decode] dict     : {len(id2lab)} states  ({args.dict})')

    lex = load_lexicon(args.lex) if args.lex else None
    if lex:
        print(f'[decode] lexicon  : {len(lex)} words  ({args.lex})')

    # ── 参考 MLF ──
    ref_mlf = None
    if args.mlf:
        ref_mlf = load_mlf(args.mlf, level=args.mlf_level)
        print(f'[decode] mlf      : {len(ref_mlf)} utts  ({args.mlf})')

    # ── 收集待解码的 wav 文件 ──
    if args.wav:
        wav_list = [args.wav]
    else:
        wav_list = sorted(
            os.path.join(root, f)
            for root, _, files in os.walk(args.wav_dir)
            for f in files if f.lower().endswith('.wav')
        )
        print(f'[decode] wav_dir  : {args.wav_dir}  ({len(wav_list)} files)')

    if not wav_list:
        print('[error] 未找到任何 .wav 文件')
        return

    # ── 输出文件句柄 ──
    out_fh = open(args.output, 'w', encoding='utf-8') if args.output else None

    # ── 全局统计 ──
    total_phone_dist, total_phone_ref = 0, 0
    total_word_dist,  total_word_ref  = 0, 0
    n_done, n_err = 0, 0

    print()
    print('=' * 72)

    for wav_path in wav_list:
        utt_id = os.path.splitext(os.path.basename(wav_path))[0]
        try:
            phones, words, hyp_ids = decode_one(
                wav_path, model, device, args, norm_params, id2lab, lex
            )
        except Exception as e:
            print(f'[error] {utt_id}: {e}')
            n_err += 1
            continue

        n_done += 1

        # ── 逐句打印 ──
        print(f'UTT  : {utt_id}')
        if args.phone and phones:
            print(f'HYP  : {" ".join(phones)}')
        elif not args.phone:
            print(f'HYP  : {hyp_ids}')

        # ── MLF 对比 ──
        if ref_mlf is not None:
            ref_seq = ref_mlf.get(utt_id)
            if ref_seq is None:
                print(f'REF  : [未找到 MLF 条目: {utt_id}]')
            else:
                # Phone 级对比
                if args.phone and phones:
                    stats = error_stats(ref_seq, phones)
                    total_phone_dist += stats['dist']
                    total_phone_ref  += stats['ref_len']
                    per = stats['dist'] / max(stats['ref_len'], 1) * 100
                    print(f'REF  : {" ".join(ref_seq)}')
                    print(f'PER  : {per:.1f}%  (err={stats["dist"]}, ref={stats["ref_len"]})')

                # Word 级对比
                if words is not None:
                    # MLF word 列 (若 mlf_level=word 则 ref_seq 已是 word)
                    word_ref = ref_mlf.get(utt_id + '_w', ref_seq) \
                               if args.mlf_level == 'phone' else ref_seq
                    stats_w = error_stats(word_ref, words)
                    total_word_dist += stats_w['dist']
                    total_word_ref  += stats_w['ref_len']
                    wer = stats_w['dist'] / max(stats_w['ref_len'], 1) * 100
                    print(f'WRD  : {" ".join(words)}')
                    print(f'WER  : {wer:.1f}%  (err={stats_w["dist"]}, ref={stats_w["ref_len"]})')

        # ── 写入输出文件 ──
        if out_fh:
            seq = phones if (args.phone and phones) else [str(i) for i in hyp_ids]
            out_fh.write(f'{utt_id}\t{" ".join(seq)}\n')

        print('-' * 72)

    # ── 汇总 ──
    print()
    print('=' * 72)
    print(f'[summary] 完成: {n_done}  失败: {n_err}  共: {len(wav_list)}')
    if ref_mlf and args.phone and total_phone_ref > 0:
        print(f'[summary] PER (phone) : {format_error_rate(total_phone_dist, total_phone_ref)}')
    if ref_mlf and lex and total_word_ref > 0:
        print(f'[summary] WER (word)  : {format_error_rate(total_word_dist, total_word_ref)}')
    print('=' * 72)

    if out_fh:
        out_fh.close()
        print(f'[decode] 结果已保存: {args.output}')


if __name__ == '__main__':
    main()
