"""
分析实车录音的 T60（混响时间）和 SNR（信噪比）分布
用法：python3 analyze_car_audio.py
"""

import os
import glob
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import fftconvolve
from scipy.stats import describe
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------
# 配置：替换为实际路径
# -----------------------------------------------------------------------
AUDIO_DIR   = "PLACEHOLDER_CAR_AUDIO_DIR"   # 实车录音所在目录，如 /data/car_phone_num
AUDIO_GLOB  = "**/*.wav"                     # 递归匹配 wav，按实际后缀改
OUTPUT_CSV  = "car_audio_stats.csv"          # 输出统计结果
# -----------------------------------------------------------------------

SAMPLE_RATE = 16000
FRAME_MS    = 20       # VAD 帧长（ms）
FRAME_LEN   = int(SAMPLE_RATE * FRAME_MS / 1000)


def load_audio(path):
    wav, sr = sf.read(path, always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
    return wav.astype(np.float32)


def estimate_snr_energy(wav):
    """
    基于能量的盲 SNR 估计：
    - 把帧按 RMS 排序，最低 20% 当噪声帧，最高 20% 当语音帧
    - SNR = 10 * log10(speech_rms^2 / noise_rms^2)
    返回 dB，若静音帧能量为 0 返回 None
    """
    frames = [wav[i:i+FRAME_LEN] for i in range(0, len(wav)-FRAME_LEN, FRAME_LEN)]
    rms = np.array([np.sqrt(np.mean(f**2) + 1e-12) for f in frames])
    n = max(1, len(rms) // 5)
    noise_rms   = np.mean(np.sort(rms)[:n])
    speech_rms  = np.mean(np.sort(rms)[-n:])
    if noise_rms < 1e-10:
        return None
    return 20 * np.log10(speech_rms / noise_rms)


def estimate_t60_blind(wav, sr=SAMPLE_RATE):
    """
    盲 T60 估计（基于能量包络衰减斜率）：
    在语音结束后的尾段拟合指数衰减，推算衰减 60dB 的时间。
    精度有限（±50ms 量级），适合做相对比较用。
    返回秒，若信号太短或无明显衰减返回 None
    """
    # 计算短时 RMS 包络
    hop = FRAME_LEN // 2
    rms = np.array([
        np.sqrt(np.mean(wav[i:i+FRAME_LEN]**2) + 1e-12)
        for i in range(0, len(wav)-FRAME_LEN, hop)
    ])
    rms_db = 20 * np.log10(rms)

    # 找峰值后的衰减段
    peak_idx = np.argmax(rms_db)
    tail = rms_db[peak_idx:]
    if len(tail) < 20:
        return None

    # 线性回归拟合衰减斜率（dB/frame）
    x = np.arange(len(tail))
    slope, intercept = np.polyfit(x, tail, 1)
    if slope >= 0:
        return None  # 没有衰减

    # 衰减 60dB 需要多少帧 → 转换为秒
    frames_for_60db = -60.0 / slope
    t60 = frames_for_60db * hop / sr
    # 合理范围过滤
    if t60 < 0.05 or t60 > 2.0:
        return None
    return round(t60, 3)


def analyze_all(audio_dir, pattern):
    paths = sorted(glob.glob(os.path.join(audio_dir, pattern), recursive=True))
    if not paths:
        print(f"[ERROR] 未找到音频：{os.path.join(audio_dir, pattern)}")
        return

    print(f"共找到 {len(paths)} 条音频，开始分析...\n")

    results = []
    for i, path in enumerate(paths, 1):
        try:
            wav = load_audio(path)
            snr  = estimate_snr_energy(wav)
            t60  = estimate_t60_blind(wav)
            dur  = len(wav) / SAMPLE_RATE
            results.append({
                "file": os.path.basename(path),
                "dur_s": round(dur, 2),
                "snr_db": round(snr, 1) if snr is not None else None,
                "t60_s":  t60,
            })
            print(f"[{i:3d}/{len(paths)}] {os.path.basename(path):40s}  "
                  f"dur={dur:.1f}s  SNR={snr:.1f}dB  T60={t60}s")
        except Exception as e:
            print(f"[SKIP] {path}: {e}")

    # 汇总统计
    snr_vals = [r["snr_db"] for r in results if r["snr_db"] is not None]
    t60_vals = [r["t60_s"]  for r in results if r["t60_s"]  is not None]

    print("\n" + "="*60)
    print("SNR 统计 (dB):")
    if snr_vals:
        print(f"  样本数={len(snr_vals)}  mean={np.mean(snr_vals):.1f}  "
              f"std={np.std(snr_vals):.1f}  "
              f"min={np.min(snr_vals):.1f}  max={np.max(snr_vals):.1f}")
        print(f"  p25={np.percentile(snr_vals,25):.1f}  "
              f"p50={np.percentile(snr_vals,50):.1f}  "
              f"p75={np.percentile(snr_vals,75):.1f}")

    print("\nT60 统计 (s):")
    if t60_vals:
        print(f"  样本数={len(t60_vals)}  mean={np.mean(t60_vals):.3f}  "
              f"std={np.std(t60_vals):.3f}  "
              f"min={np.min(t60_vals):.3f}  max={np.max(t60_vals):.3f}")
        print(f"  p25={np.percentile(t60_vals,25):.3f}  "
              f"p50={np.percentile(t60_vals,50):.3f}  "
              f"p75={np.percentile(t60_vals,75):.3f}")
    print("="*60)

    # 写 CSV
    with open(OUTPUT_CSV, "w") as f:
        f.write("file,dur_s,snr_db,t60_s\n")
        for r in results:
            f.write(f"{r['file']},{r['dur_s']},{r['snr_db']},{r['t60_s']}\n")
    print(f"\n详细结果已写入：{OUTPUT_CSV}")


if __name__ == "__main__":
    analyze_all(AUDIO_DIR, AUDIO_GLOB)
