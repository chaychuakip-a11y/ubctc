#!/usr/bin/env python3
"""
gen_tts_kr_number.py — Multi-voice multi-speed Korean TTS for phone number ASR.

Edge TTS only has 3 Korean voices, so diversity comes from:
  1. 3 voices × 5 SSML rates = 15 combinations per utterance
  2. 4 break-pattern templates (grouped / no-break / irregular / short-pause)
  3. Optional sox time-stretch post-processing (0.9 / 1.1 / 1.2×) for extreme speeds

Input file format (one per line):
    <utt_id> <text>
  text can be Arabic digits ("010 1234 5678") or Korean words
  ("공일공 일이삼사 오육칠팔").

Output:
  <out_dir>/wav/<utt_id>.wav       (16 kHz mono PCM)
  <out_dir>/manifest.txt           (utt_id  wav_path  text)

Usage:
  python gen_tts_kr_number.py input.txt out_dir/ [--variants N] [--stretch]
"""

import argparse
import asyncio
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import edge_tts
except ImportError:
    sys.exit("pip install edge-tts")

# ─── constants ───────────────────────────────────────────────────────────────

VOICES = [
    "ko-KR-SunHiNeural",           # Female
    "ko-KR-InJoonNeural",          # Male
    "ko-KR-HyunsuMultilingualNeural",  # Male (multilingual, more natural)
]

# edge-tts rate strings; weights: slow 1 : normal 2 : fast 2 : very-fast 1
RATES = ["-20%", "+0%", "+0%", "+15%", "+30%"]

# Extra sox time-stretch factors applied *after* TTS generation.
# These create acoustic speed beyond SSML limits and change coarticulation.
STRETCH_FACTORS = [0.9, 1.1, 1.2]

# Korean digit word map (Arabic → Korean)
DIGIT_MAP = {
    "0": "공", "1": "일", "2": "이", "3": "삼",
    "4": "사", "5": "오", "6": "육", "7": "칠",
    "8": "팔", "9": "구",
}


# ─── text utilities ──────────────────────────────────────────────────────────

def digits_to_korean(text: str) -> str:
    """Convert Arabic digit runs to Korean digit words (preserves non-digits)."""
    result = []
    for ch in text:
        result.append(DIGIT_MAP.get(ch, ch))
    return "".join(result)


def group_korean_digits(words: list[str]) -> list[list[str]]:
    """
    Randomly partition a list of digit-words into prosodic groups.
    Returns a list of groups. E.g. ['공','일','공','일','이','삼','사','오','육','칠','팔']
    → [['공','일','공'], ['일','이','삼','사'], ['오','육','칠','팔']]
    """
    n = len(words)
    if n <= 3:
        return [words]

    # Random group sizes: pick 2 or 3 split points
    num_groups = random.choice([2, 3])
    split_positions = sorted(random.sample(range(1, n), min(num_groups - 1, n - 1)))
    groups = []
    prev = 0
    for sp in split_positions:
        groups.append(words[prev:sp])
        prev = sp
    groups.append(words[prev:])
    return [g for g in groups if g]  # drop empty


def make_ssml(text_korean: str, rate: str, template: str) -> str:
    """
    Build SSML string with prosody rate and one of four break-pattern templates.

    template: 'grouped' | 'nobreak' | 'irregular' | 'shortpause'
    """
    words = text_korean.split()

    if template == "nobreak":
        # Fastest case: single run, no word boundaries
        inner = "".join(words)
        return f'<speak><prosody rate="{rate}">{inner}</prosody></speak>'

    if template == "grouped":
        groups = group_korean_digits(words)
        pause_ms = random.randint(50, 200)
        parts = [" ".join(g) for g in groups]
        # random pause between groups
        break_tag = f'<break time="{pause_ms}ms"/>'
        inner = break_tag.join(parts)
        return f'<speak><prosody rate="{rate}">{inner}</prosody></speak>'

    if template == "irregular":
        # Random micro-pauses between individual words
        parts = []
        for i, w in enumerate(words):
            parts.append(w)
            if i < len(words) - 1 and random.random() < 0.4:
                p = random.choice([30, 50, 80, 120])
                parts.append(f'<break time="{p}ms"/>')
        inner = "".join(parts)
        return f'<speak><prosody rate="{rate}">{inner}</prosody></speak>'

    if template == "shortpause":
        # Comma-separated groups → short pauses (natural reading)
        groups = group_korean_digits(words)
        parts = [" ".join(g) for g in groups]
        inner = ", ".join(parts)  # commas produce short TTS pauses
        return f'<speak><prosody rate="{rate}">{inner}</prosody></speak>'

    raise ValueError(f"Unknown template: {template}")


TEMPLATES = ["grouped", "nobreak", "irregular", "shortpause"]
# Distribution: grouped 4 : nobreak 3 : irregular 2 : shortpause 1
TEMPLATE_WEIGHTS = [4, 3, 2, 1]


# ─── audio utilities ─────────────────────────────────────────────────────────

def mp3_to_wav16k(mp3_path: str, wav_path: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp3_path, "-ar", "16000", "-ac", "1",
         "-sample_fmt", "s16", wav_path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def sox_time_stretch(in_wav: str, out_wav: str, factor: float) -> None:
    """
    Time-stretch without pitch change using sox 'tempo -s'.
    factor > 1 → faster; factor < 1 → slower.
    """
    subprocess.run(
        ["sox", in_wav, out_wav, "tempo", "-s", str(factor)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


# ─── TTS generation ──────────────────────────────────────────────────────────

async def synthesize_ssml(ssml: str, voice: str, out_mp3: str, max_retries: int = 5) -> None:
    for attempt in range(max_retries):
        try:
            communicate = edge_tts.Communicate(ssml, voice)
            await communicate.save(out_mp3)
            return
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt + random.uniform(0, 1)
            await asyncio.sleep(wait)


async def generate_variant(
    utt_id: str,
    text_kr: str,
    voice: str,
    rate: str,
    template: str,
    wav_path: str,
) -> None:
    if os.path.exists(wav_path):  # resume: skip already done
        return
    ssml = make_ssml(text_kr, rate, template)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
        mp3_path = tf.name
    try:
        await synthesize_ssml(ssml, voice, mp3_path)
        mp3_to_wav16k(mp3_path, wav_path)
    finally:
        if os.path.exists(mp3_path):
            os.unlink(mp3_path)


# ─── main ────────────────────────────────────────────────────────────────────

def build_variants(utt_id: str, n_variants: int, use_stretch: bool) -> list[dict]:
    """
    Return a list of variant configs.
    Each config: {utt_id, voice, rate, template, stretch (1.0 or factor), suffix}
    """
    variants = []

    # Primary variants: voice × rate × template combinations
    combos = []
    for v in VOICES:
        for r in RATES:
            for t in TEMPLATES:
                combos.append((v, r, t))
    random.shuffle(combos)

    for i, (voice, rate, template) in enumerate(combos[:n_variants]):
        variants.append({
            "utt_id": f"{utt_id}_v{i:03d}",
            "voice": voice,
            "rate": rate,
            "template": template,
            "stretch": 1.0,
        })

    # Stretch variants: re-use first TTS output, apply sox tempo
    if use_stretch and variants:
        base = variants[0]
        for factor in STRETCH_FACTORS:
            tag = str(factor).replace(".", "")
            variants.append({
                "utt_id": f"{utt_id}_s{tag}",
                "voice": base["voice"],
                "rate": base["rate"],
                "template": base["template"],
                "stretch": factor,
            })

    return variants


async def process_utterance(
    utt_id: str,
    text: str,
    wav_dir: Path,
    manifest_lines: list,
    n_variants: int,
    use_stretch: bool,
    semaphore: asyncio.Semaphore,
) -> None:
    text_kr = digits_to_korean(text)

    variants = build_variants(utt_id, n_variants, use_stretch)

    # Generate base wavs (may reuse for stretch)
    base_wav = None
    tasks = []

    async def gen(var):
        nonlocal base_wav
        wav_path = wav_dir / f"{var['utt_id']}.wav"
        if var["stretch"] != 1.0:
            return  # handled after base wavs are done
        async with semaphore:
            await generate_variant(
                var["utt_id"], text_kr,
                var["voice"], var["rate"], var["template"],
                str(wav_path),
            )
        if base_wav is None:
            base_wav = str(wav_path)

    await asyncio.gather(*[gen(v) for v in variants if v["stretch"] == 1.0])

    # Apply sox stretch to base wav
    for var in variants:
        if var["stretch"] == 1.0:
            continue
        if base_wav is None:
            continue
        wav_path = wav_dir / f"{var['utt_id']}.wav"
        sox_time_stretch(base_wav, str(wav_path), var["stretch"])

    # Record manifest
    for var in variants:
        wav_path = wav_dir / f"{var['utt_id']}.wav"
        if wav_path.exists():
            manifest_lines.append(f"{var['utt_id']}\t{wav_path}\t{text_kr}")


async def run(args):
    out_dir = Path(args.out_dir)
    wav_dir = out_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    lines = Path(args.input).read_text(encoding="utf-8").splitlines()
    utterances = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            utterances.append((parts[0], parts[1]))
        else:
            # No utt_id: use line index
            utterances.append((f"utt{len(utterances):06d}", parts[0]))

    random.seed(args.seed)
    semaphore = asyncio.Semaphore(args.jobs)
    manifest_lines = []

    tasks = [
        process_utterance(uid, text, wav_dir, manifest_lines,
                          args.variants, args.stretch, semaphore)
        for uid, text in utterances
    ]

    total = len(tasks)
    for i, coro in enumerate(asyncio.as_completed(tasks), 1):
        await coro
        if i % 50 == 0 or i == total:
            print(f"  [{i}/{total}] done", flush=True)

    manifest_path = out_dir / "manifest.txt"
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(f"\nDone. {len(manifest_lines)} wavs → {out_dir}")
    print(f"Manifest: {manifest_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="Input text list (utt_id text, one per line)")
    ap.add_argument("out_dir", help="Output directory")
    ap.add_argument("--variants", type=int, default=5,
                    help="TTS variants per utterance before stretch (default: 5)")
    ap.add_argument("--stretch", action="store_true",
                    help="Also generate sox time-stretch variants (0.9/1.1/1.2x)")
    ap.add_argument("--jobs", type=int, default=4,
                    help="Concurrent TTS requests (default: 4)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
