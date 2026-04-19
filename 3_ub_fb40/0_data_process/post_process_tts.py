#!/usr/bin/env python3
"""
post_process_tts.py — Wait for TTS generation, build HTK MLF, compress output.

Steps:
  1. Poll until gen_tts process exits and manifest.txt is complete
  2. Generate HTK MLF (one Korean syllable per line, sp between words)
  3. tar.gz wav/ + mlf into a single archive

Usage:
  python post_process_tts.py <tts_out_dir> [--sp]

  <tts_out_dir>  directory containing manifest.txt and wav/
  --sp           insert <sp> label between words (default: no)

Output:
  <tts_out_dir>/tts_kr_num.mlf
  <tts_out_dir>/tts_kr_num.tar.gz
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


# ─── MLF generation ──────────────────────────────────────────────────────────

def text_to_labels(text: str, insert_sp: bool) -> list[str]:
    """
    Convert Korean digit text to label sequence.
    "공일공 일이삼사 오육칠팔" → ['공','일','공','sp','일','이','삼','사','sp','오','육','칠','팔']
    Spaces between groups → optional <sp> token.
    """
    labels = []
    words = text.strip().split()
    for i, word in enumerate(words):
        if i > 0 and insert_sp:
            labels.append("sp")
        for ch in word:
            if ch.strip():
                labels.append(ch)
    return labels


def generate_mlf(manifest_path: Path, mlf_path: Path, insert_sp: bool) -> int:
    lines_written = 0
    with open(manifest_path, encoding="utf-8") as fin, \
         open(mlf_path, "w", encoding="utf-8") as fout:
        fout.write("#!MLF!#\n")
        for line in fin:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            utt_id, _wav_path, text = parts[0], parts[1], parts[2]
            labels = text_to_labels(text, insert_sp)
            if not labels:
                continue
            fout.write(f'"*/{utt_id}.lab"\n')
            for lbl in labels:
                fout.write(lbl + "\n")
            fout.write(".\n")
            lines_written += 1
    return lines_written


# ─── compression ─────────────────────────────────────────────────────────────

def compress(out_dir: Path, archive_path: Path) -> None:
    wav_dir = out_dir / "wav"
    mlf_path = out_dir / "tts_kr_num.mlf"
    manifest = out_dir / "manifest.txt"
    print(f"Compressing → {archive_path} ...")
    subprocess.run(
        ["tar", "-czf", str(archive_path),
         "-C", str(out_dir),
         "wav", "tts_kr_num.mlf", "manifest.txt"],
        check=True,
    )
    size_gb = archive_path.stat().st_size / 1024**3
    print(f"Archive size: {size_gb:.2f} GB")


# ─── wait logic ──────────────────────────────────────────────────────────────

def wait_for_completion(out_dir: Path, poll_interval: int = 30) -> Path:
    """
    Poll until:
      - manifest.txt exists
      - gen_tts process is no longer running
    Returns manifest path.
    """
    manifest = out_dir / "manifest.txt"
    print("Waiting for TTS generation to complete ...")
    while True:
        # check process
        result = subprocess.run(
            ["pgrep", "-f", "gen_tts_kr_number.py"],
            capture_output=True,
        )
        proc_alive = result.returncode == 0

        wav_count = len(list((out_dir / "wav").glob("*.wav"))) if (out_dir / "wav").exists() else 0

        if proc_alive:
            print(f"  still running — {wav_count} wavs so far", flush=True)
            time.sleep(poll_interval)
            continue

        # process dead
        if manifest.exists():
            n = sum(1 for l in manifest.read_text().splitlines() if l.strip())
            print(f"Generation done. manifest has {n} entries, {wav_count} wavs.")
            return manifest
        else:
            # process exited but no manifest → likely crashed
            print(f"WARNING: process exited but manifest.txt not found. "
                  f"{wav_count} wavs exist. Generating MLF from wavs instead.")
            return _build_manifest_from_wavs(out_dir)


def _build_manifest_from_wavs(out_dir: Path) -> Path:
    """Fallback: reconstruct manifest from existing wav filenames + phone_numbers_10k.txt."""
    # Load phone number map from sibling input file
    input_file = out_dir.parent / "phone_numbers_10k.txt"
    num_map = {}
    if input_file.exists():
        for line in input_file.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                num_map[parts[0]] = parts[1]

    digit_map = {
        "0": "공", "1": "일", "2": "이", "3": "삼",
        "4": "사", "5": "오", "6": "육", "7": "칠",
        "8": "팔", "9": "구",
    }

    def to_korean(text):
        return "".join(digit_map.get(c, c) for c in text)

    manifest = out_dir / "manifest.txt"
    wav_dir = out_dir / "wav"
    lines = []
    for wav in sorted(wav_dir.glob("*.wav")):
        utt_id = wav.stem
        # utt_id format: num000000_v001 or num000000_s12
        base = utt_id.rsplit("_", 1)[0]
        raw_text = num_map.get(base, "")
        text_kr = to_korean(raw_text) if raw_text else ""
        lines.append(f"{utt_id}\t{wav}\t{text_kr}")

    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Reconstructed manifest with {len(lines)} entries.")
    return manifest


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tts_out_dir", help="Directory with manifest.txt and wav/")
    ap.add_argument("--sp", action="store_true",
                    help="Insert <sp> label between word groups")
    ap.add_argument("--poll", type=int, default=30,
                    help="Poll interval in seconds (default: 30)")
    ap.add_argument("--no-wait", action="store_true",
                    help="Skip waiting; process manifest.txt immediately")
    args = ap.parse_args()

    out_dir = Path(args.tts_out_dir).resolve()
    if not out_dir.exists():
        sys.exit(f"Directory not found: {out_dir}")

    # Step 1: wait
    manifest = out_dir / "manifest.txt"
    if args.no_wait and manifest.exists():
        print(f"--no-wait: using existing manifest ({sum(1 for l in manifest.read_text().splitlines() if l.strip())} entries)")
    else:
        manifest = wait_for_completion(out_dir, args.poll)

    # Step 2: MLF
    mlf_path = out_dir / "tts_kr_num.mlf"
    print(f"Generating MLF → {mlf_path}")
    n = generate_mlf(manifest, mlf_path, args.sp)
    print(f"  {n} utterances written to MLF")

    # Step 3: compress
    archive = out_dir / "tts_kr_num.tar.gz"
    compress(out_dir, archive)

    print(f"\nDone. Archive ready: {archive}")
    print("Next step: upload to Google Drive (run upload step separately).")


if __name__ == "__main__":
    main()
