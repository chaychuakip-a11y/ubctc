#!/usr/bin/env python3
"""
gen_phone_text.py — Generate Korean phone-number text input for TTS pipeline.

Output format (one per line):
    <utt_id> <korean_text>
  Korean text is space-separated digit words:
    공일공 일이삼사 오육칠팔

Coverage strategy:
  - Realistic prefix distribution (010 dominant + 011/02/070/1588/etc.)
  - Boosted consecutive-same-digit (叠字) density to teach model blank
    emission between same tokens
  - Mix of common command-tail patterns (전화 걸어줘 etc.)

Usage:
    python gen_phone_text.py output.txt [--n-base 10000] [--n-double 3000] \\
        [--with-tail] [--seed 42]
"""

import argparse
import random
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows console (default cp936/gbk can't print Korean)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# Korean digit map
DIGIT_KR = {
    "0": "공", "1": "일", "2": "이", "3": "삼", "4": "사",
    "5": "오", "6": "육", "7": "칠", "8": "팔", "9": "구",
}

# Korean phone prefix templates with weights (matches realistic distribution)
# Format: (prefix_digits, body_lengths, weight)
# body_lengths: list of (mid_len, tail_len) tuples for variability
PREFIX_PATTERNS = [
    # 010-XXXX-XXXX (modern mobile, most common)
    ("010", [(4, 4)], 65),
    # Older mobile 011/016/017/018/019 — XXX-XXXX
    ("011", [(3, 4)], 4),
    ("016", [(3, 4)], 1),
    ("017", [(3, 4)], 1),
    ("018", [(3, 4)], 1),
    ("019", [(3, 4)], 1),
    # Seoul landline 02-XXX-XXXX or 02-XXXX-XXXX
    ("02", [(3, 4), (4, 4)], 8),
    # Other landline 031-064 — XXX-XXXX
    ("031", [(3, 4)], 2),
    ("032", [(3, 4)], 2),
    ("033", [(3, 4)], 1),
    ("041", [(3, 4)], 1),
    ("042", [(3, 4)], 1),
    ("051", [(3, 4)], 2),
    ("053", [(3, 4)], 1),
    ("062", [(3, 4)], 1),
    # Internet phone
    ("070", [(4, 4)], 4),
    # Free call
    ("080", [(3, 4)], 1),
    # Service numbers
    ("1588", [(4,)], 2),
    ("1599", [(4,)], 1),
    ("1644", [(4,)], 1),
]

# Common Korean trailing command phrases ("call XXX", "dial XXX")
TAIL_PHRASES = [
    "전화 걸어줘",
    "전화 걸어",
    "전화해 줘",
    "전화 좀 해줘",
    "걸어줘",
    "에 전화",
    "전화 부탁해",
]


def digits_to_korean(digits_str: str) -> str:
    """Convert "01012345678" → "공일공일이삼사오육칠팔"."""
    return "".join(DIGIT_KR[d] for d in digits_str)


def format_grouped(digits_str: str, prefix_len: int, body_groups: list[int]) -> str:
    """
    Format digits with space separation:
      "01012345678", prefix_len=3, body_groups=[4,4]
      → "공일공 일이삼사 오육칠팔"
    """
    parts = [digits_str[:prefix_len]]
    pos = prefix_len
    for g in body_groups:
        parts.append(digits_str[pos:pos+g])
        pos += g
    return " ".join(digits_to_korean(p) for p in parts)


def gen_random_number(prefix: str, lengths: list[tuple[int, ...]]) -> str:
    """Generate one random phone number digit string given prefix + body length pattern."""
    body_groups = list(random.choice(lengths))
    body_len = sum(body_groups)
    body = "".join(str(random.randint(0, 9)) for _ in range(body_len))
    full = prefix + body
    return format_grouped(full, len(prefix), body_groups)


def gen_double_dense_number(prefix_pattern: tuple) -> str:
    """
    Generate phone number with FORCED double-digit (叠字) patterns.
    Inserts 2-3 pairs of consecutive same digits at random body positions.
    """
    prefix, lengths, _ = prefix_pattern
    body_groups = list(random.choice(lengths))
    body_len = sum(body_groups)

    # Generate body, then force 2-3 consecutive-same pairs
    body = [str(random.randint(0, 9)) for _ in range(body_len)]
    n_pairs = random.choice([2, 2, 3])  # weighted toward 2 pairs
    available_positions = list(range(body_len - 1))
    random.shuffle(available_positions)
    used = set()
    placed = 0
    for pos in available_positions:
        if placed >= n_pairs:
            break
        if pos in used or pos + 1 in used:
            continue
        d = random.randint(0, 9)
        body[pos] = str(d)
        body[pos + 1] = str(d)
        used.add(pos)
        used.add(pos + 1)
        placed += 1

    full = prefix + "".join(body)
    return format_grouped(full, len(prefix), body_groups)


def gen_triple_dense_number(prefix_pattern: tuple) -> str:
    """
    Even more aggressive: include at least one TRIPLE same digit (e.g., 555).
    """
    prefix, lengths, _ = prefix_pattern
    body_groups = list(random.choice(lengths))
    body_len = sum(body_groups)
    if body_len < 3:
        # fallback to double
        return gen_double_dense_number(prefix_pattern)

    body = [str(random.randint(0, 9)) for _ in range(body_len)]
    # Place a triple
    pos = random.randint(0, body_len - 3)
    d = random.randint(0, 9)
    body[pos] = body[pos + 1] = body[pos + 2] = str(d)
    # Optionally a separate double elsewhere
    if random.random() < 0.5 and body_len >= 5:
        used = set(range(pos, pos + 3))
        candidates = [i for i in range(body_len - 1) if i not in used and i + 1 not in used]
        if candidates:
            p = random.choice(candidates)
            d2 = random.randint(0, 9)
            body[p] = body[p + 1] = str(d2)

    full = prefix + "".join(body)
    return format_grouped(full, len(prefix), body_groups)


def maybe_attach_tail(text_kr: str) -> str:
    """Optionally append a common Korean trailing command phrase."""
    if random.random() < 0.5:
        return text_kr + " " + random.choice(TAIL_PHRASES)
    return text_kr


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output", help="Output text file path")
    ap.add_argument("--n-base", type=int, default=10000,
                    help="Number of random phone numbers (natural distribution)")
    ap.add_argument("--n-double", type=int, default=3000,
                    help="Number of forced-double-digit phone numbers (叠字 emphasis)")
    ap.add_argument("--n-triple", type=int, default=1000,
                    help="Number of forced-triple-digit phone numbers")
    ap.add_argument("--with-tail", action="store_true",
                    help="Randomly append Korean command phrases (전화 걸어줘 etc.)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    # Build weighted prefix sampler
    prefixes_with_weights = [(p, l, w) for p, l, w in PREFIX_PATTERNS]
    weights = [w for _, _, w in prefixes_with_weights]

    out_lines = []
    utt_idx = 0

    # 1. Base random numbers (natural distribution)
    for _ in range(args.n_base):
        pp = random.choices(prefixes_with_weights, weights=weights, k=1)[0]
        text_kr = gen_random_number(pp[0], pp[1])
        if args.with_tail:
            text_kr = maybe_attach_tail(text_kr)
        out_lines.append(f"phone{utt_idx:06d} {text_kr}")
        utt_idx += 1

    # 2. Double-digit-dense numbers (叠字 emphasis)
    for _ in range(args.n_double):
        pp = random.choices(prefixes_with_weights, weights=weights, k=1)[0]
        text_kr = gen_double_dense_number(pp)
        if args.with_tail:
            text_kr = maybe_attach_tail(text_kr)
        out_lines.append(f"double{utt_idx:06d} {text_kr}")
        utt_idx += 1

    # 3. Triple-digit-dense numbers
    for _ in range(args.n_triple):
        pp = random.choices(prefixes_with_weights, weights=weights, k=1)[0]
        text_kr = gen_triple_dense_number(pp)
        if args.with_tail:
            text_kr = maybe_attach_tail(text_kr)
        out_lines.append(f"triple{utt_idx:06d} {text_kr}")
        utt_idx += 1

    random.shuffle(out_lines)

    Path(args.output).write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    # Stats
    total = len(out_lines)
    print(f"=== Phone Text Generation Stats ===")
    print(f"Total utterances: {total}")
    print(f"  Base (random):     {args.n_base}  ({100*args.n_base/total:.1f}%)")
    print(f"  Double-dense:      {args.n_double}  ({100*args.n_double/total:.1f}%)")
    print(f"  Triple-dense:      {args.n_triple}  ({100*args.n_triple/total:.1f}%)")
    print(f"With tail phrases: {args.with_tail}")
    print(f"\nOutput: {args.output}")
    print(f"\nSample output (first 10 lines):")
    for l in out_lines[:10]:
        print(f"  {l}")


if __name__ == "__main__":
    main()
