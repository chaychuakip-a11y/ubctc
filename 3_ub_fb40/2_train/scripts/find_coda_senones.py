"""
Parse the senone mapping file (format: 'triphone <whitespace> senone_id')
and produce a boolean mask marking senones whose CENTER phoneme is a Korean
coda (받침/末尾辅音).

File format expected per line:
    <left>-<center>+<right>    <senone_id>
    e.g.,
        kk-pp+ae     7207
        wi-ll+sil    5362       <- ll is coda, sid=5362
        ch-mm+rr     5683       <- mm is coda
        hh-hh+ng     2521       <- center=hh (not coda), right=ng
        wa-jg+dg     4276       <- multiple triphones can share senone

Multiple triphones may share the same senone_id (decision-tree clustering).
A senone is marked as coda iff its CENTER phoneme is in the coda set.

Usage:
    python find_coda_senones.py <senone_map_file> [output_mask.pt] [--vocab-size 9004]

Output:
    coda_senone_mask.pt -- torch bool tensor of shape [vocab_size]
                          True where senone is a coda-related state.
"""

import sys
import os
import argparse
import torch
from collections import Counter, defaultdict

# 韩语数字场景的 coda phonemes
UNAMBIGUOUS_CODA = {'ll', 'mm', 'ng'}
KG_AMBIGUOUS = 'kg'

# Right contexts that mean "end of word" (so kg in this position is coda)
END_CONTEXTS = {'sil', 'sp', 'sb', 'sf', '#', '<eps>', '_'}


def parse_triphone(triphone):
    if '-' not in triphone or '+' not in triphone:
        return None, None, None
    left, rest = triphone.split('-', 1)
    if '+' not in rest:
        return None, None, None
    center, right = rest.split('+', 1)
    return left.strip(), center.strip(), right.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('senone_map', help='senone mapping file (triphone -> sid)')
    ap.add_argument('output', nargs='?', default='coda_senone_mask.pt')
    ap.add_argument('--vocab-size', type=int, default=9004,
                    help='Total vocab/senone count (model output dim). Default 9004.')
    args = ap.parse_args()

    if not os.path.exists(args.senone_map):
        print(f"Error: {args.senone_map} not found")
        sys.exit(1)

    # senone_id -> set of triphones sharing this senone
    sid_to_triphones = defaultdict(list)
    # senone_id -> dominant center phoneme
    sid_centers = defaultdict(Counter)
    # senone_id -> set of right contexts seen
    sid_right_ctx = defaultdict(set)

    total_lines = 0
    parse_failures = 0
    max_sid = -1

    with open(args.senone_map, 'r', encoding='utf-8', errors='ignore') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            total_lines += 1

            parts = line.split()
            if len(parts) < 2:
                parse_failures += 1
                continue

            triphone = parts[0]
            try:
                sid = int(parts[-1])
            except ValueError:
                parse_failures += 1
                continue

            left, center, right = parse_triphone(triphone)
            if center is None:
                parse_failures += 1
                continue

            max_sid = max(max_sid, sid)
            sid_to_triphones[sid].append(triphone)
            sid_centers[sid][center] += 1
            sid_right_ctx[sid].add(right)

    # ============ Determine coda senones ============
    coda_senones = set()
    kg_coda_senones = set()
    kg_onset_senones = set()

    for sid, centers in sid_centers.items():
        # Sanity: a senone usually has ONE center phoneme (decision tree splits by center)
        dominant_center = centers.most_common(1)[0][0]

        if dominant_center in UNAMBIGUOUS_CODA:
            coda_senones.add(sid)
        elif dominant_center == KG_AMBIGUOUS:
            # Check right contexts for this senone
            rights = sid_right_ctx[sid]
            if any(r in END_CONTEXTS for r in rights):
                coda_senones.add(sid)
                kg_coda_senones.add(sid)
            else:
                kg_onset_senones.add(sid)

    # ============ Report ============
    print("=" * 64)
    print("Senone Mapping Statistics")
    print("=" * 64)
    print(f"Total triphone lines:        {total_lines}")
    print(f"Parse failures:              {parse_failures}")
    print(f"Unique senones:              {len(sid_to_triphones)}")
    print(f"Max senone_id seen:          {max_sid}")
    print(f"Vocab size (mask length):    {args.vocab_size}")
    if max_sid >= args.vocab_size:
        print(f"  !! WARNING: max_sid {max_sid} >= vocab_size {args.vocab_size}")
    print()
    print(f"Coda senones (ll/mm/ng + kg-coda):  {len(coda_senones)}")
    print(f"  - kg in coda position:           {len(kg_coda_senones)}")
    print(f"  - kg in onset position (excl):   {len(kg_onset_senones)}")
    print(f"Coda ratio (vs vocab):              {100 * len(coda_senones) / args.vocab_size:.2f}%")
    print(f"Coda ratio (vs unique senones):     {100 * len(coda_senones) / max(1,len(sid_to_triphones)):.2f}%")

    # Phonemes of interest
    phoneme_senone_count = Counter()
    for sid, centers in sid_centers.items():
        dominant = centers.most_common(1)[0][0]
        phoneme_senone_count[dominant] += 1

    print(f"\nUnique senones per phoneme of interest:")
    for ph in ['ll', 'mm', 'ng', 'kg', 'ii', 'aa', 'ow', 'uw', 'ss', 'tch', 'pp', 'jax', 'ju']:
        cnt = phoneme_senone_count.get(ph, 0)
        marker = "  ← CODA" if ph in UNAMBIGUOUS_CODA else \
                 ("  ← AMBIGUOUS (split by ctx)" if ph == KG_AMBIGUOUS else "")
        print(f"  {ph:<8s}: {cnt:>5d}{marker}")

    print(f"\nTop 20 most common centers (by unique senone count):")
    for ph, cnt in phoneme_senone_count.most_common(20):
        print(f"  {ph:<10s}: {cnt}")

    # Sample
    print(f"\nSample coda senones (sid -> example triphones):")
    sample_count = 0
    for sid in sorted(coda_senones)[:15]:
        triphones = sid_to_triphones[sid]
        sample = triphones[:3] + (['...'] if len(triphones) > 3 else [])
        print(f"  sid={sid:>5d}  ({len(triphones):>3d} triphones)  {sample}")
        sample_count += 1
        if sample_count >= 15:
            break

    # ============ Build & save mask ============
    mask = torch.zeros(args.vocab_size, dtype=torch.bool)
    out_of_range = 0
    for sid in coda_senones:
        if 0 <= sid < args.vocab_size:
            mask[sid] = True
        else:
            out_of_range += 1
    if out_of_range > 0:
        print(f"\n!! WARNING: {out_of_range} coda sids out of [0, {args.vocab_size}) range, dropped")

    torch.save(mask, args.output)
    print(f"\nSaved mask to: {os.path.abspath(args.output)}")
    print(f"  shape={list(mask.shape)}, true_count={int(mask.sum().item())}")

    # ============ Sanity check ============
    if len(coda_senones) == 0:
        print("\n!!! No coda senones found. Check phoneme naming.")
    elif int(mask.sum()) / args.vocab_size > 0.30:
        print(f"\n!!! WARNING: coda ratio {100*int(mask.sum())/args.vocab_size:.1f}% too high. Likely false positives.")
    elif int(mask.sum()) / args.vocab_size < 0.005:
        print(f"\n!!! WARNING: coda ratio {100*int(mask.sum())/args.vocab_size:.1f}% too low. Likely missing matches.")
    else:
        print(f"\n✓ Coda ratio looks reasonable. mask file ready for training.")


if __name__ == '__main__':
    main()
