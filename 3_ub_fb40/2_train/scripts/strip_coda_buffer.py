"""
Strip 'loss.boost_vec' (and any other loss.* buffers) from existing
PyTorch checkpoints so that pytorch2mat conversion doesn't choke on
unknown keys.

Why needed:
    Earlier version of CodaWeightedCTCLoss registered boost_vec as a
    persistent buffer, so it got saved into checkpoints. Newer code uses
    persistent=False (won't save), but old checkpoints already have the key.

Usage:
    # Single file
    python scripts/strip_coda_buffer.py /path/to/model.iter12.part0

    # Glob pattern (in-place edit, makes .bak)
    python scripts/strip_coda_buffer.py /path/to/model.iter*.part*

Side effects:
    - Backs up original to <file>.bak
    - Saves stripped version to original path
"""

import sys
import os
import glob
import shutil
import torch


# Keys to strip from state_dict
KEYS_TO_STRIP = {
    'loss.boost_vec',
    # Add any other future loss-related buffers here
}


def strip_one(path):
    print(f"Processing: {path}")
    if not os.path.exists(path):
        print(f"  SKIP: not found")
        return False

    ckpt = torch.load(path, map_location='cpu')

    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        sd = ckpt['state_dict']
        wrapper = True
    elif isinstance(ckpt, dict):
        sd = ckpt
        wrapper = False
    else:
        print(f"  SKIP: unexpected checkpoint format ({type(ckpt)})")
        return False

    stripped_keys = []
    for k in list(sd.keys()):
        if k in KEYS_TO_STRIP:
            del sd[k]
            stripped_keys.append(k)

    if not stripped_keys:
        print(f"  No keys to strip; checkpoint already clean.")
        return False

    # Backup
    bak_path = path + '.bak'
    if not os.path.exists(bak_path):
        shutil.copy2(path, bak_path)
        print(f"  Backup saved: {bak_path}")
    else:
        print(f"  Backup already exists: {bak_path}")

    # Save stripped checkpoint
    if wrapper:
        ckpt['state_dict'] = sd
        torch.save(ckpt, path)
    else:
        torch.save(sd, path)

    print(f"  Stripped keys: {stripped_keys}")
    print(f"  Saved cleaned checkpoint -> {path}")
    return True


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <checkpoint_path_or_glob> [more_paths...]")
        sys.exit(1)

    targets = []
    for arg in sys.argv[1:]:
        # Expand glob
        matches = glob.glob(arg)
        if matches:
            targets.extend(matches)
        else:
            targets.append(arg)

    print(f"Total targets: {len(targets)}")
    print("=" * 60)

    n_changed = 0
    for t in targets:
        if strip_one(t):
            n_changed += 1
        print()

    print("=" * 60)
    print(f"Done. Modified {n_changed}/{len(targets)} checkpoints.")
    print("Originals backed up as <file>.bak")


if __name__ == '__main__':
    main()
