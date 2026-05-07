"""
Analyze HResults-style stat file to break down errors by type, position
and confusion pair. Supports A/B comparison between two stat files.

Stat file format expected:
    GB310_F001_normal_53.wav:  100.00( 92.86) [H= 14, D= 0, S= 0, I= 1, N= 14]
    Aligned transcription: /path/to/.lab vs /path/to/.wav
    LAB: 일 육 육 일 이     오 육 구 로 전 화 걸 어 줘
    REC: 일 육 육 일 이 오 오 육 구 로 전 화 걸 어 줘
    GB310_F001_normal_65.wav:  100.00(100.00) [H= 14, D= 0, S= 0, I= 0, N= 14]
    ...

Usage:
    # Single file analysis
    python scripts/analyze_stat.py path/to/result.stat

    # Save report to file
    python scripts/analyze_stat.py path/to/result.stat --output report.txt

    # A/B compare baseline vs codaR1
    python scripts/analyze_stat.py codaR1.stat --compare-with baseline.stat \
        --output compare_report.txt
"""

import sys
import re
import os
import argparse
from collections import Counter, defaultdict


# 韩语数字字符
DIGITS = set('영공일이삼사오육칠팔구')

# 易混对（声学相似），按重要性排序
CONFUSION_PAIRS = [
    ('일', '이', '1↔2 (coda ll missing)'),
    ('이', '일', '2↔1 (coda ll spurious)'),
    ('삼', '사', '3↔4 (coda mm missing)'),
    ('사', '삼', '4↔3 (coda mm spurious)'),
    ('칠', '팔', '7↔8 (vowel)'),
    ('팔', '칠', '8↔7 (vowel)'),
    ('일', '칠', '1↔7'),
    ('칠', '일', '7↔1'),
    ('공', '구', '0↔9'),
    ('구', '공', '9↔0'),
    ('영', '오', '0↔5'),
    ('오', '영', '5↔0'),
    ('오', '구', '5↔9'),
    ('구', '오', '9↔5'),
    ('육', '구', '6↔9'),
    ('구', '육', '9↔6'),
]


def needleman_wunsch(ref_tokens, hyp_tokens):
    """Align ref vs hyp. Returns list of (ref, hyp, op) tuples.
    op in {'M', 'S', 'D', 'I'}: match, substitution, deletion, insertion.
    Tie-breaking: prefer S > D > I (consistent with HTK/HResults convention).
    """
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    bt = [[None] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
        if i > 0:
            bt[i][0] = 'D'
    for j in range(n + 1):
        dp[0][j] = j
        if j > 0:
            bt[0][j] = 'I'

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == hyp_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1]
                bt[i][j] = 'M'
            else:
                sub = dp[i-1][j-1] + 1
                dele = dp[i-1][j] + 1
                ins = dp[i][j-1] + 1
                if sub <= dele and sub <= ins:
                    dp[i][j] = sub
                    bt[i][j] = 'S'
                elif dele <= ins:
                    dp[i][j] = dele
                    bt[i][j] = 'D'
                else:
                    dp[i][j] = ins
                    bt[i][j] = 'I'

    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        op = bt[i][j]
        if op == 'M':
            ops.append((ref_tokens[i-1], hyp_tokens[j-1], 'M'))
            i -= 1; j -= 1
        elif op == 'S':
            ops.append((ref_tokens[i-1], hyp_tokens[j-1], 'S'))
            i -= 1; j -= 1
        elif op == 'D':
            ops.append((ref_tokens[i-1], None, 'D'))
            i -= 1
        elif op == 'I':
            ops.append((None, hyp_tokens[j-1], 'I'))
            j -= 1
        else:
            break
    ops.reverse()
    return ops


def _is_ref_line(line):
    """Match LAB:/REF:/lab:/ref: prefix with optional leading whitespace."""
    s = line.lstrip()
    for prefix in ('LAB:', 'REF:', 'Lab:', 'Ref:', 'lab:', 'ref:'):
        if s.startswith(prefix):
            return True
    return False


def _is_hyp_line(line):
    """Match REC:/HYP:/rec:/hyp: prefix with optional leading whitespace."""
    s = line.lstrip()
    for prefix in ('REC:', 'HYP:', 'Rec:', 'Hyp:', 'rec:', 'hyp:'):
        if s.startswith(prefix):
            return True
    return False


def _extract_after_colon(line):
    """Strip leading whitespace, return content after first ':'."""
    s = line.strip()
    idx = s.find(':')
    if idx < 0:
        return ''
    return s[idx+1:].strip()


def parse_stat_file(path, debug=False):
    """Yield record dicts (utt, sent_acc, word_acc, H/D/S/I/N, ref, hyp).

    Handles HResults-style format with summary line followed by optional
    'Aligned transcription:' / LAB: / REC: block. Tolerates different
    prefix casing and leading whitespace.
    """
    summary_re = re.compile(
        r'^\s*(\S+\.wav):\s+([\d.]+)\(\s*([\d.]+)\)\s+'
        r'\[H=\s*(\d+),\s*D=\s*(\d+),\s*S=\s*(\d+),\s*I=\s*(\d+),\s*N=\s*(\d+)\]'
    )

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    n_summary = 0
    n_with_errors = 0
    n_alignment_found = 0
    debug_misses = []

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        m = summary_re.match(line)
        if not m:
            i += 1
            continue

        n_summary += 1
        rec = {
            'utt': m.group(1),
            'sent_acc': float(m.group(2)),
            'word_acc': float(m.group(3)),
            'H': int(m.group(4)),
            'D': int(m.group(5)),
            'S': int(m.group(6)),
            'I': int(m.group(7)),
            'N': int(m.group(8)),
            'ref': None,
            'hyp': None,
        }

        if (rec['D'] + rec['S'] + rec['I']) > 0:
            n_with_errors += 1
            # Look ahead up to 15 lines for LAB/REC block
            j = i + 1
            lab_idx = None
            rec_idx = None
            while j < len(lines) and j - i <= 15:
                nl = lines[j]
                if _is_ref_line(nl):
                    lab_idx = j
                if _is_hyp_line(nl):
                    rec_idx = j
                if lab_idx is not None and rec_idx is not None:
                    break
                # Stop early if we hit next summary
                if summary_re.match(nl.rstrip()):
                    break
                j += 1

            if lab_idx is not None:
                rec['ref'] = _extract_after_colon(lines[lab_idx]).split()
            if rec_idx is not None:
                rec['hyp'] = _extract_after_colon(lines[rec_idx]).split()

            if lab_idx is not None and rec_idx is not None:
                n_alignment_found += 1
                i = max(lab_idx, rec_idx)
            elif debug and len(debug_misses) < 3:
                # Capture what came after the summary for debug
                ctx = []
                for k in range(i, min(i + 6, len(lines))):
                    ctx.append(f"  L{k}: {repr(lines[k].rstrip())}")
                debug_misses.append((rec['utt'], '\n'.join(ctx)))

        yield rec
        i += 1

    if debug:
        print(f"\n[DEBUG] Parser stats:", file=sys.stderr)
        print(f"  Summary lines parsed:    {n_summary}", file=sys.stderr)
        print(f"  Utts with errors:        {n_with_errors}", file=sys.stderr)
        print(f"  Alignment blocks found:  {n_alignment_found}", file=sys.stderr)
        if n_alignment_found < n_with_errors:
            print(f"  !! MISSING {n_with_errors - n_alignment_found} alignment blocks !!", file=sys.stderr)
            for utt, ctx in debug_misses:
                print(f"\n  Sample miss '{utt}':", file=sys.stderr)
                print(ctx, file=sys.stderr)


def classify_digit_position(ref, ref_idx):
    """Given ref tokens and current ref_idx, return position label
    relative to digit sequence: 'start' / 'middle' / 'end' / 'command'.
    """
    if ref_idx >= len(ref):
        return 'unknown'
    tok = ref[ref_idx]
    if tok not in DIGITS:
        return 'command'

    # Find positions of all digits in ref
    digit_positions = [i for i, t in enumerate(ref) if t in DIGITS]
    if not digit_positions:
        return 'unknown'

    if ref_idx == digit_positions[0]:
        return 'start'
    if ref_idx == digit_positions[-1]:
        return 'end'
    return 'middle'


def is_consecutive_same_digit(ref, ref_idx):
    """Check if ref[ref_idx] is part of consecutive same-digit run (叠字)."""
    if ref_idx >= len(ref):
        return False
    tok = ref[ref_idx]
    if tok not in DIGITS:
        return False
    prev_same = ref_idx > 0 and ref[ref_idx-1] == tok
    next_same = ref_idx + 1 < len(ref) and ref[ref_idx+1] == tok
    return prev_same or next_same


def analyze_records(records):
    """Compute aggregate stats over all records."""
    stats = {
        'total_utts': 0,
        'perfect_utts': 0,
        'total_H': 0, 'total_D': 0, 'total_S': 0, 'total_I': 0, 'total_N': 0,
        # Substitutions
        'sub_pairs': Counter(),         # (ref, hyp) -> count
        'sub_digit_pairs': Counter(),   # only digit↔digit
        # Deletions
        'del_tokens': Counter(),
        'del_position': Counter(),      # start/middle/end/command
        'del_consecutive': Counter(),   # 叠字漏字
        # Insertions
        'ins_tokens': Counter(),
        'ins_position': Counter(),
        # Per-utterance status
        'sent_correct': 0,
    }

    for rec in records:
        stats['total_utts'] += 1
        stats['total_H'] += rec['H']
        stats['total_D'] += rec['D']
        stats['total_S'] += rec['S']
        stats['total_I'] += rec['I']
        stats['total_N'] += rec['N']

        if (rec['D'] + rec['S'] + rec['I']) == 0:
            stats['perfect_utts'] += 1
            stats['sent_correct'] += 1
            continue

        if rec['ref'] is None or rec['hyp'] is None:
            continue

        ref = rec['ref']
        hyp = rec['hyp']
        ops = needleman_wunsch(ref, hyp)

        ref_idx = 0
        hyp_idx = 0
        for r, h, op in ops:
            if op == 'M':
                ref_idx += 1
                hyp_idx += 1
            elif op == 'S':
                stats['sub_pairs'][(r, h)] += 1
                if r in DIGITS and h in DIGITS:
                    stats['sub_digit_pairs'][(r, h)] += 1
                ref_idx += 1
                hyp_idx += 1
            elif op == 'D':
                stats['del_tokens'][r] += 1
                pos = classify_digit_position(ref, ref_idx)
                stats['del_position'][pos] += 1
                if is_consecutive_same_digit(ref, ref_idx):
                    stats['del_consecutive'][r] += 1
                ref_idx += 1
            elif op == 'I':
                stats['ins_tokens'][h] += 1
                # Position relative to current alignment in ref
                pos = classify_digit_position(ref, min(ref_idx, len(ref)-1)) if ref else 'unknown'
                stats['ins_position'][pos] += 1
                hyp_idx += 1

    return stats


def format_report(stats, label='Analysis'):
    out = []
    n_utts = stats['total_utts']
    correct = stats['sent_correct']
    sent_acc = 100 * correct / max(1, n_utts)

    word_total = stats['total_N']
    word_correct = stats['total_H']
    word_acc = 100 * word_correct / max(1, word_total)
    cer = 100 * (stats['total_D'] + stats['total_S'] + stats['total_I']) / max(1, word_total)

    out.append("=" * 72)
    out.append(f"{label}")
    out.append("=" * 72)
    out.append(f"Total utterances:    {n_utts}")
    out.append(f"Sentence accuracy:   {sent_acc:.2f}%   ({correct}/{n_utts})")
    out.append(f"Token accuracy:      {word_acc:.2f}%")
    out.append(f"Token error rate:    {cer:.2f}%   (D={stats['total_D']} S={stats['total_S']} I={stats['total_I']} N={word_total})")
    out.append("")

    # Substitution confusion pairs (digit↔digit only)
    out.append("-" * 72)
    out.append("Digit↔Digit Substitution Pairs (top 25)")
    out.append("-" * 72)
    out.append(f"{'REF':<6}{'HYP':<6}{'Count':<8}{'Type':<30}")
    digit_subs = sorted(stats['sub_digit_pairs'].items(), key=lambda x: -x[1])
    for (r, h), c in digit_subs[:25]:
        label_str = ''
        for d1, d2, lbl in CONFUSION_PAIRS:
            if r == d1 and h == d2:
                label_str = lbl
                break
        out.append(f"{r:<6}{h:<6}{c:<8}{label_str:<30}")
    out.append("")

    # Key confusion pairs summary
    out.append("-" * 72)
    out.append("Key Confusion Pairs (named)")
    out.append("-" * 72)
    for d1, d2, lbl in CONFUSION_PAIRS:
        c = stats['sub_pairs'].get((d1, d2), 0)
        if c > 0:
            out.append(f"  {lbl:<35}: {c}")
    out.append("")

    # Deletions
    out.append("-" * 72)
    out.append("Deletions: Tokens missing in HYP (REF dropped)")
    out.append("-" * 72)
    out.append(f"{'Token':<8}{'Count':<8}{'Note'}")
    for tok, c in stats['del_tokens'].most_common(20):
        note = '(digit)' if tok in DIGITS else ''
        out.append(f"  {tok:<6}{c:<8}{note}")
    out.append("")
    out.append("Deletion position breakdown:")
    total_del = sum(stats['del_position'].values()) or 1
    for pos in ['start', 'middle', 'end', 'command', 'unknown']:
        c = stats['del_position'].get(pos, 0)
        if c > 0:
            pct = 100 * c / total_del
            out.append(f"  {pos:<12}{c:<8}({pct:.1f}%)")
    out.append("")
    cons = sum(stats['del_consecutive'].values())
    out.append(f"Consecutive-same-digit deletions (叠字漏字): {cons}")
    if cons > 0:
        for tok, c in stats['del_consecutive'].most_common(10):
            out.append(f"  {tok:<6}{c}")
    out.append("")

    # Insertions
    out.append("-" * 72)
    out.append("Insertions: Tokens in HYP not in REF (extra)")
    out.append("-" * 72)
    out.append(f"{'Token':<8}{'Count':<8}{'Note'}")
    for tok, c in stats['ins_tokens'].most_common(20):
        note = '(digit)' if tok in DIGITS else ''
        out.append(f"  {tok:<6}{c:<8}{note}")
    out.append("")

    return '\n'.join(out)


def format_diff(base_stats, new_stats, base_label='BASELINE', new_label='NEW'):
    """A/B diff between two stats."""
    out = []
    out.append("=" * 72)
    out.append(f"DIFF: {base_label}  →  {new_label}")
    out.append("=" * 72)

    # Top-level metrics
    def _pct(cnt, total):
        return 100 * cnt / max(1, total)

    base_sent_acc = _pct(base_stats['sent_correct'], base_stats['total_utts'])
    new_sent_acc = _pct(new_stats['sent_correct'], new_stats['total_utts'])
    base_cer = _pct(base_stats['total_D'] + base_stats['total_S'] + base_stats['total_I'], base_stats['total_N'])
    new_cer = _pct(new_stats['total_D'] + new_stats['total_S'] + new_stats['total_I'], new_stats['total_N'])

    out.append(f"\n{'Metric':<25}{base_label:<14}{new_label:<14}{'Δ':<10}")
    out.append("-" * 65)
    out.append(f"{'Sentence accuracy':<25}{base_sent_acc:<14.2f}{new_sent_acc:<14.2f}{new_sent_acc - base_sent_acc:+.2f}")
    out.append(f"{'Token error rate':<25}{base_cer:<14.2f}{new_cer:<14.2f}{new_cer - base_cer:+.2f}")
    out.append(f"{'Total deletions':<25}{base_stats['total_D']:<14}{new_stats['total_D']:<14}{new_stats['total_D'] - base_stats['total_D']:+d}")
    out.append(f"{'Total substitutions':<25}{base_stats['total_S']:<14}{new_stats['total_S']:<14}{new_stats['total_S'] - base_stats['total_S']:+d}")
    out.append(f"{'Total insertions':<25}{base_stats['total_I']:<14}{new_stats['total_I']:<14}{new_stats['total_I'] - base_stats['total_I']:+d}")

    # Confusion pair diffs
    out.append(f"\n{'Confusion Pair':<35}{base_label:<10}{new_label:<10}{'Δ':<8}{'%Δ'}")
    out.append("-" * 75)
    for d1, d2, lbl in CONFUSION_PAIRS:
        b = base_stats['sub_pairs'].get((d1, d2), 0)
        n = new_stats['sub_pairs'].get((d1, d2), 0)
        if b == 0 and n == 0:
            continue
        delta = n - b
        pct = 100 * delta / max(1, b) if b > 0 else float('inf')
        marker = '↓' if delta < 0 else ('↑' if delta > 0 else '=')
        pct_str = f"{pct:+.1f}%" if b > 0 else "(new)"
        out.append(f"  {lbl:<33}{b:<10}{n:<10}{delta:+5d}   {pct_str:<8} {marker}")

    # Deletion position diffs
    out.append(f"\n{'Deletion Position':<25}{base_label:<10}{new_label:<10}{'Δ':<8}")
    out.append("-" * 55)
    for pos in ['start', 'middle', 'end', 'command']:
        b = base_stats['del_position'].get(pos, 0)
        n = new_stats['del_position'].get(pos, 0)
        if b == 0 and n == 0:
            continue
        delta = n - b
        out.append(f"  {pos:<23}{b:<10}{n:<10}{delta:+d}")

    # Consecutive deletions diff
    base_cons = sum(base_stats['del_consecutive'].values())
    new_cons = sum(new_stats['del_consecutive'].values())
    out.append(f"\n  {'consecutive-same-digit':<23}{base_cons:<10}{new_cons:<10}{new_cons - base_cons:+d}")

    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser(description='Analyze HResults stat file')
    ap.add_argument('stat_file', help='Stat file to analyze (the "new" one in compare mode)')
    ap.add_argument('--compare-with', help='Baseline stat file to diff against')
    ap.add_argument('--output', help='Save report to file')
    ap.add_argument('--label-new', default='NEW', help='Label for new stat (e.g., codaR1)')
    ap.add_argument('--label-base', default='BASELINE', help='Label for baseline stat')
    ap.add_argument('--debug', action='store_true', help='Print parser debug info to stderr')
    args = ap.parse_args()

    if not os.path.exists(args.stat_file):
        print(f"Error: {args.stat_file} not found")
        sys.exit(1)

    new_stats = analyze_records(parse_stat_file(args.stat_file, debug=args.debug))
    new_report = format_report(new_stats, label=f'{args.label_new}: {args.stat_file}')

    output_parts = [new_report]

    if args.compare_with:
        if not os.path.exists(args.compare_with):
            print(f"Error: {args.compare_with} not found")
            sys.exit(1)
        base_stats = analyze_records(parse_stat_file(args.compare_with, debug=args.debug))
        base_report = format_report(base_stats, label=f'{args.label_base}: {args.compare_with}')
        diff_report = format_diff(base_stats, new_stats,
                                   base_label=args.label_base,
                                   new_label=args.label_new)
        output_parts = [base_report, '\n', new_report, '\n', diff_report]

    final = '\n'.join(output_parts)
    print(final)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(final)
        print(f"\nReport saved to: {args.output}")


if __name__ == '__main__':
    main()
