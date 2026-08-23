"""Cross-world decoding error distribution plots.

Plot 1 (error_dist_singleworld.svg):
  A) Same world — train 45 → test 45
  B) Different world — train 50 → test 45
  C) Untrained control → test 45

Plot 2 (error_dist_multiworld.svg):
  A) Same world — train 45 → test 45
  B) Multi-world — train 104 (fixed_seed=False) → test 45

Usage:
  # original worlds_test_* runs (default):
  MPLBACKEND=Agg python dreamerv3/plot_error_dist.py
  # masked_reg_worlds_test_* runs:
  MPLBACKEND=Agg python dreamerv3/plot_error_dist.py --run_prefix masked_reg_worlds_test
"""

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

LAYER = 'dyn/deter'


def load(p, layer=LAYER):
    with open(p, 'rb') as f:
        d = pickle.load(f)
    return np.asarray(d['layer_fold_values'][layer])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run_prefix', default='worlds_test',
                    help="run-dir prefix under logdir/ (e.g. 'worlds_test' or 'masked_reg_worlds_test'); "
                         "runs are '<prefix>_{45,50,104}'")
    ap.add_argument('--decode_subdir', default='layer_decoder_results_seed45',
                    help='trained decode subdir name inside each run')
    ap.add_argument('--a_from', default='',
                    help='optional full path to the A-condition layer_decode_results.pkl '
                         '(overrides <base45>/<decode_subdir>; e.g. point A at masked_k5_invert_reg_wrld45)')
    ap.add_argument('--b_from', default='',
                    help='optional full path to the B (diff-world) layer_decode_results.pkl '
                         '(overrides base50; e.g. A-trajectories replayed through the train-50 net)')
    ap.add_argument('--multi_from', default='',
                    help='optional full path to the multi-world (C-inf) layer_decode_results.pkl '
                         '(overrides base104; e.g. A-trajectories replayed through the train-104 net)')
    ap.add_argument('--untrained_subdir', default='layer_decoder_results_untrained',
                    help='untrained-control decode subdir inside the test-45 run')
    ap.add_argument('--untrained_from', default='',
                    help='optional full path to an untrained layer_decode_results.pkl to use as C '
                         '(overrides untrained_subdir; useful to borrow another run\'s untrained control)')
    ap.add_argument('--save', default='', help='output dir (default: <base45>/plots)')
    ap.add_argument('--layer', default=LAYER)
    args = ap.parse_args()

    layer = args.layer
    base45 = Path('logdir') / f'{args.run_prefix}_45'
    base50 = Path('logdir') / f'{args.run_prefix}_50'
    base104 = Path('logdir') / f'{args.run_prefix}_104'

    a_path = Path(args.a_from) if args.a_from else base45 / args.decode_subdir / 'layer_decode_results.pkl'
    b_path = Path(args.b_from) if args.b_from else base50 / args.decode_subdir / 'layer_decode_results.pkl'
    multi_path = Path(args.multi_from) if args.multi_from else base104 / args.decode_subdir / 'layer_decode_results.pkl'
    same = load(a_path, layer)
    diff = load(b_path, layer)
    multi = load(multi_path, layer)

    untrained_path = Path(args.untrained_from) if args.untrained_from \
        else base45 / args.untrained_subdir / 'layer_decode_results.pkl'
    untrained = load(untrained_path, layer) if untrained_path.exists() else None

    save_dir = Path(args.save) if args.save else base45 / 'plots'
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- Plot 1: Same world vs Different world vs Untrained ----
    if untrained is None:
        print(f'[skip] singleworld plot: untrained control not found at {untrained_path}\n'
              f'       (pass --untrained_from <pkl> to borrow one, or generate the untrained control)')
    else:
        all_v = np.concatenate([same, diff, untrained])
        max_err = int(np.percentile(all_v, 95)) + 1
        bins = np.arange(0, max_err + 1) - 0.5

        fig, ax = plt.subplots(figsize=(9, 5))
        for v, color, label in [
            (same, '#2196F3', f'A: Same world, train 45 (mean={same.mean():.1f}, med={np.median(same):.0f})'),
            (diff, '#FF9800', f'B: Diff world, train 50 (mean={diff.mean():.1f}, med={np.median(diff):.0f})'),
            (untrained, '#9E9E9E', f'C: Untrained (mean={untrained.mean():.1f}, med={np.median(untrained):.0f})'),
        ]:
            ax.hist(v, bins=bins, color=color, alpha=0.45, edgecolor=color, linewidth=0.8, label=label)
            ax.axvline(np.median(v), color=color, ls=':', lw=1.5, alpha=0.9)
        ax.set_xlabel('Manhattan error (tiles)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'Decoding Error — {LAYER} (all tested on World 45)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        fig.savefig(str(save_dir / 'error_dist_singleworld.svg'), bbox_inches='tight')
        print(f'Saved {save_dir / "error_dist_singleworld.svg"}')

    # ---- Plot 2: Same vs Diff vs Multi-world (+ untrained chance floor) ----
    series2 = [
        (same, '#2196F3', f'A: Same world (train 45) (mean={same.mean():.1f}, med={np.median(same):.0f})'),
        (diff, '#FF9800', f'B: Diff world (train 50) (mean={diff.mean():.1f}, med={np.median(diff):.0f})'),
        (multi, '#4CAF50', f'C∞: Multi-world (train 104) (mean={multi.mean():.1f}, med={np.median(multi):.0f})'),
    ]
    if untrained is not None:
        series2.append(
            (untrained, '#9E9E9E', f'Untrained (mean={untrained.mean():.1f}, med={np.median(untrained):.0f})'))
    all_v2 = np.concatenate([v for v, _, _ in series2])
    max_err2 = int(np.percentile(all_v2, 95)) + 1
    bins2 = np.arange(0, max_err2 + 1) - 0.5

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    for v, color, label in series2:
        ax2.hist(v, bins=bins2, color=color, alpha=0.45, edgecolor=color, linewidth=0.8, label=label)
        ax2.axvline(np.median(v), color=color, ls=':', lw=1.5, alpha=0.9)
    ax2.set_xlabel('Manhattan error (tiles)', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title(f'Decoding Error — {LAYER} (all tested on World 45)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    fig2.tight_layout()
    fig2.savefig(str(save_dir / 'error_dist_multiworld.svg'), bbox_inches='tight')
    print(f'Saved {save_dir / "error_dist_multiworld.svg"}')


if __name__ == '__main__':
    main()
