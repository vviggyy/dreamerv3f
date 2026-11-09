#!/usr/bin/env python
"""Overlaid decode-error distribution across conditions (deter by default).

Overlays the per-sample Manhattan position-decode errors of one layer (default
dyn/deter) from N ``layer_decode_results.pkl`` files as density histograms — the
"deter decoder error" multiworld-comparison view (a distributional companion to
the grouped boxplot in plot_layer_comparison_multiworld / compare_conditions).

Same ``--cond "LABEL=path"`` convention as plot_ev_dist / the layer-comparison
scripts (explicit paths, no default-network hazard).

Example (A vs B-on-A vs untrained, dyn/deter):
    MPLBACKEND=Agg python dreamerv3/plot_decode_error_dist.py \\
        --cond "A=logdir/A/layer_decoder_results/layer_decode_results.pkl" \\
        --cond "B=logdir/B/layer_decode_on_A/layer_decode_results.pkl" \\
        --cond "untrained=logdir/A/layer_decoder_results_untrained/layer_decode_results.pkl" \\
        --layer dyn/deter --save logdir/A/plots_sfn --presentable
"""
import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import plot_presentable as pp

DEFAULT_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#999999']


def _load_err(path, layer):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    lfv = d['layer_fold_values']
    if layer not in lfv:
        raise SystemExit(f'{layer!r} not in {path} (have {sorted(lfv)})')
    return np.asarray(lfv[layer], dtype=float).ravel()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cond', action='append', default=[], required=True,
                    help='"LABEL=path/to/layer_decode_results.pkl" (repeatable, in order)')
    ap.add_argument('--layer', default='dyn/deter', help='layer key (default dyn/deter)')
    ap.add_argument('--save', required=True, help='output directory')
    ap.add_argument('--out_name', default='', help='output filename')
    ap.add_argument('--bins', type=int, default=40)
    ap.add_argument('--density', action='store_true', default=True,
                    help='normalize to density (default True)')
    ap.add_argument('--counts', dest='density', action='store_false')
    ap.add_argument('--xrange', default='', help='"lo,hi" x-limits')
    ap.add_argument('--colors', default='', help='comma-separated color overrides')
    ap.add_argument('--title', default='', help='plot title')
    pp.add_presentable_arg(ap)
    args = ap.parse_args()
    pp.maybe_apply(args, poster=True, small_figsize=False)

    conds = []
    for c in args.cond:
        if '=' not in c:
            raise SystemExit(f'--cond must be "LABEL=path", got: {c}')
        label, path = c.split('=', 1)
        conds.append((label.strip(), path.strip()))

    colors = args.colors.split(',') if args.colors else DEFAULT_COLORS
    allvals = [_load_err(p, args.layer) for _, p in conds]
    if args.xrange:
        lo, hi = (float(x) for x in args.xrange.split(','))
        xr = (lo, hi)
    else:
        xr = (0.0, float(max(np.percentile(v, 99) for v in allvals)))
    bins = np.linspace(xr[0], xr[1], args.bins + 1)

    fig, ax = plt.subplots(figsize=(9.0, 6.0) if pp.is_poster() else (5.0, 4.0))
    for i, ((label, _), vals) in enumerate(zip(conds, allvals)):
        col = colors[i % len(colors)]
        ax.hist(vals, bins=bins, density=args.density, histtype='stepfilled',
                color=col, alpha=0.45, edgecolor=col, linewidth=1.4,
                label=f'{label}  (N={len(vals)}, med={np.median(vals):.1f})')
        ax.axvline(np.median(vals), color=col, linestyle='--', linewidth=1.2, alpha=0.9)

    ax.set_xlabel('Manhattan decode error (tiles)')
    ax.set_ylabel('Density' if args.density else 'Count')
    ax.set_xlim(*xr)
    ax.set_title(args.title or f'{args.layer} — decode error')
    ax.legend(fontsize=(11 if pp.is_poster() else 8), frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()

    Path(args.save).mkdir(parents=True, exist_ok=True)
    out = Path(args.save) / (args.out_name
                             or f'{args.layer.replace("/", "_")}_decode_error_dist.svg')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
