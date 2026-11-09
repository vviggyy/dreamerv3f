#!/usr/bin/env python
"""Overlaid metric-distribution comparison across conditions (default: EV reliability).

Reproduces the ad-hoc ``dyn_deter_dist_EVs_AvsC_apples`` figures (which were
originally made in an unsaved interactive session) as a committed, retraceable
script. Loads N ``tuning_results.pkl`` files, pulls a per-neuron metric
(``EVs`` by default) for one layer, and overlays their distributions.

Uses the same ``--cond "LABEL=path"`` convention as
``plot_layer_comparison_multiworld.py`` so conditions + labels are explicit
(no default-network hazard — every path is passed in).

Example (C-anchored apples-to-apples, A vs C on dyn/deter EV):
    MPLBACKEND=Agg python dreamerv3/plot_ev_dist.py \
        --cond "A: Same world (train 45)=logdir/masked_k5_invert_reg_wrld45/apples_to_applesC/tuning_A_train45/tuning_results.pkl" \
        --cond "C∞: Multi-world (train 104)=logdir/masked_k5_invert_reg_wrld45/apples_to_applesC/tuning_Cinf_train104/tuning_results.pkl" \
        --layer dyn/deter --metric EVs \
        --out_name dyn_deter_dist_EVs_AvsC_applesC.svg \
        --save logdir/masked_k5_invert_reg_wrld45/plots_applesC
"""
import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import plot_presentable as pp

# metric_key -> (axis label, sensible x-range or None for auto)
METRIC_INFO = {
    'EVs': ('EV reliability', (0.0, 1.0)),
    'SI': ('Spatial information (bits)', None),
    'HD_info': ('HD mutual info (bits)', None),
    'border_score': ('Border score', None),
    'fieldsize': ('Field size (tiles)', None),
    'morans_i': ("Moran's I", None),
    'gearys_c': ("Geary's C", None),
    'getis_ord_g': ('Getis-Ord G', None),
}

# default color cycle (green highlighted last so A-vs-C reads green like the original)
DEFAULT_COLORS = ['#377eb8', '#2ca02c', '#e41a1c', '#984ea3', '#ff7f00', '#666666']


def _load_metric(pkl_path, layer, metric):
    d = pickle.load(open(pkl_path, 'rb'))
    layers = d['layers'] if 'layers' in d else d
    if layer not in layers:
        raise SystemExit(f'layer {layer!r} not in {pkl_path} (have: {list(layers)})')
    metrics = layers[layer]['metrics']
    if metric not in metrics:
        raise SystemExit(f'metric {metric!r} not in {pkl_path}:{layer} (have: {list(metrics)})')
    vals = np.asarray(metrics[metric], dtype=float)
    return vals[np.isfinite(vals)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cond', action='append', default=[], required=True,
                    help='"LABEL=path/to/tuning_results.pkl" (repeatable, drawn in order)')
    ap.add_argument('--layer', default='dyn/deter', help='layer key (default dyn/deter)')
    ap.add_argument('--metric', default='EVs',
                    help='per-neuron metric key in layer["metrics"] (default EVs)')
    ap.add_argument('--save', required=True, help='output directory')
    ap.add_argument('--out_name', default='',
                    help='output filename (default {layer}_dist_{metric}_compare.svg)')
    ap.add_argument('--bins', type=int, default=40)
    ap.add_argument('--density', action='store_true', default=True,
                    help='normalize histograms to density (default True)')
    ap.add_argument('--counts', dest='density', action='store_false',
                    help='plot raw counts instead of density')
    ap.add_argument('--xrange', default='', help='"lo,hi" x-limits (overrides metric default)')
    ap.add_argument('--colors', default='',
                    help='comma-separated colors overriding the default cycle')
    ap.add_argument('--title', default='', help='plot title')
    pp.add_presentable_arg(ap)
    args = ap.parse_args()
    pp.maybe_apply(args, poster=True, small_figsize=False)  # Tier-2 poster; no-op unless set

    conds = []
    for c in args.cond:
        if '=' not in c:
            raise SystemExit(f'--cond must be "LABEL=path", got: {c}')
        label, path = c.split('=', 1)
        conds.append((label.strip(), path.strip()))

    disp, default_range = METRIC_INFO.get(args.metric, (args.metric, None))
    if args.xrange:
        lo, hi = (float(x) for x in args.xrange.split(','))
        xr = (lo, hi)
    else:
        xr = default_range

    colors = args.colors.split(',') if args.colors else DEFAULT_COLORS

    fig, ax = plt.subplots(figsize=(9.0, 6.0) if pp.is_poster() else (5.0, 4.0))
    bins = np.linspace(*xr, args.bins + 1) if xr else args.bins
    for i, (label, path) in enumerate(conds):
        vals = _load_metric(path, args.layer, args.metric)
        col = colors[i % len(colors)]
        ax.hist(vals, bins=bins, density=args.density, histtype='stepfilled',
                color=col, alpha=0.45, edgecolor=col, linewidth=1.4,
                label=f'{label}  (N={len(vals)}, med={np.median(vals):.3f})')
        ax.axvline(np.median(vals), color=col, linestyle='--', linewidth=1.2, alpha=0.9)

    ax.set_xlabel(disp)
    ax.set_ylabel('Density' if args.density else 'Count')
    if xr:
        ax.set_xlim(*xr)
    ax.set_title(args.title or f'{args.layer} — {disp} distribution')
    ax.legend(fontsize=(11 if pp.is_poster() else 8), frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()

    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.layer.replace('/', '_')
    out = save_dir / (args.out_name or f'{prefix}_dist_{args.metric}_compare.svg')
    fig.savefig(str(out), bbox_inches='tight')
    print(f'Saved {out}')
    for label, path in conds:
        vals = _load_metric(path, args.layer, args.metric)
        print(f'  {label:28s} N={len(vals):5d} median={np.median(vals):.3f} '
              f'mean={np.mean(vals):.3f}')


if __name__ == '__main__':
    main()
