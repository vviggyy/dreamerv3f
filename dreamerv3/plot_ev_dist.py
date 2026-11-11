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


def _dead_mask(layer_dict, tol=0.0):
    """Per-neuron 'dead' mask: a (ReLU) unit that never activates anywhere.

    Literature: dying-ReLU / dead units (Glorot 2011; Lu 2019) and the
    dormant-neuron score at tau=0 (Sokar et al., ICML 2023) both reduce, for a
    non-negative activation, to "max activation over the dataset <= 0". We read
    it off the stored spatial tuning curve (mean activation per visited bin):
    an all-(near-)zero tuning curve => the neuron fires nowhere. Returns None if
    tuning curves are unavailable (older pkls).
    """
    tc = layer_dict.get('tuning_curves')
    if tc is None:
        return None
    tc = np.asarray(tc, dtype=float)                 # (N, X, Y)
    peak = np.nanmax(tc.reshape(tc.shape[0], -1), axis=1)
    peak = np.where(np.isfinite(peak), peak, 0.0)    # all-NaN tuning curve -> 0
    return peak <= tol


def _load_metric(pkl_path, layer, metric, exclude_dead=False, dead_tol=0.0):
    d = pickle.load(open(pkl_path, 'rb'))
    layers = d['layers'] if 'layers' in d else d
    if layer not in layers:
        raise SystemExit(f'layer {layer!r} not in {pkl_path} (have: {list(layers)})')
    ldict = layers[layer]
    metrics = ldict['metrics']
    if metric not in metrics:
        raise SystemExit(f'metric {metric!r} not in {pkl_path}:{layer} (have: {list(metrics)})')
    vals = np.asarray(metrics[metric], dtype=float)
    keep = np.isfinite(vals)
    n_total = int(keep.sum())   # finite units BEFORE dead exclusion (= full pop)
    if exclude_dead:
        dead = _dead_mask(ldict, tol=dead_tol)
        if dead is None:
            print(f'  [warn] no tuning_curves in {pkl_path}:{layer}; cannot '
                  f'exclude dead neurons (keeping all)')
        else:
            n_dead = int((dead & keep).sum())
            print(f'  [exclude_dead] {pkl_path}:{layer} dropped {n_dead} dead '
                  f'(never-activating) neurons of {int(keep.sum())}')
            keep &= ~dead
    return vals[keep], n_total


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
    ap.add_argument('--exclude_dead', action='store_true',
                    help='drop dead (never-activating) neurons: tuning-curve '
                         'peak <= --dead_tol (ReLU dead-unit / tau=0 dormant). '
                         'For EVs this removes the EV=0 spike.')
    ap.add_argument('--dead_tol', type=float, default=0.0,
                    help='max-activation threshold for "dead" (default 0.0)')
    ap.add_argument('--fraction', action='store_true',
                    help='y-axis = fraction of neurons per bin (bars sum to 1 '
                         'per condition); more interpretable than density. '
                         'Overrides --density/--counts.')
    ap.add_argument('--tuned_thresh', type=float, default=None,
                    help='Mark "tuned" cells at metric > THRESH (e.g. 0.4 for EV): '
                         'draws a threshold line and annotates each condition with '
                         'the %% of tuned cells (of the plotted population; the '
                         'of-all-units %% incl. dead is also printed).')
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
    # Draw so the FIRST condition is frontmost (e.g. A over B, A over C):
    # matplotlib's default z-order follows draw order (last = on top), which
    # would bury the first condition; set an explicit decreasing z-order.
    n = len(conds)
    for i, (label, path) in enumerate(conds):
        vals, n_total = _load_metric(path, args.layer, args.metric,
                                     exclude_dead=args.exclude_dead,
                                     dead_tol=args.dead_tol)
        col = colors[i % len(colors)]
        z = (n - i)  # cond 0 (A) highest -> drawn in front
        # --fraction: weight each neuron by 1/N so bar heights are the fraction
        # of neurons in that bin (summing to 1 per condition); mutually
        # exclusive with density normalization.
        weights = (np.full(len(vals), 1.0 / len(vals)) if args.fraction and len(vals)
                   else None)
        lab = f'{label}  (N={len(vals)}, med={np.median(vals):.3f})'
        if args.tuned_thresh is not None and len(vals):
            n_tuned = int((vals > args.tuned_thresh).sum())
            # Fraction of ALL cells (denominator = every finite unit incl. dead),
            # NOT just the non-dead plotted population.
            pct_all = 100.0 * n_tuned / n_total if n_total else float('nan')
            pct_plot = 100.0 * n_tuned / len(vals)
            lab = f'{label}  ({pct_all:.0f}% tuned, {n_tuned}/{n_total})'
            print(f'  {label:20s} tuned ({args.metric}>{args.tuned_thresh:g}): '
                  f'{n_tuned:4d}  ->  {pct_all:5.1f}% of ALL units ({n_total}), '
                  f'{pct_plot:5.1f}% of non-dead plotted ({len(vals)})')
        ax.hist(vals, bins=bins, density=(args.density and not args.fraction),
                weights=weights, histtype='stepfilled', zorder=z,
                color=col, alpha=0.45, edgecolor=col, linewidth=1.4, label=lab)
        ax.axvline(np.median(vals), color=col, linestyle='--', linewidth=1.2,
                   alpha=0.9, zorder=z + 0.5)

    if args.tuned_thresh is not None:
        ax.axvline(args.tuned_thresh, color='0.3', linestyle=':', linewidth=1.6,
                   zorder=20, label=f'tuned: {args.metric} > {args.tuned_thresh:g}')

    ax.set_xlabel(disp)
    ax.set_ylabel('Fraction of neurons' if args.fraction
                  else ('Density' if args.density else 'Count'))
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
        vals, _ = _load_metric(path, args.layer, args.metric,
                               exclude_dead=args.exclude_dead, dead_tol=args.dead_tol)
        print(f'  {label:28s} N={len(vals):5d} median={np.median(vals):.3f} '
              f'mean={np.mean(vals):.3f}')


if __name__ == '__main__':
    main()
