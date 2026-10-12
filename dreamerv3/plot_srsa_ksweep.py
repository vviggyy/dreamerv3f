#!/usr/bin/env python
"""sRSA-vs-k line plot: vanilla baseline vs the masked invert_reg sweep.

Pulls ``srsa_rho`` (Spearman rank corr of spatial vs neural distances, computed by
``manifold_analysis.py`` and stored per-layer in ``manifold_results.pkl``) for the
masked invert_reg networks at k = 1, 3, 5, 10 and overlays a horizontal vanilla
baseline.

Several of the masked_k runs live in the scratch archive rather than ``./logdir``;
each default path below is resolved against both roots (first hit wins), so the
plot works whether or not the runs have been restored. k=6 is intentionally
excluded: only a non-regularized ``masked_k6_invert`` exists, which is not
comparable to the invert_reg sweep.

Example:
    MPLBACKEND=Agg python dreamerv3/plot_srsa_ksweep.py \
        --layer dyn/deter \
        --save logdir/masked_k5_invert_reg_wrld45/plots_ksweep
"""
import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Roots searched (in order) for every relative run path below.
SEARCH_ROOTS = [
    Path('logdir'),
    Path('/gpfs/radev/scratch/levenstein/vv266/logdir_archive'),
]

# Masked invert_reg sweep: k -> (run dir, manifold subdir). k=6 dropped (no reg run).
MASKED_RUNS = {
    1:  ('masked_k1_invert_reg',        'manifold_results'),
    3:  ('masked_k3_invert_reg',        'manifold_results'),
    5:  ('masked_k5_invert_reg_wrld45', 'manifold_results'),
    10: ('masked_k10_invert_reg',       'manifold_results'),
}

# Vanilla baseline (fully trained, world 45, unmasked).
VANILLA_RUN = ('vanilla_wrld45_fulltrain', 'manifold_results')


def resolve(run_dir, subdir):
    """Return the first existing manifold_results.pkl across SEARCH_ROOTS, else None."""
    for root in SEARCH_ROOTS:
        p = root / run_dir / subdir / 'manifold_results.pkl'
        if p.exists():
            return p
    return None


def get_srsa(run_dir, subdir, layer):
    """Load srsa_rho for a layer; returns (rho, resolved_path) with rho=nan if missing."""
    p = resolve(run_dir, subdir)
    if p is None:
        print(f'  MISSING: {run_dir}/{subdir}/manifold_results.pkl (searched all roots)')
        return np.nan, None
    with open(p, 'rb') as f:
        d = pickle.load(f)
    if layer not in d or 'srsa_rho' not in d[layer]:
        print(f'  NO sRSA for layer {layer} in {p}')
        return np.nan, p
    return float(d[layer]['srsa_rho']), p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--layer', default='dyn/deter',
                    help='Layer whose srsa_rho to plot (default dyn/deter; masked runs only have dyn/deter)')
    ap.add_argument('--save', default='logdir/masked_k5_invert_reg_wrld45/plots_ksweep',
                    help='Output directory')
    ap.add_argument('--out_name', default=None,
                    help='Output filename (default srsa_ksweep_<layer>.svg)')
    args = ap.parse_args()

    layer = args.layer
    slug = layer.replace('/', '_')

    print(f'Layer: {layer}')
    print('Masked invert_reg sweep:')
    ks, vals = [], []
    for k in sorted(MASKED_RUNS):
        run_dir, subdir = MASKED_RUNS[k]
        rho, _ = get_srsa(run_dir, subdir, layer)
        print(f'  k={k:<2d} {run_dir:<30s} sRSA={rho:.4f}')
        ks.append(k)
        vals.append(rho)
    ks = np.array(ks)
    vals = np.array(vals)

    print('Vanilla baseline:')
    van_rho, _ = get_srsa(*VANILLA_RUN, layer)
    print(f'  {VANILLA_RUN[0]:<30s} sRSA={van_rho:.4f}')

    # --- plot ---
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    if not np.isnan(van_rho):
        ax.axhline(van_rho, color='#7f7f7f', ls='--', lw=1.8, zorder=0,
                   label=f'Vanilla (wrld45)  ρ={van_rho:.3f}')

    finite = ~np.isnan(vals)
    ax.plot(ks[finite], vals[finite], 'o-', color='#d62728', markersize=7,
            lw=2.0, label='Masked (invert_reg)')
    for k, v in zip(ks[finite], vals[finite]):
        ax.annotate(f'{v:.3f}', (k, v), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=8, color='#d62728')

    ax.set_xlabel('k (mask span)')
    ax.set_ylabel('sRSA (Spearman ρ)')
    ax.set_title(f'Spatial representational structure vs k — {layer}', fontsize=11)
    ax.set_xticks(sorted(MASKED_RUNS))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc='best')
    fig.tight_layout()

    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_name = args.out_name or f'srsa_ksweep_{slug}.svg'
    out = save_dir / out_name
    fig.savefig(out, bbox_inches='tight')
    # also drop a png for quick viewing
    fig.savefig(out.with_suffix('.png'), dpi=150, bbox_inches='tight')
    print(f'\nSaved: {out}')
    print(f'Saved: {out.with_suffix(".png")}')


if __name__ == '__main__':
    main()
