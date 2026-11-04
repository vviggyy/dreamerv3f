#!/usr/bin/env python
"""Rebuild the step-12 condition-overlay SW-over-time plot from the per-condition
manifold_results.pkl files (manifold_{A,B,C,D}), WITHOUT the deleted dream_deter
dumps. Each per-condition pkl already stored sw_min_dists + sw_t_idx against the
shared wake manifold, which is exactly what the overlay needs (200k pts/cond).
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, 'dreamerv3')
import manifold_analysis as M  # noqa: E402

DSA = Path('logdir/default_masked_25m_cowmobsweep_cow2p0x_mob0p5x/'
           'dream_seed_ablation_16')
SAVE = DSA / 'manifold_conditions'
SAVE.mkdir(exist_ok=True)
LAYER = 'dyn/deter'

cond_results = {}
for c in ['A', 'B', 'C', 'D']:
    pkl = DSA / f'manifold_{c}' / 'manifold_results.pkl'
    if not pkl.exists():
        print(f'  {c}: missing {pkl}, skipping'); continue
    layer = pickle.load(open(pkl, 'rb'))[LAYER]
    cond_results[c] = {
        'min_dists': layer['sw_min_dists'],
        't_idx': layer['sw_t_idx'],
        'median_sw': float(layer['sw_median']),
        'n_dream': int(len(layer['sw_min_dists'])),
    }
    print(f"  {c}: {cond_results[c]['n_dream']} pts, median SW="
          f"{cond_results[c]['median_sw']:.4f}")

if not cond_results:
    sys.exit('No per-condition pkls found.')

M.plot_sw_over_time_by_condition(cond_results, LAYER, SAVE, shading='band')

# Save the replot source so it regenerates like the pipeline's output.
out_pkl = SAVE / 'manifold_condition_results.pkl'
with open(out_pkl, 'wb') as f:
    pickle.dump({LAYER: cond_results}, f)
print(f'\nSaved overlay figure + {out_pkl.name} to {SAVE}')
