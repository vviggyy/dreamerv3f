"""Decode per-vital status-bar PIXELS from world-model activations.

STEP 2 of the "does the WM represent the vital-bar pixels better than other
pixels?" experiment (Step 1 = localize_vital_pixels.py, which produces the
per-vital pixel masks this script consumes).

Hypothesis: if the vitals are behaviorally important, the world model should
preserve the pixels that render each vital's status icon MORE faithfully than
arbitrary pixels. We test this by training a linear decoder

      deter  ->  the RGB values of a vital's ~20-pixel tile       (regression)

and comparing its reconstruction error against two baselines:
  * within-bar  : the OTHER inventory icons (wood/stone/coal/...) -- same bottom
                  bar, same tile size, same rendering; isolates "vitals vs other
                  equally-stable bar pixels".
  * random null : many matched-size sets of random NON-bar pixels -> a null
                  distribution of error for "decode N arbitrary pixels".

Error metrics (BOTH reported, per the design):
  * R2 (variance-weighted)  -- fraction of pixel variance explained. HEADLINE:
        normalizes out constant/low-variance pixels being trivially easy, which
        raw error does not. 1 - sum(SS_res)/sum(SS_tot) over all pixel-channels.
  * RMSE (in [0,1] and 0-255) -- raw intensity error, for interpretability.

Cross-entropy is deliberately NOT used: pixel intensities are not a probability
distribution. This is a regression (MSE-fit ridge), so the error is MSE/R2.

All targets (vitals, items, null draws) share the SAME deter design matrix, so we
factorize the ridge normal equations ONCE per train/test split and solve each
target by back-substitution -- making the large null distribution cheap.

Requires a FULL eval trajectory set (deter + image recorded, RUN_ANALYSIS=true);
vitals_only runs have no activations. The pixel MASKS come from
localize_vital_pixels.py run on the SAME env config (so the bar lands identically).

Usage:
  # 1) localize (once per env config):
  PYTHONPATH=.:dreamerv3 python dreamerv3/localize_vital_pixels.py \
      --logdir ./logdir/my_run --save ./logdir/my_run/vital_pixels
  # 2) decode:
  PYTHONPATH=.:dreamerv3 MPLBACKEND=Agg python dreamerv3/decode_pixels.py \
      --data ./logdir/my_run/trajectories \
      --masks ./logdir/my_run/vital_pixels/vital_pixel_masks.pkl \
      --save ./logdir/my_run/pixel_decode --repr deter

  # replot:
  python dreamerv3/decode_pixels.py --from_pkl ./.../pixel_decode_results.pkl
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.linalg as sla

from decode_position import load_episodes
from decode_vitals import episode_holdout, VITALS

VITAL_COLORS = {'health': '#d62728', 'food': '#2ca02c',
                'drink': '#1f77b4', 'energy': '#ff7f0e'}
REPR_CHOICES = ('deter', 'stoch', 'combined')


# ---------------------------------------------------------------------------
# Masks + candidate pixel set
# ---------------------------------------------------------------------------
def _flat(mask):
    return np.flatnonzero(np.asarray(mask).reshape(-1))


def build_candidates(mask_data, n_random, seed):
    """Union of all bar-icon pixels + a pool of random NON-bar pixels.

    We store image values only at these candidate locations (memory-bounded).
    Returns (cand_idx sorted unique flat indices, rand_pool_positions) where
    rand_pool_positions index INTO cand_idx (so null draws are O(1))."""
    H, W = mask_data['img_shape']
    bar = np.zeros(H * W, bool)
    for m in mask_data['vital_masks'].values():
        if m is not None:
            bar[_flat(m)] = True
    for m in mask_data.get('item_masks', {}).values():
        if m is not None:
            bar[_flat(m)] = True
    rng = np.random.RandomState(seed)
    nonbar = np.flatnonzero(~bar)
    rand_pool = rng.choice(nonbar, size=min(n_random, len(nonbar)), replace=False)
    cand = np.union1d(np.flatnonzero(bar), rand_pool)
    pos_of = {int(idx): p for p, idx in enumerate(cand)}
    rand_pool_pos = np.array([pos_of[int(i)] for i in rand_pool], dtype=np.int64)
    return cand, rand_pool_pos, pos_of


def _target_positions(mask, pos_of):
    """Positions (into cand_idx) of a mask's pixels."""
    return np.array([pos_of[int(i)] for i in _flat(mask)], dtype=np.int64)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def prepare(episodes, cand_idx, reprs):
    """Returns X {repr:(N,D)}, P (N, n_cand, 3) float32 in [0,1], groups (N,)."""
    need_stoch = ('stoch' in reprs) or ('combined' in reprs)
    Xd, Xs, Pix, groups = [], [], [], []
    kept = 0
    for i, ep in enumerate(episodes):
        if 'deter' not in ep or 'image' not in ep:
            continue
        d = np.asarray(ep['deter'], np.float32)
        img = np.asarray(ep['image'])
        if img.ndim != 4:
            continue
        T = min(len(d), len(img))
        if T == 0:
            continue
        d, img = d[:T], img[:T]
        H, W = img.shape[1:3]
        flat = img.reshape(T, H * W, 3)[:, cand_idx, :].astype(np.float32) / 255.0
        Xd.append(d)
        if need_stoch:
            if 'stoch' not in ep:
                continue
            s = np.asarray(ep['stoch'], np.float32)[:T]
            if s.ndim == 3:
                s = s.reshape(T, -1)
            Xs.append(s)
        Pix.append(flat)
        groups.append(np.full(T, i))
        kept += 1
    if not Xd:
        raise ValueError(
            "No episodes with BOTH deter and image. This needs a FULL eval "
            "(RUN_ANALYSIS=true); vitals_only runs lack activations.")
    deter = np.concatenate(Xd)
    X = {'deter': deter}
    if need_stoch:
        stoch = np.concatenate(Xs)
        X['stoch'] = stoch
        X['combined'] = np.concatenate([deter, stoch], axis=1)
    P = np.concatenate(Pix)              # (N, n_cand, 3)
    groups = np.concatenate(groups)
    print(f"Prepared {len(groups)} samples from {kept} episodes "
          f"(deter={deter.shape[1]}d, n_cand_pixels={P.shape[1]})")
    return X, P, groups


# ---------------------------------------------------------------------------
# Shared-factorization ridge
# ---------------------------------------------------------------------------
class RidgeSolver:
    """Centered ridge with the normal matrix factorized ONCE; solve many targets.

    All targets share X, so A = Xc^T Xc + alpha I is built & Cholesky-factored
    one time; each target is a cheap cho_solve + matmul."""

    def __init__(self, X_tr, X_te, alpha):
        self.xm = X_tr.mean(0)
        self.Xtr = X_tr - self.xm
        self.Xte = X_te - self.xm
        D = self.Xtr.shape[1]
        A = self.Xtr.T @ self.Xtr + alpha * np.eye(D, dtype=np.float64)
        self.cf = sla.cho_factor(A, lower=True)
        self.XtrT = self.Xtr.T

    def predict(self, Y_tr):
        ym = Y_tr.mean(0)
        W = sla.cho_solve(self.cf, self.XtrT @ (Y_tr - ym))
        return self.Xte @ W + ym


def score(true, pred):
    """true/pred: (Nte, k). Returns dict with variance-weighted R2 + RMSE."""
    diff = pred - true
    ss_res = (diff ** 2).sum(0)                        # per column
    ss_tot = ((true - true.mean(0)) ** 2).sum(0)
    r2 = 1.0 - ss_res.sum() / max(ss_tot.sum(), 1e-12)
    rmse = float(np.sqrt((diff ** 2).mean()))
    return dict(r2=float(r2), rmse=rmse, rmse255=rmse * 255.0,
                var=float(ss_tot.sum() / len(true)))


# ---------------------------------------------------------------------------
# Decode one representation
# ---------------------------------------------------------------------------
def decode_repr(Xr, P, groups, mask_data, pos_of, rand_pool_pos, var_pix,
                alpha, holdout_frac, n_null, seed, min_var, keep_recon=True):
    train_mask, test_mask = episode_holdout(groups, holdout_frac, seed)
    Xtr, Xte = Xr[train_mask], Xr[test_mask]
    Ptr, Pte = P[train_mask], P[test_mask]
    solver = RidgeSolver(Xtr.astype(np.float64), Xte.astype(np.float64), alpha)

    def target(positions):
        return (Ptr[:, positions, :].reshape(len(Ptr), -1),
                Pte[:, positions, :].reshape(len(Pte), -1))

    out = {'vitals': {}, 'items': {}, 'null': {}, 'dropped_items': []}
    recon = {}

    # vitals (always reported, even if near-constant -- but flag low-variance)
    for v in VITALS:
        m = mask_data['vital_masks'].get(v)
        if m is None or not np.asarray(m).any():
            continue
        pos = _target_positions(m, pos_of)
        Ytr, Yte = target(pos)
        pred = solver.predict(Ytr)
        s = score(Yte, pred)
        s['n_px'] = len(pos)
        s['mean_var'] = float(var_pix[pos].mean())
        s['low_var'] = bool(s['mean_var'] < min_var)
        out['vitals'][v] = s
        if keep_recon:
            recon[v] = dict(true=Yte[:16].astype(np.float32),
                            pred=pred[:16].astype(np.float32),
                            bbox=_bbox(m), mask=np.asarray(m))

    # within-bar control items -- DROP near-constant tiles (e.g. items this net
    # never collects render as empty -> variance ~0 -> trivially R2=1, which
    # would unfairly inflate the baseline).
    for it, m in mask_data.get('item_masks', {}).items():
        if m is None or not np.asarray(m).any():
            continue
        pos = _target_positions(m, pos_of)
        if float(var_pix[pos].mean()) < min_var:
            out['dropped_items'].append(it)
            continue
        Ytr, Yte = target(pos)
        s = score(Yte, solver.predict(Ytr))
        s['n_px'] = len(pos)
        s['mean_var'] = float(var_pix[pos].mean())
        out['items'][it] = s

    # random null: matched-size draws per vital, from VARYING non-bar pixels only
    # (constant background would otherwise give trivial R2=1).
    active_pool = rand_pool_pos[var_pix[rand_pool_pos] >= min_var]
    out['null_pool_size'] = int(len(active_pool))
    rng = np.random.RandomState(seed + 1)
    for v in VITALS:
        if v not in out['vitals']:
            continue
        P_v = out['vitals'][v]['n_px']
        if len(active_pool) < P_v:
            continue
        r2s, rmses = [], []
        for _ in range(n_null):
            pos = rng.choice(active_pool, size=P_v, replace=False)
            Ytr, Yte = target(pos)
            s = score(Yte, solver.predict(Ytr))
            r2s.append(s['r2']); rmses.append(s['rmse'])
        out['null'][v] = dict(r2=np.array(r2s), rmse=np.array(rmses), n_px=P_v)

    return out, recon, dict(n_train=int(train_mask.sum()),
                            n_test=int(test_mask.sum()))


def _bbox(mask):
    ys, xs = np.where(np.asarray(mask))
    return (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _pct(null_arr, val, higher_better=True):
    """Percentile of val within the null array."""
    if higher_better:
        return float((null_arr < val).mean() * 100)
    return float((null_arr > val).mean() * 100)


def plot_summary(results, save_dir, repr_name):
    res = results['reprs'][repr_name]
    vitals = [v for v in VITALS if v in res['vitals']]
    x = np.arange(len(vitals))
    fig, (axr, axe) = plt.subplots(1, 2, figsize=(13, 5))

    # --- R2 panel (headline) ---
    vit_r2 = [res['vitals'][v]['r2'] for v in vitals]
    item_r2 = [res['items'][it]['r2'] for it in res['items']]
    axr.bar(x, vit_r2, 0.6, color=[VITAL_COLORS[v] for v in vitals],
            edgecolor='k', label='vital', zorder=3)
    # within-bar control band
    if item_r2:
        axr.axhspan(np.min(item_r2), np.max(item_r2), color='0.75', alpha=0.4,
                    label='within-bar items (range)', zorder=0)
        axr.axhline(np.mean(item_r2), color='0.4', ls='-', lw=1, zorder=1)
    # random null: boxplot per vital
    for i, v in enumerate(vitals):
        if v in res['null']:
            axr.boxplot(res['null'][v]['r2'], positions=[i], widths=0.28,
                        patch_artist=True,
                        boxprops=dict(facecolor='#e0a0a0', alpha=0.6),
                        medianprops=dict(color='r'), showfliers=False, zorder=2)
    for i, v in enumerate(vitals):
        p = _pct(res['null'][v]['r2'], res['vitals'][v]['r2']) if v in res['null'] else np.nan
        axr.text(i, vit_r2[i], f"{vit_r2[i]:.2f}\n(p{p:.0f})", ha='center',
                 va='bottom', fontsize=8)
    axr.set_xticks(x); axr.set_xticklabels(vitals)
    axr.set_ylabel('R$^2$ (variance explained, higher=better)')
    axr.set_title(f'Vital-pixel decodability from {repr_name}\n'
                  'bar=vital, pink box=random-pixel null, gray=within-bar items')
    axr.legend(fontsize=8, loc='lower right')
    axr.grid(axis='y', alpha=0.3)

    # --- RMSE panel ---
    vit_rm = [res['vitals'][v]['rmse255'] for v in vitals]
    item_rm = [res['items'][it]['rmse255'] for it in res['items']]
    axe.bar(x, vit_rm, 0.6, color=[VITAL_COLORS[v] for v in vitals],
            edgecolor='k', zorder=3)
    if item_rm:
        axe.axhspan(np.min(item_rm), np.max(item_rm), color='0.75', alpha=0.4, zorder=0)
    for i, v in enumerate(vitals):
        if v in res['null']:
            axe.boxplot(res['null'][v]['rmse'] * 255, positions=[i], widths=0.28,
                        patch_artist=True,
                        boxprops=dict(facecolor='#e0a0a0', alpha=0.6),
                        medianprops=dict(color='r'), showfliers=False, zorder=2)
    for i in range(len(vitals)):
        axe.text(i, vit_rm[i], f"{vit_rm[i]:.1f}", ha='center', va='bottom', fontsize=8)
    axe.set_xticks(x); axe.set_xticklabels(vitals)
    axe.set_ylabel('RMSE (0-255 intensity, lower=better)')
    axe.set_title('Raw reconstruction error')
    axe.grid(axis='y', alpha=0.3)

    fig.suptitle('Per-vital pixel decoding vs within-bar + random baselines', fontsize=13)
    fig.tight_layout()
    p = Path(save_dir) / f'pixel_decode_summary_{repr_name}.svg'
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {p}")


def plot_recon(recon, save_dir, repr_name, n_examples=6):
    """True vs predicted vital tiles for a few test frames."""
    vitals = list(recon.keys())
    if not vitals:
        return
    n = min(n_examples, next(iter(recon.values()))['true'].shape[0])
    fig, axes = plt.subplots(len(vitals), 2 * n,
                             figsize=(1.3 * 2 * n, 1.5 * len(vitals)),
                             squeeze=False)
    for r, v in enumerate(vitals):
        rec = recon[v]
        y0, y1, x0, x1 = rec['bbox']
        mask = rec['mask'][y0:y1 + 1, x0:x1 + 1]
        h, w = mask.shape
        for k in range(n):
            for c, (arr, lab) in enumerate([(rec['true'], 'true'),
                                            (rec['pred'], 'pred')]):
                ax = axes[r, 2 * k + c]
                canvas = np.zeros((h, w, 3), np.float32)
                canvas[mask] = np.clip(arr[k].reshape(-1, 3), 0, 1)
                ax.imshow(canvas, interpolation='nearest')
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(lab, fontsize=7)
                if c == 0 and k == 0:
                    ax.set_ylabel(v, fontsize=9, color=VITAL_COLORS[v])
    fig.suptitle(f'Reconstructed vital tiles from {repr_name} (test frames)',
                 fontsize=12)
    fig.tight_layout()
    p = Path(save_dir) / f'pixel_decode_recon_{repr_name}.svg'
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {p}")


def make_plots(results, save_dir):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for r in results['reprs']:
        plot_summary(results, save_dir, r)
        if results['recon'].get(r):
            plot_recon(results['recon'][r], save_dir, r)


def print_verdict(results):
    print("\n=== Verdict (R2 vs baselines) ===")
    for r, res in results['reprs'].items():
        print(f"[{r}]")
        for v in VITALS:
            if v not in res['vitals']:
                continue
            vr2 = res['vitals'][v]['r2']
            item_mean = (np.mean([res['items'][it]['r2'] for it in res['items']])
                         if res['items'] else np.nan)
            if v in res['null']:
                p = _pct(res['null'][v]['r2'], vr2)
                nm = res['null'][v]['r2'].mean()
            else:
                p, nm = np.nan, np.nan
            tag = ('ABOVE both' if (vr2 > item_mean and p >= 95)
                   else 'above null only' if p >= 95
                   else 'not above baselines')
            print(f"  {v:7s} R2={vr2:.3f} | items_mean={item_mean:.3f} "
                  f"null_mean={nm:.3f} (p{p:.0f})  -> {tag}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', help='full-eval trajectory dir (deter+image)')
    ap.add_argument('--masks', help='vital_pixel_masks.pkl from localize_vital_pixels.py')
    ap.add_argument('--save', default='./pixel_decode')
    ap.add_argument('--repr', default='deter', choices=list(REPR_CHOICES) + ['all'])
    ap.add_argument('--alpha', type=float, default=10.0, help='ridge regularization')
    ap.add_argument('--holdout_frac', type=float, default=0.2)
    ap.add_argument('--max_episodes', type=int, default=0)
    ap.add_argument('--max_samples', type=int, default=30000,
                    help='subsample timesteps for tractable X^T X (0=all)')
    ap.add_argument('--n_null', type=int, default=50, help='random null draws/vital')
    ap.add_argument('--n_random_pool', type=int, default=3000,
                    help='size of the non-bar random pixel pool to draw null from')
    ap.add_argument('--min_var', type=float, default=0.005,
                    help='min per-pixel intensity variance ([0,1] units) for a '
                    'pixel/tile to count in the baselines -- drops near-constant '
                    'targets (unseen item icons, static background) that would '
                    'trivially score R2~1. Set 0 to disable filtering.')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--from_pkl', default='', help='replot from pixel_decode_results.pkl')
    args = ap.parse_args()

    if args.from_pkl:
        with open(args.from_pkl, 'rb') as f:
            results = pickle.load(f)
        save_dir = args.save if args.save != './pixel_decode' else str(Path(args.from_pkl).parent)
        make_plots(results, save_dir)
        print_verdict(results)
        print(f"\nReplotted from {args.from_pkl} -> {save_dir}")
        return

    if not args.data or not args.masks:
        raise SystemExit("--data and --masks required (unless --from_pkl)")
    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(args.masks, 'rb') as f:
        mask_data = pickle.load(f)
    print(f"Masks: img {mask_data['img_shape']}, "
          f"{sum(1 for m in mask_data['vital_masks'].values() if m is not None)} vitals, "
          f"{len(mask_data.get('item_masks', {}))} control items")

    reprs = list(REPR_CHOICES) if args.repr == 'all' else [args.repr]
    cand_idx, rand_pool_pos, pos_of = build_candidates(
        mask_data, args.n_random_pool, args.seed)

    print(f"Loading episodes from {args.data} ...")
    episodes, _ = load_episodes(args.data, max_episodes=args.max_episodes)
    X, P, groups = prepare(episodes, cand_idx, reprs)

    # subsample timesteps (keep whole episodes' grouping intact for the split)
    if args.max_samples and len(groups) > args.max_samples:
        rng = np.random.RandomState(args.seed)
        sel = rng.choice(len(groups), args.max_samples, replace=False)
        sel.sort()
        X = {k: v[sel] for k, v in X.items()}
        P, groups = P[sel], groups[sel]
        print(f"Subsampled to {len(groups)} samples")

    # per-candidate-pixel variance (mean over RGB) -> defines "active" pixels for
    # the baselines (constant targets are dropped; see decode_repr).
    var_pix = P.var(0).mean(1)                          # (n_cand,)
    n_active = int((var_pix >= args.min_var).sum())
    print(f"Per-pixel variance: median {np.median(var_pix):.4f}, "
          f"{n_active}/{len(var_pix)} candidates active (var>={args.min_var})")

    results = {'reprs': {}, 'recon': {}, 'split': {},
               'masks_meta': {'img_shape': mask_data['img_shape'],
                              'bar_row0': mask_data.get('bar_row0')},
               'args': vars(args)}
    for r in reprs:
        print(f"\n=== repr={r} ({X[r].shape[1]}d) ===")
        out, recon, split = decode_repr(
            X[r], P, groups, mask_data, pos_of, rand_pool_pos, var_pix,
            args.alpha, args.holdout_frac, args.n_null, args.seed, args.min_var)
        if out['dropped_items']:
            print(f"  dropped {len(out['dropped_items'])} constant control items: "
                  f"{out['dropped_items']}")
        results['reprs'][r] = out
        results['recon'][r] = recon
        results['split'] = split
        for v in VITALS:
            if v in out['vitals']:
                s = out['vitals'][v]
                nm = out['null'][v]['r2'].mean() if v in out['null'] else np.nan
                print(f"  {v:7s} ({s['n_px']}px): R2={s['r2']:.3f} "
                      f"RMSE={s['rmse255']:.1f}/255 | null R2 mean={nm:.3f}")

    pkl = save_dir / 'pixel_decode_results.pkl'
    with open(pkl, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved {pkl}")

    make_plots(results, save_dir)
    print_verdict(results)

    try:
        from run_info import log_run_info
        log_run_info(str(save_dir), 'decode_pixels', vars(args), [str(pkl)],
                     extra={'split': results['split']})
    except Exception as e:
        print(f"(run_info not logged: {e})")


if __name__ == '__main__':
    main()
