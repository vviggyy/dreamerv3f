"""Decode interoceptive vitals (health/food/drink/energy) from WM activations.

The behavioral analog of `decode_position.py`. Position decoding asks "does the
world model represent WHERE the agent is?"; this asks "does it represent its own
vitals?". Motivation: if the agent keeps dying of hunger/thirst, maybe the world
model doesn't linearly encode food/drink in its latent state -- it can't act on a
signal it doesn't represent. Each vital is an int 0-9, so we train one 10-class
linear classifier per (vital, representation) -- the SAME `LinearClassifier`
(single linear layer + CrossEntropyLoss) used for position decoding.

Metric: mean absolute error (MAE) in vital points (0-9), plus exact accuracy.

Baselines (both essential -- vitals are heavily skewed, e.g. health usually 9):
  * mode     -- always predict the training-set mode. A strong baseline for
                skewed vitals; the decoder must beat THIS to show it carries
                vital information beyond the marginal prior.
  * shuffle  -- shuffle the test labels (chance under the marginal).
A vital is "decodable" when the decoder MAE is clearly below the mode baseline.

Requires a FULL eval trajectory set (RUN_ANALYSIS=true) so deter/stoch are
recorded -- vitals_only runs have vitals but no activations.

Usage:
  MPLBACKEND=Agg python dreamerv3/decode_vitals.py \
      --data ./logdir/my_run/trajectories \
      --save ./logdir/my_run/vitals_decode \
      --repr all --device cuda

Replot without retraining:
  python dreamerv3/decode_vitals.py --from_pkl ./.../vitals_decode_results.pkl
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from decode_position import load_episodes, LinearClassifier, HAS_TORCH
from run_info import log_run_info

VITALS = ['health', 'food', 'drink', 'energy']
N_LEVELS = 10  # vitals are ints 0..9
VITAL_COLORS = {'health': '#d62728', 'food': '#2ca02c',
                'drink': '#1f77b4', 'energy': '#ff7f0e'}
REPR_CHOICES = ('deter', 'stoch', 'combined')


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def prepare_vitals_data(episodes):
    """Extract aligned activations + vitals.

    Returns (X, Y, groups):
      X: dict {'deter':(N,Dd), 'stoch':(N,Ds), 'combined':(N,Dd+Ds)}
      Y: dict {vital: (N,) int 0..9}
      groups: (N,) episode index per sample (for episode-level train/test split)
    Episodes missing deter/stoch or any vital are skipped."""
    all_deter, all_stoch, all_groups = [], [], []
    all_y = {v: [] for v in VITALS}
    kept = 0
    for i, ep in enumerate(episodes):
        if 'deter' not in ep or 'stoch' not in ep:
            continue
        if any(f'vital_{v}' not in ep for v in VITALS):
            continue
        d = np.asarray(ep['deter'], dtype=np.float32)
        s = np.asarray(ep['stoch'], dtype=np.float32)
        ys = {v: np.asarray(ep[f'vital_{v}']).astype(np.int64) for v in VITALS}
        T = min([len(d), len(s)] + [len(ys[v]) for v in VITALS])
        if T == 0:
            continue
        d, s = d[:T], s[:T]
        if s.ndim == 3:  # (T, stoch_dim, classes) -> (T, stoch_dim*classes)
            s = s.reshape(T, -1)
        all_deter.append(d)
        all_stoch.append(s)
        for v in VITALS:
            all_y[v].append(ys[v][:T])
        all_groups.append(np.full(T, i))
        kept += 1
    if not all_deter:
        raise ValueError(
            "No episodes with BOTH activations (deter/stoch) and vitals.\n"
            "This needs a full eval (RUN_ANALYSIS=true); vitals_only runs have "
            "vitals but no activations.")
    deter = np.concatenate(all_deter)
    stoch = np.concatenate(all_stoch)
    X = {'deter': deter, 'stoch': stoch,
         'combined': np.concatenate([deter, stoch], axis=1)}
    Y = {v: np.concatenate(all_y[v]) for v in VITALS}
    groups = np.concatenate(all_groups)
    print(f"Prepared {len(groups)} samples from {kept} episodes "
          f"(deter={deter.shape[1]}d, stoch={stoch.shape[1]}d)")
    for v in VITALS:
        vals, cnts = np.unique(Y[v], return_counts=True)
        rng = f"{Y[v].min()}-{Y[v].max()}"
        print(f"  vital {v:7s}: range {rng}, mean {Y[v].mean():.2f}, "
              f"mode {vals[cnts.argmax()]} ({cnts.max()/len(Y[v])*100:.0f}%)")
    return X, Y, groups


def episode_holdout(groups, holdout_frac, seed=0):
    """Boolean (train_mask, test_mask) splitting whole EPISODES (no leakage of
    temporally-correlated samples across the split)."""
    eps = np.unique(groups)
    rng = np.random.RandomState(seed)
    rng.shuffle(eps)
    n_test = max(1, int(round(len(eps) * holdout_frac)))
    test_eps = set(eps[:n_test].tolist())
    test_mask = np.isin(groups, list(test_eps))
    return ~test_mask, test_mask


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------
def _decode_one(X_tr, y_tr, X_te, y_te, device, n_iters, lr, wd, seed):
    clf = LinearClassifier(X_tr.shape[1], N_LEVELS, lr=lr,
                           weight_decay=wd, device=device)
    clf.fit(X_tr, y_tr, n_iters=n_iters, verbose=False)
    pred = clf.predict(X_te)
    mae = float(np.mean(np.abs(pred - y_te)))
    acc = float(np.mean(pred == y_te))
    # baselines
    rng = np.random.RandomState(seed)
    shuf = rng.permutation(y_te)
    shuffle_mae = float(np.mean(np.abs(shuf - y_te)))
    mode = int(np.bincount(y_tr, minlength=N_LEVELS).argmax())
    mode_mae = float(np.mean(np.abs(mode - y_te)))
    return dict(mae=mae, acc=acc, shuffle_mae=shuffle_mae, mode_mae=mode_mae,
                mode=mode, frac_mode=float(np.mean(y_te == mode)),
                true=y_te.astype(np.int8), pred=pred.astype(np.int8))


def run_standard(X, Y, groups, reprs, device, n_iters, lr, wd,
                 holdout_frac, seed):
    train_mask, test_mask = episode_holdout(groups, holdout_frac, seed)
    n_tr, n_te = int(train_mask.sum()), int(test_mask.sum())
    print(f"\nEpisode holdout: {n_tr} train / {n_te} test samples\n")
    results = {}
    for r in reprs:
        Xr = X[r]
        X_tr, X_te = Xr[train_mask], Xr[test_mask]
        results[r] = {}
        print(f"repr={r} ({Xr.shape[1]}d):")
        for v in VITALS:
            y = Y[v]
            res = _decode_one(X_tr, y[train_mask], X_te, y[test_mask],
                              device, n_iters, lr, wd, seed)
            results[r][v] = res
            beat = res['mode_mae'] - res['mae']
            print(f"  {v:7s} MAE={res['mae']:.3f} acc={res['acc']:.3f}  "
                  f"| mode={res['mode_mae']:.3f} shuffle={res['shuffle_mae']:.3f}  "
                  f"| beats mode by {beat:+.3f} "
                  f"(mode={res['mode']} @ {res['frac_mode']*100:.0f}%)")
    return results, dict(n_train=n_tr, n_test=n_te)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _primary_repr(results):
    return 'deter' if 'deter' in results else next(iter(results))


def plot_baselines(results, save_dir, repr_name=None):
    """Grouped bars per vital: decoder MAE vs mode vs shuffle baseline."""
    r = repr_name or _primary_repr(results)
    res = results[r]
    x = np.arange(len(VITALS))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.5, 5))
    dec = [res[v]['mae'] for v in VITALS]
    mod = [res[v]['mode_mae'] for v in VITALS]
    shf = [res[v]['shuffle_mae'] for v in VITALS]
    ax.bar(x - w, dec, w, label='decoder', color='#333333')
    ax.bar(x, mod, w, label='mode baseline', color='#bbbbbb')
    ax.bar(x + w, shf, w, label='shuffle baseline', color='#e0a0a0')
    for i, v in enumerate(VITALS):
        ax.text(i - w, dec[i], f"{dec[i]:.2f}", ha='center', va='bottom', fontsize=8)
        ax.text(i - w, -0.03 * max(shf), f"acc {res[v]['acc']:.2f}",
                ha='center', va='top', fontsize=7, color='#333')
    ax.set_xticks(x)
    ax.set_xticklabels(VITALS)
    ax.set_ylabel('MAE (vital points, lower = better)')
    ax.set_title(f'Vital decodability from {r}  (decoder must beat the mode baseline)')
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=0.3, lw=0.5)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    p = Path(save_dir) / f'vitals_decode_baselines_{r}.svg'
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {p}")


def plot_by_repr(results, save_dir):
    """Decoder MAE per vital, grouped by representation (which repr carries it)."""
    reprs = list(results.keys())
    if len(reprs) < 2:
        return
    x = np.arange(len(VITALS))
    w = 0.8 / len(reprs)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for j, r in enumerate(reprs):
        vals = [results[r][v]['mae'] for v in VITALS]
        ax.bar(x + (j - (len(reprs) - 1) / 2) * w, vals, w, label=r)
    # mode baseline reference (repr-independent) as dashed markers
    base = results[reprs[0]]
    ax.plot(x, [base[v]['mode_mae'] for v in VITALS], 'k_', ms=22,
            mew=2, label='mode baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(VITALS)
    ax.set_ylabel('decoder MAE (vital points)')
    ax.set_title('Vital decoding MAE by representation')
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=0.3, lw=0.5)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    p = Path(save_dir) / 'vitals_decode_by_repr.svg'
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {p}")


def plot_confusion(results, save_dir, repr_name=None):
    """2x2 row-normalized confusion heatmaps (true vs predicted vital level)."""
    r = repr_name or _primary_repr(results)
    res = results[r]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, v in zip(axes.ravel(), VITALS):
        true, pred = res[v]['true'].astype(int), res[v]['pred'].astype(int)
        cm = np.zeros((N_LEVELS, N_LEVELS), dtype=float)
        for t, p in zip(true, pred):
            cm[t, p] += 1
        row = cm.sum(1, keepdims=True)
        cmn = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)
        im = ax.imshow(cmn, origin='lower', cmap='magma', vmin=0, vmax=1,
                       aspect='equal')
        ax.plot([0, N_LEVELS - 1], [0, N_LEVELS - 1], color='cyan', lw=0.8, ls='--')
        ax.set_title(f"{v}  (MAE {res[v]['mae']:.2f}, acc {res[v]['acc']:.2f})",
                     color=VITAL_COLORS[v])
        ax.set_xlabel('predicted level')
        ax.set_ylabel('true level')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f'Vital decode confusion from {r} (row-normalized)', fontsize=13)
    fig.tight_layout()
    p = Path(save_dir) / f'vitals_confusion_{r}.svg'
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {p}")


def make_plots(results, save_dir):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for r in results:
        plot_baselines(results, save_dir, r)
        plot_confusion(results, save_dir, r)
    plot_by_repr(results, save_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=str, help='trajectory dir (episode_*.pkl)')
    ap.add_argument('--save', type=str, default='./vitals_decode')
    ap.add_argument('--repr', type=str, default='all',
                    choices=list(REPR_CHOICES) + ['all'])
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--n_iters', type=int, default=3000)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=0.3)
    ap.add_argument('--holdout_frac', type=float, default=0.2)
    ap.add_argument('--max_episodes', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--from_pkl', type=str, default='',
                    help='replot from a saved vitals_decode_results.pkl')
    args = ap.parse_args()

    if args.from_pkl:
        with open(args.from_pkl, 'rb') as f:
            saved = pickle.load(f)
        results = saved['results']
        save_dir = args.save if args.save != './vitals_decode' else str(Path(args.from_pkl).parent)
        make_plots(results, save_dir)
        print(f"\nReplotted from {args.from_pkl} -> {save_dir}")
        return

    if not HAS_TORCH:
        raise SystemExit("torch required for decoding")
    if not args.data:
        raise SystemExit("--data is required (unless --from_pkl)")

    reprs = list(REPR_CHOICES) if args.repr == 'all' else [args.repr]
    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading episodes from {args.data} ...")
    episodes, _ = load_episodes(args.data, max_episodes=args.max_episodes)
    X, Y, groups = prepare_vitals_data(episodes)
    results, split = run_standard(X, Y, groups, reprs, args.device,
                                  args.n_iters, args.lr, args.weight_decay,
                                  args.holdout_frac, args.seed)

    # strip raw arrays optional? keep them -- needed for confusion replot.
    out = {'results': results, 'split': split, 'vitals': VITALS,
           'reprs': reprs, 'args': vars(args)}
    pkl = save_dir / 'vitals_decode_results.pkl'
    with open(pkl, 'wb') as f:
        pickle.dump(out, f)
    print(f"\nSaved {pkl}")

    make_plots(results, save_dir)

    # verdict
    print("\n=== Verdict (decoder vs mode baseline, deter) ===")
    r = _primary_repr(results)
    for v in VITALS:
        d, m = results[r][v]['mae'], results[r][v]['mode_mae']
        tag = 'DECODABLE' if d < 0.75 * m else ('weak' if d < 0.95 * m else 'NOT above prior')
        print(f"  {v:7s}: MAE {d:.3f} vs mode {m:.3f}  -> {tag}")

    try:
        log_run_info(str(save_dir), 'decode_vitals', vars(args),
                     [str(pkl)], extra={'split': split})
    except Exception as e:
        print(f"(run_info not logged: {e})")


if __name__ == '__main__':
    main()
