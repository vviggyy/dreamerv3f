"""
Position decoder for DreamerV3 world model representations.

Trains linear decoders to predict agent (x, y) position from world model
latent states (deter, stoch, or both), following the approach in the pRNN
repo (LinearDecoder.py). Supports both classification (cross-entropy over
grid cells, as in pRNN) and Ridge regression.

Usage:
  python dreamerv3/decode_position.py \
    --data ./logdir/crafter_small_1m/trajectories \
    --save ./logdir/crafter_small_1m/decoder_results \
    --method both

Methods:
  classification  - pRNN-style: linear layer, CrossEntropyLoss over grid cells
  regression      - Ridge regression (sklearn), predicts continuous (x, y)
  both            - Run both methods
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut

# ---------------------------------------------------------------------------
# Data loading (reuses plot_trajectories.py conventions)
# ---------------------------------------------------------------------------

def load_episodes(data_path):
    data_path = Path(data_path)
    all_file = data_path / 'all_episodes.pkl'
    if all_file.exists():
        with open(all_file, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, dict) and 'episodes' in data:
            metadata = {k: v for k, v in data.items() if k != 'episodes'}
            return data['episodes'], metadata
        return data, None
    episodes = []
    for ep_file in sorted(data_path.glob('episode_*.pkl')):
        with open(ep_file, 'rb') as f:
            episodes.append(pickle.load(f))
    return episodes, None


def prepare_data(episodes):
    """Extract aligned (features, positions, groups) from episode list.

    Returns:
        deter:  np.ndarray (N, D_deter)
        stoch:  np.ndarray (N, D_stoch)  -- flattened from (stoch, classes)
        pos:    np.ndarray (N, 2)        -- agent (x, y)
        groups: np.ndarray (N,)          -- episode index per sample
    """
    all_deter, all_stoch, all_pos, all_groups = [], [], [], []
    for i, ep in enumerate(episodes):
        if 'deter' not in ep or 'player_pos' not in ep:
            print(f"  Skipping episode {i+1}: missing deter or player_pos")
            continue
        d = np.array(ep['deter'], dtype=np.float32)
        s = np.array(ep['stoch'], dtype=np.float32)
        p = np.array(ep['player_pos'], dtype=np.float32)
        # Align lengths (should already match, but be safe)
        T = min(len(d), len(s), len(p))
        d, s, p = d[:T], s[:T], p[:T]
        # Flatten stoch: (T, stoch_dim, classes) -> (T, stoch_dim*classes)
        if s.ndim == 3:
            s = s.reshape(T, -1)
        all_deter.append(d)
        all_stoch.append(s)
        all_pos.append(p)
        all_groups.append(np.full(T, i))
    deter = np.concatenate(all_deter)
    stoch = np.concatenate(all_stoch)
    pos = np.concatenate(all_pos)
    groups = np.concatenate(all_groups)
    return deter, stoch, pos, groups


# ---------------------------------------------------------------------------
# Classification decoder (pRNN-style)
# ---------------------------------------------------------------------------

try:
    import torch
    from torch import nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class LinearClassifier:
    """Single linear layer, no bias, CrossEntropyLoss -- mirrors pRNN's
    linearDecoder (LinearDecoder.py:15-102)."""

    def __init__(self, n_units, n_classes, lr=1e-3, weight_decay=0.3):
        self.n_classes = n_classes
        self.model = nn.Sequential(nn.Linear(n_units, n_classes, bias=False))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = nn.CrossEntropyLoss()

    def fit(self, X, y_idx, batch_frac=0.75, n_iters=5000, verbose=True):
        """Train on (X, y_idx) where y_idx are integer class labels."""
        device = 'cpu'
        self.model.to(device)
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        y_t = torch.tensor(y_idx, dtype=torch.long, device=device)
        N = X_t.shape[0]
        batch_size = max(1, int(batch_frac * N))
        for step in range(n_iters):
            idx = np.random.choice(N, batch_size, replace=False)
            logits = self.model(X_t[idx])
            loss = self.loss_fn(logits, y_t[idx])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if verbose and (step % 1000 == 0 or step == n_iters - 1):
                print(f"  [{step:>5d}/{n_iters}] loss={loss.item():.4f}")

    def predict(self, X):
        """Return predicted class indices."""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X, dtype=torch.float32))
        return logits.argmax(dim=1).numpy()

    def predict_proba(self, X):
        """Return softmax probabilities (N, n_classes)."""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X, dtype=torch.float32))
            probs = torch.softmax(logits, dim=1)
        return probs.numpy()


def classification_decode(X, pos, groups, width, height, n_iters=5000):
    """Leave-one-episode-out classification decoding (pRNN-style).

    Position (x,y) is linearized into width*height classes.  Returns
    per-fold Manhattan distance errors, shuffle baseline, and softmax
    probabilities reshaped to (N, width, height).
    """
    pos_int = pos.astype(int)
    pos_int[:, 0] = np.clip(pos_int[:, 0], 0, width - 1)
    pos_int[:, 1] = np.clip(pos_int[:, 1], 0, height - 1)
    y_cls = np.ravel_multi_index((pos_int[:, 0], pos_int[:, 1]), (width, height))

    logo = LeaveOneGroupOut()
    all_errors, all_shuffle = [], []
    pred_all = np.zeros_like(pos_int)
    true_all = pos_int.copy()
    proba_all = np.zeros((len(pos), width, height), dtype=np.float32)

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y_cls, groups)):
        print(f"  Classification fold {fold+1} "
              f"(train={len(train_idx)}, test={len(test_idx)})")
        clf = LinearClassifier(X.shape[1], width * height)
        clf.fit(X[train_idx], y_cls[train_idx], n_iters=n_iters, verbose=False)
        pred_cls = clf.predict(X[test_idx])
        proba = clf.predict_proba(X[test_idx])  # (N_test, width*height)
        pred_xy = np.stack(np.unravel_index(pred_cls, (width, height)), axis=1)
        true_xy = pos_int[test_idx]
        err = np.sum(np.abs(pred_xy - true_xy), axis=1)  # Manhattan
        shuf = np.sum(np.abs(
            np.column_stack([np.random.randint(0, width, len(test_idx)),
                             np.random.randint(0, height, len(test_idx))])
            - true_xy), axis=1)
        all_errors.append(err)
        all_shuffle.append(shuf)
        pred_all[test_idx] = pred_xy
        proba_all[test_idx] = proba.reshape(-1, width, height)
    return (np.concatenate(all_errors), np.concatenate(all_shuffle),
            pred_all, true_all, proba_all)


# ---------------------------------------------------------------------------
# Ridge regression decoder
# ---------------------------------------------------------------------------

def ridge_decode_cv(X, pos, groups):
    """Leave-one-episode-out Ridge regression.  Returns R², MAE per fold
    and overall, plus per-neuron R² (univariate decoding per feature)."""
    logo = LeaveOneGroupOut()
    fold_r2, fold_mae = [], []
    pred_all = np.zeros_like(pos)

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, pos, groups)):
        model = RidgeCV(alphas=np.logspace(-2, 4, 20))
        model.fit(X[train_idx], pos[train_idx])
        pred = model.predict(X[test_idx])
        pred_all[test_idx] = pred
        r2 = r2_score(pos[test_idx], pred, multioutput='uniform_average')
        mae = mean_absolute_error(pos[test_idx], pred)
        fold_r2.append(r2)
        fold_mae.append(mae)
        print(f"  Fold {fold+1}: R²={r2:.4f}  MAE={mae:.3f}")

    overall_r2 = r2_score(pos, pred_all, multioutput='uniform_average')
    overall_mae = mean_absolute_error(pos, pred_all)
    r2_x = r2_score(pos[:, 0], pred_all[:, 0])
    r2_y = r2_score(pos[:, 1], pred_all[:, 1])
    return {
        'fold_r2': fold_r2, 'fold_mae': fold_mae,
        'overall_r2': overall_r2, 'overall_mae': overall_mae,
        'r2_x': r2_x, 'r2_y': r2_y,
        'pred': pred_all,
    }


def per_neuron_r2(X, pos, groups):
    """Decode (x,y) from each individual neuron.  Returns (n_features, 2)
    array of R² values [r2_x, r2_y] per neuron."""
    logo = LeaveOneGroupOut()
    n_feat = X.shape[1]
    r2 = np.full((n_feat, 2), np.nan)
    for j in range(n_feat):
        xj = X[:, j:j+1]
        pred = np.zeros_like(pos)
        for train_idx, test_idx in logo.split(xj, pos, groups):
            model = RidgeCV(alphas=np.logspace(-2, 4, 10))
            model.fit(xj[train_idx], pos[train_idx])
            pred[test_idx] = model.predict(xj[test_idx])
        r2_x = r2_score(pos[:, 0], pred[:, 0])
        r2_y = r2_score(pos[:, 1], pred[:, 1])
        r2[j] = [r2_x, r2_y]
    return r2


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_regression_summary(results, layer_name, save_dir, pos):
    """Bar chart of R² per representation + decoded vs true scatter."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # R² bars
    ax = axes[0]
    names = list(results.keys())
    r2s = [results[n]['overall_r2'] for n in names]
    colors = ['#2196F3', '#4CAF50', '#FF9800'][:len(names)]
    ax.bar(names, r2s, color=colors)
    ax.set_ylabel('R² (cross-validated)')
    ax.set_title(f'Position decoding: {layer_name}')
    ax.set_ylim(0, max(1, max(r2s) * 1.15))
    for i, v in enumerate(r2s):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

    # Decoded vs true (best model)
    best = max(results, key=lambda k: results[k]['overall_r2'])
    pred = results[best]['pred']
    for dim, label in enumerate(['x', 'y']):
        ax = axes[dim + 1]
        ax.scatter(pos[:, dim], pred[:, dim], s=4, alpha=0.5)
        lo = min(pos[:, dim].min(), pred[:, dim].min()) - 1
        hi = max(pos[:, dim].max(), pred[:, dim].max()) + 1
        ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8)
        r2_dim = r2_score(pos[:, dim], pred[:, dim])
        ax.set_xlabel(f'True {label}')
        ax.set_ylabel(f'Decoded {label}')
        ax.set_title(f'{best} → {label}  (R²={r2_dim:.3f})')

    fig.tight_layout()
    fig.savefig(save_dir / f'regression_summary.png', dpi=150)
    plt.close(fig)
    print(f"  Saved regression_summary.png")


def plot_classification_summary(errors, shuffle, layer_name, save_dir):
    """Histogram of Manhattan decoding error vs shuffle."""
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.arange(0, max(errors.max(), shuffle.max()) + 2) - 0.5
    ax.hist(errors, bins=bins, alpha=0.6, label=f'Decoder (mean={errors.mean():.2f})',
            density=True)
    ax.hist(shuffle, bins=bins, alpha=0.4, label=f'Shuffle (mean={shuffle.mean():.2f})',
            density=True, color='grey')
    ax.set_xlabel('Manhattan distance error')
    ax.set_ylabel('Density')
    ax.set_title(f'Classification decoder: {layer_name}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir / f'classification_{layer_name}.png', dpi=150)
    plt.close(fig)
    print(f"  Saved classification_{layer_name}.png")


def plot_decoder_probmap(proba_all, pos, groups, layer_name, save_dir,
                         n_steps=12, episode=0):
    """Show decoder softmax probability over the grid with true agent position.

    Picks evenly-spaced timesteps from one episode and plots:
      - Heatmap of P(position) from the classification decoder
      - Red dot at the agent's true (x, y)
      - White star at the argmax decoded position
    """
    ep_mask = groups == episode
    ep_proba = proba_all[ep_mask]   # (T_ep, width, height)
    ep_pos = pos[ep_mask].astype(int)
    T = len(ep_proba)
    step_indices = np.linspace(0, T - 1, n_steps, dtype=int)

    ncols = min(6, n_steps)
    nrows = int(np.ceil(n_steps / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
    axes = np.atleast_2d(axes)

    # Compute shared extent from occupied positions (zoom to relevant region)
    x_min, x_max = int(pos[:, 0].min()) - 1, int(pos[:, 0].max()) + 2
    y_min, y_max = int(pos[:, 1].min()) - 1, int(pos[:, 1].max()) + 2

    for idx, ax in enumerate(axes.flat):
        if idx >= n_steps:
            ax.axis('off')
            continue
        t = step_indices[idx]
        p_grid = ep_proba[t]  # (width, height)
        # Crop to occupied region for visibility
        p_crop = p_grid[x_min:x_max, y_min:y_max]
        im = ax.imshow(p_crop.T, origin='lower', aspect='equal',
                       cmap='hot', interpolation='nearest',
                       extent=[x_min, x_max, y_min, y_max])
        # True position
        tx, ty = ep_pos[t]
        ax.plot(tx + 0.5, ty + 0.5, 'o', color='cyan', markersize=7,
                markeredgecolor='white', markeredgewidth=1.0, label='true')
        # Argmax decoded position
        dec_x, dec_y = np.unravel_index(p_grid.argmax(), p_grid.shape)
        ax.plot(dec_x + 0.5, dec_y + 0.5, '*', color='lime', markersize=9,
                markeredgecolor='white', markeredgewidth=0.5, label='decoded')
        ax.set_title(f't={t}', fontsize=8)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.tick_params(labelsize=6)
        if idx == 0:
            ax.legend(fontsize=6, loc='upper left')

    fig.suptitle(f'Decoder P(pos) — {layer_name}, episode {episode+1}',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(save_dir / f'probmap_{layer_name}_ep{episode+1}.png', dpi=150)
    plt.close(fig)
    print(f"  Saved probmap_{layer_name}_ep{episode+1}.png")


def plot_per_neuron(r2_deter, r2_stoch, save_dir):
    """Scatter of per-neuron R² for x and y, for deter and stoch."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, r2_arr, name in [(axes[0], r2_deter, 'deter'),
                              (axes[1], r2_stoch, 'stoch')]:
        r2_mean = r2_arr.mean(axis=1)  # average of r2_x, r2_y
        order = np.argsort(r2_mean)[::-1]
        ax.scatter(r2_arr[:, 0], r2_arr[:, 1], s=8, alpha=0.6)
        ax.set_xlabel('R² (x)')
        ax.set_ylabel('R² (y)')
        ax.set_title(f'Per-neuron decoding: {name} (n={len(r2_arr)})')
        ax.axhline(0, color='grey', lw=0.5)
        ax.axvline(0, color='grey', lw=0.5)
        # Annotate top 5
        for rank in range(min(5, len(order))):
            j = order[rank]
            ax.annotate(f'{j}', (r2_arr[j, 0], r2_arr[j, 1]), fontsize=7)
    fig.tight_layout()
    fig.savefig(save_dir / 'per_neuron_r2.png', dpi=150)
    plt.close(fig)
    print(f"  Saved per_neuron_r2.png")


def plot_top_neurons(r2_arr, name, pos, X, groups, save_dir, top_n=6):
    """Plot spatial tuning maps for the top-N decoded neurons."""
    r2_mean = r2_arr.mean(axis=1)
    order = np.argsort(r2_mean)[::-1]
    n = min(top_n, len(order))
    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 7))
    if n == 1:
        axes = axes.reshape(2, 1)
    for col in range(n):
        j = order[col]
        # Top row: activation heatmap (mean activation per position bin)
        ax = axes[0, col]
        xbins = np.arange(int(pos[:, 0].min()), int(pos[:, 0].max()) + 2)
        ybins = np.arange(int(pos[:, 1].min()), int(pos[:, 1].max()) + 2)
        from scipy.stats import binned_statistic_2d
        stat = binned_statistic_2d(
            pos[:, 0], pos[:, 1], X[:, j],
            statistic='mean', bins=[xbins, ybins])
        im = ax.imshow(stat.statistic.T, origin='lower', aspect='equal',
                       cmap='viridis')
        ax.set_title(f'{name}[{j}]\nR²={r2_mean[j]:.3f}', fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Bottom row: activation timeseries colored by episode
        ax = axes[1, col]
        unique_groups = np.unique(groups)
        for g in unique_groups:
            mask = groups == g
            ax.plot(np.where(mask)[0], X[mask, j], lw=0.5, alpha=0.7,
                    label=f'ep{int(g)+1}')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Activation')
        if col == 0:
            ax.legend(fontsize=6)

    fig.suptitle(f'Top-{n} spatially informative {name} neurons', fontsize=12)
    fig.tight_layout()
    fig.savefig(save_dir / f'top_neurons_{name}.png', dpi=150)
    plt.close(fig)
    print(f"  Saved top_neurons_{name}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Decode agent position from world model states')
    parser.add_argument('--data', required=True, help='Path to trajectory pkl directory')
    parser.add_argument('--save', default=None, help='Output directory for plots/results')
    parser.add_argument('--method', default='both', choices=['classification', 'regression', 'both'])
    parser.add_argument('--n_iters', type=int, default=5000,
                        help='Training iterations for classification decoder')
    parser.add_argument('--per_neuron', action='store_true', default=True,
                        help='Run per-neuron R² analysis (regression only)')
    parser.add_argument('--no_per_neuron', dest='per_neuron', action='store_false')
    args = parser.parse_args()

    data_path = Path(args.data)
    save_dir = Path(args.save) if args.save else data_path.parent / 'decoder_results'
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading trajectory data...")
    episodes, metadata = load_episodes(data_path)
    print(f"  {len(episodes)} episodes loaded")
    if metadata:
        area = metadata.get('area', None)
        print(f"  Metadata: {metadata}")

    deter, stoch, pos, groups = prepare_data(episodes)
    combined = np.concatenate([deter, stoch], axis=1)
    print(f"  deter: {deter.shape}  stoch: {stoch.shape}  "
          f"combined: {combined.shape}  pos: {pos.shape}")
    print(f"  Position range: x=[{pos[:,0].min():.1f}, {pos[:,0].max():.1f}]  "
          f"y=[{pos[:,1].min():.1f}, {pos[:,1].max():.1f}]")

    # Determine grid size
    if metadata and 'area' in metadata:
        width, height = metadata['area']
    else:
        width = int(pos[:, 0].max()) + 1
        height = int(pos[:, 1].max()) + 1
    print(f"  Grid: {width}x{height} ({width*height} cells)")

    representations = {'deter': deter, 'stoch': stoch, 'combined': combined}

    # ---- Ridge regression ----
    if args.method in ('regression', 'both'):
        print("\n=== Ridge Regression Decoding ===")
        reg_results = {}
        for name, X in representations.items():
            print(f"\n--- {name} ({X.shape[1]} dims) ---")
            res = ridge_decode_cv(X, pos, groups)
            reg_results[name] = res
            print(f"  Overall R²={res['overall_r2']:.4f}  "
                  f"R²_x={res['r2_x']:.4f}  R²_y={res['r2_y']:.4f}  "
                  f"MAE={res['overall_mae']:.3f}")

        plot_regression_summary(reg_results, 'all', save_dir, pos)

        # Per-neuron analysis
        if args.per_neuron:
            print("\n--- Per-neuron R² (deter) ---")
            r2_deter = per_neuron_r2(deter, pos, groups)
            r2_mean_d = r2_deter.mean(axis=1)
            top5_d = np.argsort(r2_mean_d)[::-1][:5]
            for j in top5_d:
                print(f"  deter[{j}]: R²_x={r2_deter[j,0]:.4f}  "
                      f"R²_y={r2_deter[j,1]:.4f}  mean={r2_mean_d[j]:.4f}")

            print("\n--- Per-neuron R² (stoch) ---")
            r2_stoch = per_neuron_r2(stoch, pos, groups)
            r2_mean_s = r2_stoch.mean(axis=1)
            top5_s = np.argsort(r2_mean_s)[::-1][:5]
            for j in top5_s:
                print(f"  stoch[{j}]: R²_x={r2_stoch[j,0]:.4f}  "
                      f"R²_y={r2_stoch[j,1]:.4f}  mean={r2_mean_s[j]:.4f}")

            plot_per_neuron(r2_deter, r2_stoch, save_dir)
            plot_top_neurons(r2_deter, 'deter', pos, deter, groups, save_dir)
            plot_top_neurons(r2_stoch, 'stoch', pos, stoch, groups, save_dir)

    # ---- Classification (pRNN-style) ----
    if args.method in ('classification', 'both'):
        if not HAS_TORCH:
            print("\nSkipping classification decoder: torch not installed")
        else:
            print("\n=== Classification Decoding (pRNN-style) ===")
            for name, X in representations.items():
                print(f"\n--- {name} ({X.shape[1]} dims) ---")
                errors, shuffle, pred_xy, true_xy, proba = classification_decode(
                    X, pos, groups, width, height, n_iters=args.n_iters)
                print(f"  Mean Manhattan error: {errors.mean():.3f} "
                      f"(shuffle: {shuffle.mean():.3f})")
                plot_classification_summary(errors, shuffle, name, save_dir)
                # Probability heatmap for each episode (deter only to avoid clutter)
                if name == 'deter':
                    for ep_idx in range(len(np.unique(groups))):
                        plot_decoder_probmap(
                            proba, pos, groups, name, save_dir,
                            n_steps=12, episode=ep_idx)

    # Save numerical results
    results_file = save_dir / 'decode_results.pkl'
    save_data = {
        'representations': list(representations.keys()),
        'n_samples': len(pos),
        'n_episodes': len(np.unique(groups)),
        'grid': (width, height),
    }
    if args.method in ('regression', 'both'):
        save_data['regression'] = {
            name: {k: v for k, v in res.items() if k != 'pred'}
            for name, res in reg_results.items()
        }
        if args.per_neuron:
            save_data['per_neuron_r2_deter'] = r2_deter
            save_data['per_neuron_r2_stoch'] = r2_stoch
    with open(results_file, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nResults saved to {results_file}")
    print("Done.")
