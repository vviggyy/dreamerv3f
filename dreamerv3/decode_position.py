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

  # Parallel on a GPU cluster (8 CPU workers, classification on CUDA):
  python dreamerv3/decode_position.py \
    --data ./logdir/crafter_small_1m/trajectories \
    --save ./logdir/crafter_small_1m/decoder_results \
    --method both --n_jobs 8 --device cuda

Methods:
  classification  - pRNN-style: linear layer, CrossEntropyLoss over grid cells
  regression      - Ridge regression (sklearn), predicts continuous (x, y)
  both            - Run both methods

Parallelism:
  --n_jobs N      - Parallel workers. All (layer, fold) pairs are submitted as
                    one flat job pool for maximum parallelism. Use -1 for all
                    CPUs. (default: 1)
  --device DEV    - Torch device for classification decoder (cpu, cuda, cuda:0).
                    With n_jobs>1 and device=cuda, jobs are round-robin
                    distributed across all available GPUs. (default: cpu)

Layer-wise decoding (--mode layers):
  --ridge_layers  - Use Ridge regression (closed-form, no GPU) instead of the
                    PyTorch classifier. ~100x faster; metric becomes R² instead
                    of CE loss. Recommended for initial/cluster runs.
  --resume PATH   - Resume from a partial checkpoint. Already-finished layers
                    are skipped. Auto-checkpoint is always saved to
                    <save>/layer_decode_checkpoint.pkl.

  --max_samples N - Subsample to N timesteps before fitting (eliminates O(N)
                    scaling). Default 10000. Set 0 to use all data.
  --max_dims D    - Truncated PCA to D dims before Ridge (eliminates O(D²)
                    scaling). Default 256. Set 0 to disable.

Fast cluster command (completes in minutes):
  python dreamerv3/decode_position.py \\
    --data ./trajectories --save ./decoder_results \\
    --mode layers --ridge_layers --n_jobs -1 \\
    --max_samples 10000 --max_dims 256
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
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

    def __init__(self, n_units, n_classes, lr=1e-3, weight_decay=0.3,
                 device='cpu'):
        self.n_classes = n_classes
        self.device = device
        self.model = nn.Sequential(nn.Linear(n_units, n_classes, bias=False))
        self.model.to(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = nn.CrossEntropyLoss()

    def fit(self, X, y_idx, batch_frac=0.75, n_iters=5000, verbose=True):
        """Train on (X, y_idx) where y_idx are integer class labels."""
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y_idx, dtype=torch.long, device=self.device)
        N = X_t.shape[0]
        batch_size = max(1, int(batch_frac * N))
        self.model.train()
        for step in range(n_iters):
            idx = torch.randint(N, (batch_size,), device=self.device)
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
            logits = self.model(torch.tensor(
                X, dtype=torch.float32, device=self.device))
        return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X):
        """Return softmax probabilities (N, n_classes)."""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(
                X, dtype=torch.float32, device=self.device))
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()


def _classification_fold(fold, train_idx, test_idx, X, y_cls, pos_int,
                         width, height, n_iters, device):
    """Run a single classification fold. Returns (fold, test_idx, err, shuf,
    pred_xy, proba) — designed to be called in parallel."""
    print(f"  Classification fold {fold+1} "
          f"(train={len(train_idx)}, test={len(test_idx)}, device={device})")
    clf = LinearClassifier(X.shape[1], width * height, device=device)
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
    return fold, test_idx, err, shuf, pred_xy, proba


def classification_decode(X, pos, groups, width, height, n_iters=5000,
                          n_jobs=1, device='cpu'):
    """Leave-one-episode-out classification decoding (pRNN-style).

    Position (x,y) is linearized into width*height classes.  Returns
    per-fold Manhattan distance errors, shuffle baseline, and softmax
    probabilities reshaped to (N, width, height).

    Args:
        n_jobs: Number of parallel fold workers. Use -1 for all CPUs.
            When using multiple GPUs, folds are round-robin assigned
            across cuda:0, cuda:1, ... etc.
        device: Base torch device ('cpu', 'cuda', 'cuda:0', etc.).
            With n_jobs>1 and device='cuda', folds are distributed
            across all available GPUs.
    """
    pos_int = pos.astype(int)
    pos_int[:, 0] = np.clip(pos_int[:, 0], 0, width - 1)
    pos_int[:, 1] = np.clip(pos_int[:, 1], 0, height - 1)
    y_cls = np.ravel_multi_index((pos_int[:, 0], pos_int[:, 1]), (width, height))

    logo = LeaveOneGroupOut()
    splits = list(logo.split(X, y_cls, groups))
    n_folds = len(splits)

    # Assign devices: round-robin across GPUs when using CUDA with parallelism
    if HAS_TORCH and device.startswith('cuda') and n_jobs != 1:
        n_gpus = torch.cuda.device_count()
        devices = [f'cuda:{i % n_gpus}' for i in range(n_folds)]
    else:
        devices = [device] * n_folds

    # Run folds in parallel (joblib spawns separate processes so each fold
    # gets its own GPU memory; prefer='threads' would share GIL)
    results = Parallel(n_jobs=n_jobs, prefer='processes')(
        delayed(_classification_fold)(
            fold, train_idx, test_idx, X, y_cls, pos_int,
            width, height, n_iters, devices[fold])
        for fold, (train_idx, test_idx) in enumerate(splits)
    )

    # Reassemble results
    pred_all = np.zeros_like(pos_int)
    true_all = pos_int.copy()
    proba_all = np.zeros((len(pos), width, height), dtype=np.float32)
    all_errors, all_shuffle = [], []
    for fold, test_idx, err, shuf, pred_xy, proba in results:
        all_errors.append(err)
        all_shuffle.append(shuf)
        pred_all[test_idx] = pred_xy
        proba_all[test_idx] = proba.reshape(-1, width, height)
    return (np.concatenate(all_errors), np.concatenate(all_shuffle),
            pred_all, true_all, proba_all)


# ---------------------------------------------------------------------------
# Decoder model save/load (for dream_decode.py)
# ---------------------------------------------------------------------------

def train_full_ridge(X, pos):
    """Train RidgeCV on ALL data (no CV split). Returns fitted model."""
    model = RidgeCV(alphas=np.logspace(-2, 4, 20))
    model.fit(X, pos)
    return model


def save_decoder_model(model, metadata, path):
    """Save a fitted decoder model + metadata to a pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump({'model': model, 'metadata': metadata}, f)
    print(f"  Saved decoder model to {path}")


def load_decoder_model(path):
    """Load a decoder model + metadata from a pickle file.

    Returns (model, metadata) dict.
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['metadata']


def save_classifier_model(clf, metadata, path):
    """Save a LinearClassifier's state_dict + metadata."""
    if not HAS_TORCH:
        raise RuntimeError("torch required to save classifier")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        'state_dict': clf.model.state_dict(),
        'metadata': metadata,
    }
    with open(path, 'wb') as f:
        pickle.dump(state, f)
    print(f"  Saved classifier model to {path}")


def load_classifier_model(path):
    """Load a LinearClassifier from saved state_dict + metadata.

    Returns (clf, metadata).
    """
    if not HAS_TORCH:
        raise RuntimeError("torch required to load classifier")
    with open(path, 'rb') as f:
        state = pickle.load(f)
    meta = state['metadata']
    clf = LinearClassifier(meta['n_units'], meta['n_classes'])
    clf.model.load_state_dict(state['state_dict'])
    return clf, meta


# ---------------------------------------------------------------------------
# Ridge regression decoder
# ---------------------------------------------------------------------------

def _ridge_fold(fold, train_idx, test_idx, X, pos):
    """Run a single Ridge CV fold. Returns (fold, test_idx, pred, r2, mae)."""
    model = RidgeCV(alphas=np.logspace(-2, 4, 20))
    model.fit(X[train_idx], pos[train_idx])
    pred = model.predict(X[test_idx])
    r2 = r2_score(pos[test_idx], pred, multioutput='uniform_average')
    mae = mean_absolute_error(pos[test_idx], pred)
    print(f"  Fold {fold+1}: R²={r2:.4f}  MAE={mae:.3f}")
    return fold, test_idx, pred, r2, mae


def ridge_decode_cv(X, pos, groups, n_jobs=1):
    """Leave-one-episode-out Ridge regression.  Returns R², MAE per fold
    and overall, plus per-neuron R² (univariate decoding per feature).

    Args:
        n_jobs: Number of parallel fold workers. Use -1 for all CPUs.
    """
    logo = LeaveOneGroupOut()
    splits = list(logo.split(X, pos, groups))

    results = Parallel(n_jobs=n_jobs, prefer='processes')(
        delayed(_ridge_fold)(fold, train_idx, test_idx, X, pos)
        for fold, (train_idx, test_idx) in enumerate(splits)
    )

    pred_all = np.zeros_like(pos)
    fold_r2, fold_mae = [], []
    for fold, test_idx, pred, r2, mae in sorted(results, key=lambda x: x[0]):
        pred_all[test_idx] = pred
        fold_r2.append(r2)
        fold_mae.append(mae)

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


def _per_neuron_one(j, X_col, pos, groups):
    """Decode (x,y) from a single neuron. Returns (j, r2_x, r2_y)."""
    logo = LeaveOneGroupOut()
    xj = X_col.reshape(-1, 1)
    pred = np.zeros_like(pos)
    for train_idx, test_idx in logo.split(xj, pos, groups):
        model = RidgeCV(alphas=np.logspace(-2, 4, 10))
        model.fit(xj[train_idx], pos[train_idx])
        pred[test_idx] = model.predict(xj[test_idx])
    r2_x = r2_score(pos[:, 0], pred[:, 0])
    r2_y = r2_score(pos[:, 1], pred[:, 1])
    return j, r2_x, r2_y


def per_neuron_r2(X, pos, groups, n_jobs=1):
    """Decode (x,y) from each individual neuron.  Returns (n_features, 2)
    array of R² values [r2_x, r2_y] per neuron.

    Args:
        n_jobs: Number of parallel workers. Use -1 for all CPUs.
            Each neuron is an independent job, so this parallelizes well.
    """
    n_feat = X.shape[1]
    results = Parallel(n_jobs=n_jobs, prefer='processes')(
        delayed(_per_neuron_one)(j, X[:, j], pos, groups)
        for j in range(n_feat)
    )
    r2 = np.full((n_feat, 2), np.nan)
    for j, r2_x, r2_y in results:
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


def plot_probmap_on_world(proba_all, pos, groups, metadata, layer_name,
                          save_dir, n_steps=12, episode=0, tile_size=8):
    """Overlay decoder P(position) heatmap on the rendered Crafter world map.

    For evenly-spaced timesteps from one episode, shows:
      - Crafter world map as background
      - Semi-transparent probability mass field from classification decoder
      - Agent trajectory drawn up to that timestep
      - Cyan circle at true position, lime star at argmax decoded position
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from plot_trajectories import _render_crafter_world

    world_img, env_seed, tile_size = _render_crafter_world(metadata, tile_size)
    if world_img is None:
        print("Could not render world for probmap overlay — skipping")
        return
    img_h, img_w = world_img.shape[:2]

    ep_mask = groups == episode
    ep_proba = proba_all[ep_mask]   # (T_ep, width, height)
    ep_pos = pos[ep_mask]
    T = len(ep_proba)
    if T == 0:
        print(f"  No data for episode {episode}, skipping probmap_on_world")
        return
    step_indices = np.linspace(0, T - 1, n_steps, dtype=int)

    def pos_to_px(p):
        """Convert grid (x, y) positions to pixel coords on world image."""
        px = p[:, 0] * tile_size + tile_size // 2
        py = img_h - (p[:, 1] * tile_size + tile_size // 2)
        return px, py

    # Zoom bounds from trajectory extent
    all_px, all_py = pos_to_px(ep_pos)
    pad = tile_size * 5
    x_lo = max(0, all_px.min() - pad)
    x_hi = min(img_w, all_px.max() + pad)
    y_lo = max(0, all_py.min() - pad)
    y_hi = min(img_h, all_py.max() + pad)

    ncols = min(4, n_steps)
    nrows = int(np.ceil(n_steps / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows),
                             facecolor='#1a1a1a')
    axes = np.atleast_2d(axes)

    cmap = plt.cm.magma

    for idx, ax in enumerate(axes.flat):
        ax.set_facecolor('#1a1a1a')
        if idx >= n_steps:
            ax.axis('off')
            continue
        t = step_indices[idx]
        p_grid = ep_proba[t]  # (width, height)

        # World background (dimmed slightly so overlay pops)
        ax.imshow((world_img * 0.6).astype(np.uint8))

        # Upscale probability grid to pixel resolution, then apply same
        # transpose + y-flip used by _render_crafter_world
        p_up = np.repeat(np.repeat(p_grid, tile_size, axis=0),
                         tile_size, axis=1)
        p_up = p_up.T[::-1]
        p_max = p_up.max()
        if p_max > 0:
            p_norm = p_up / p_max
        else:
            p_norm = p_up
        overlay_rgba = cmap(p_norm)            # (H, W, 4)
        overlay_rgba[..., 3] = p_norm * 0.85   # alpha ∝ probability
        ax.imshow(overlay_rgba)

        # Trajectory up to this timestep
        traj_px, traj_py = pos_to_px(ep_pos[:t + 1])
        if len(traj_px) > 1:
            ax.plot(traj_px, traj_py, '-', color='white', linewidth=1.5,
                    alpha=0.6, zorder=3, label='trajectory' if idx == 0 else '')

        # True position marker
        tx, ty = pos_to_px(ep_pos[t:t + 1])
        ax.plot(tx, ty, 'o', color='cyan', markersize=8,
                markeredgecolor='white', markeredgewidth=1.2, zorder=5,
                label='true' if idx == 0 else '')

        # Argmax decoded position marker
        dec_x, dec_y = np.unravel_index(p_grid.argmax(), p_grid.shape)
        dec_pos = np.array([[dec_x, dec_y]], dtype=float)
        dpx, dpy = pos_to_px(dec_pos)
        ax.plot(dpx, dpy, '*', color='lime', markersize=11,
                markeredgecolor='white', markeredgewidth=0.7, zorder=5,
                label='decoded' if idx == 0 else '')

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_hi, y_lo)  # y-down for imshow
        ax.set_title(f't={t}', fontsize=9, fontweight='bold',
                     color='white',
                     bbox=dict(facecolor='black', alpha=0.6, pad=2))
        ax.axis('off')
        if idx == 0:
            ax.legend(fontsize=7, loc='upper left',
                      facecolor='black', edgecolor='grey',
                      labelcolor='white')

    fig.suptitle(
        f'Decoder P(pos) on world — {layer_name}, '
        f'episode {episode + 1} (seed={env_seed})',
        fontsize=13, color='white')
    fig.tight_layout()
    out = save_dir / f'probmap_world_{layer_name}_ep{episode + 1}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close(fig)
    print(f"  Saved {out.name}")


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
# Layer-wise decoding helpers
# ---------------------------------------------------------------------------

# Canonical display order: early → late through the network
LAYER_ORDER = [
    'enc/cnn0', 'enc/cnn1', 'enc/cnn2', 'enc/cnn3',
    'enc/mlp0', 'enc/mlp1', 'enc/mlp2',
    'enc/tokens',
    'dyn/stoch', 'dyn/deter',
    'pol/mlp/linear0', 'pol/mlp/linear1', 'pol/mlp/linear2',
    'val/mlp/linear0', 'val/mlp/linear1', 'val/mlp/linear2',
]


def prepare_data_layers(episodes):
    """Extract (layer_name → X array), pos, groups from episodes that have
    per-layer activations recorded under act/* keys.

    Returns:
        layers: dict {layer_name: np.ndarray (N, D)}
        pos:    np.ndarray (N, 2)
        groups: np.ndarray (N,)
    """
    # Discover which layer keys are present
    all_layer_keys = set()
    for ep in episodes:
        all_layer_keys.update(k[len('act/'):] for k in ep if k.startswith('act/'))
    if not all_layer_keys:
        raise ValueError(
            "No 'act/*' keys found in episodes. "
            "Re-record trajectories with --script eval_trajectory "
            "(record_activations is auto-enabled).")

    layers = {k: [] for k in all_layer_keys}
    all_pos, all_groups = [], []

    for i, ep in enumerate(episodes):
        if 'player_pos' not in ep:
            print(f"  Skipping episode {i+1}: missing player_pos")
            continue
        p = np.array(ep['player_pos'], dtype=np.float32)
        T = len(p)
        # Check each layer has matching length
        valid = True
        for ln in all_layer_keys:
            arr = ep.get(f'act/{ln}')
            if arr is None or len(arr) == 0:
                print(f"  Skipping episode {i+1}: missing act/{ln}")
                valid = False
                break
            if len(arr) != T:
                T = min(T, len(arr))
        if not valid:
            continue
        for ln in all_layer_keys:
            arr = np.array(ep[f'act/{ln}'], dtype=np.float32)[:T]
            if arr.ndim > 2:
                arr = arr.reshape(T, -1)
            layers[ln].append(arr)
        all_pos.append(p[:T])
        all_groups.append(np.full(T, i))

    if not all_pos:
        raise ValueError("No valid episodes with layer activations found.")

    layers = {k: np.concatenate(v) for k, v in layers.items()}
    pos = np.concatenate(all_pos)
    groups = np.concatenate(all_groups)
    return layers, pos, groups


def _layer_classification_fold(layer_name, fold, train_idx, test_idx, X,
                                y_cls, width, height, n_iters, device):
    """Run one fold for a single layer; return (layer_name, fold, CE loss)."""
    clf = LinearClassifier(X.shape[1], width * height, device=device)
    clf.fit(X[train_idx], y_cls[train_idx], n_iters=n_iters, verbose=False)
    import torch
    clf.model.eval()
    with torch.no_grad():
        logits = clf.model(torch.tensor(
            X[test_idx], dtype=torch.float32, device=device))
        loss = torch.nn.functional.cross_entropy(
            logits,
            torch.tensor(y_cls[test_idx], dtype=torch.long, device=device))
    return layer_name, fold, float(loss.cpu())


def _layer_ridge_fold(layer_name, fold, train_idx, test_idx, X, pos):
    """Run one Ridge fold for a single layer. Returns (layer_name, fold, r2)."""
    model = RidgeCV(alphas=np.logspace(-2, 4, 20))
    model.fit(X[train_idx], pos[train_idx])
    pred = model.predict(X[test_idx])
    r2 = r2_score(pos[test_idx], pred, multioutput='uniform_average')
    return layer_name, fold, r2


def _ordered_layers(layers):
    ordered = [ln for ln in LAYER_ORDER if ln in layers]
    ordered += sorted(k for k in layers if k not in LAYER_ORDER)
    return ordered


def _save_layer_checkpoint(path, layer_fold_values, ordered, grid, n_samples,
                            n_episodes, metric):
    """Save partial layer decode results to disk (checkpoint)."""
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump({
            'layer_fold_values': layer_fold_values,
            'ordered': ordered,
            'grid': grid,
            'n_samples': n_samples,
            'n_episodes': n_episodes,
            'metric': metric,
        }, f)


def _subsample_layers(layers, pos, groups, max_samples, seed=42):
    """Subsample timesteps proportionally across episodes.

    Returns new (layers, pos, groups) with at most max_samples rows total.
    Stratified by episode so every episode contributes proportionally.
    """
    N = len(pos)
    if max_samples <= 0 or N <= max_samples:
        return layers, pos, groups
    rng = np.random.RandomState(seed)
    unique_g = np.unique(groups)
    per_g = max(1, max_samples // len(unique_g))
    keep = []
    for g in unique_g:
        idx = np.where(groups == g)[0]
        n = min(len(idx), per_g)
        chosen = rng.choice(idx, n, replace=False)
        keep.append(chosen)
    keep = np.sort(np.concatenate(keep))
    print(f"  Subsampled {N} → {len(keep)} timesteps "
          f"(~{per_g} per episode, {len(unique_g)} episodes)")
    layers = {ln: arr[keep] for ln, arr in layers.items()}
    return layers, pos[keep], groups[keep]


def _pca_one_layer(ln, arr, max_dims, seed=42):
    """Fit randomized truncated SVD on one layer. Returns (ln, reduced_arr)."""
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=max_dims, random_state=seed, n_iter=4)
    reduced = svd.fit_transform(arr).astype(np.float32)
    return ln, reduced


def _pca_layers(layers, max_dims, n_jobs=1, seed=42):
    """Reduce each layer to at most max_dims via randomized truncated SVD.

    Runs all layers in parallel (each SVD is independent).
    PCA is fit on all data (not per-fold); fine for a ranking scan.
    """
    from sklearn.decomposition import TruncatedSVD
    if max_dims <= 0:
        return layers
    to_reduce = {ln: arr for ln, arr in layers.items()
                 if arr.shape[1] > max_dims}
    if not to_reduce:
        return layers

    print(f"  PCA: reducing {len(to_reduce)} layers to {max_dims} dims "
          f"(parallel, n_jobs={n_jobs})")
    results = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_pca_one_layer)(ln, arr, max_dims, seed)
        for ln, arr in to_reduce.items()
    )
    out = dict(layers)  # copy, keep unreduced layers
    for ln, reduced in results:
        out[ln] = reduced
        print(f"    {ln}: {layers[ln].shape[1]} → {max_dims} dims")
    return out


def decode_layers_ridge(layers, pos, groups, n_jobs=1, checkpoint_path=None,
                        max_samples=10000, max_dims=256, n_cv_folds=5):
    """Fast layer scan using Ridge regression (closed-form, no GPU needed).

    Applies subsampling (max_samples) and truncated PCA (max_dims) before
    fitting so that runtime is independent of raw layer dimensionality.
    For deter (4096 dims) with max_dims=256 and max_samples=10000, each
    Ridge fold completes in ~1 second; total run in minutes on any CPU.

    All (layer, fold) pairs run as one flat parallel pool with threads
    (Ridge/BLAS releases the GIL; threads avoid subprocess pickling overhead).

    Args:
        max_samples: Subsample to at most this many timesteps before fitting.
            Set to 0 to disable. Default 10000.
        max_dims: Reduce features to this many dims via truncated PCA.
            Set to 0 to disable. Default 256.
        n_cv_folds: Number of cross-validation folds. Default 5. LOGO
            (leave-one-episode-out) is NOT used because with many episodes
            it creates O(N_episodes) folds which dominates runtime.
        checkpoint_path: Auto-save partial results here; load on startup to
            skip already-finished layers.

    Returns:
        layer_fold_r2: dict {layer_name: [fold_r2, ...]}
        ordered: list of layer names in display order
    """
    ordered = _ordered_layers(layers)

    # Resume: load partial results and skip already-finished layers
    layer_fold_r2 = {}
    if checkpoint_path and Path(checkpoint_path).exists():
        with open(checkpoint_path, 'rb') as f:
            prev = pickle.load(f)
        layer_fold_r2 = prev.get('layer_fold_values', {})
        print(f"  Resumed from checkpoint: {len(layer_fold_r2)} layers done")

    todo = [ln for ln in ordered if ln not in layer_fold_r2]
    print(f"  Layers to process: {len(todo)} / {len(ordered)}")
    if not todo:
        return layer_fold_r2, ordered

    # Subsample timesteps (same indices across all layers)
    layers_todo = {ln: layers[ln] for ln in todo}
    layers_todo, pos_s, groups_s = _subsample_layers(
        layers_todo, pos, groups, max_samples)

    # Per-layer PCA reduction (parallel across layers)
    layers_todo = _pca_layers(layers_todo, max_dims, n_jobs=n_jobs)

    # K-fold CV — NOT leave-one-episode-out: with many episodes LOGO creates
    # O(N_episodes) folds which dominates runtime regardless of per-fold speed.
    from sklearn.model_selection import KFold
    n_episodes = len(np.unique(groups_s))
    n_folds = min(n_cv_folds, n_episodes)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    splits = list(kf.split(pos_s))
    print(f"  CV: {n_folds}-fold KFold  "
          f"(n_episodes={n_episodes}, n_cv_folds arg={n_cv_folds})")

    # Flat job list — use threads: Ridge (BLAS) releases GIL, no pickling
    jobs = [
        (ln, fold, train_idx, test_idx, layers_todo[ln], pos_s)
        for ln in todo
        for fold, (train_idx, test_idx) in enumerate(splits)
    ]
    import os
    from joblib import cpu_count as _jcpu
    effective_jobs = _jcpu() if n_jobs == -1 else n_jobs
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK',
                  os.environ.get('SLURM_JOB_CPUS_PER_NODE', 'not set'))
    print(f"  Total jobs: {len(jobs)}  n_jobs={n_jobs} "
          f"→ effective={effective_jobs} threads  "
          f"(SLURM_CPUS_PER_TASK={slurm_cpus})")

    raw = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_layer_ridge_fold)(ln, fold, train_idx, test_idx, X, p)
        for ln, fold, train_idx, test_idx, X, p in jobs
    )

    # Reassemble per-layer
    from collections import defaultdict
    fold_map = defaultdict(dict)
    for ln, fold, r2 in raw:
        fold_map[ln][fold] = r2

    for ln in todo:
        fold_r2 = [fold_map[ln][f] for f in range(n_folds)]
        mean_r2 = np.mean(fold_r2)
        print(f"  {ln}: mean R²={mean_r2:.4f}  "
              f"(folds: {[f'{r:.3f}' for r in fold_r2]})")
        layer_fold_r2[ln] = fold_r2

    _save_layer_checkpoint(
        checkpoint_path, layer_fold_r2, ordered,
        grid=None, n_samples=len(pos_s),
        n_episodes=len(np.unique(groups_s)), metric='r2')

    return layer_fold_r2, ordered


def decode_layers(layers, pos, groups, width, height, n_iters=500,
                  n_jobs=1, device='cpu', checkpoint_path=None):
    """Run leave-one-episode-out classification for every layer.

    All (layer, fold) pairs are submitted as one flat parallel job pool.
    Supports checkpointing via checkpoint_path.

    Returns dict {layer_name: [fold_loss, ...]}
    """
    if not HAS_TORCH:
        raise RuntimeError("torch required for layer decoding")

    pos_int = pos.astype(int)
    pos_int[:, 0] = np.clip(pos_int[:, 0], 0, width - 1)
    pos_int[:, 1] = np.clip(pos_int[:, 1], 0, height - 1)
    y_cls = np.ravel_multi_index(
        (pos_int[:, 0], pos_int[:, 1]), (width, height))

    logo = LeaveOneGroupOut()
    splits = list(logo.split(pos, y_cls, groups))
    n_folds = len(splits)

    ordered = _ordered_layers(layers)

    # GPU round-robin across all (layer, fold) jobs
    if HAS_TORCH and device.startswith('cuda') and n_jobs != 1:
        import torch as _torch
        n_gpus = _torch.cuda.device_count()
    else:
        n_gpus = 0

    # Resume: load partial results
    layer_fold_losses = {}
    if checkpoint_path and Path(checkpoint_path).exists():
        with open(checkpoint_path, 'rb') as f:
            prev = pickle.load(f)
        layer_fold_losses = prev.get('layer_fold_values', {})
        print(f"  Resumed from checkpoint: {len(layer_fold_losses)} layers done")

    todo = [ln for ln in ordered if ln not in layer_fold_losses]
    print(f"  Layers to process: {len(todo)} / {len(ordered)}")
    if not todo:
        return layer_fold_losses, ordered

    # Flat job list: all (layer, fold) pairs in one pool
    all_jobs = [
        (ln, fold, train_idx, test_idx)
        for ln in todo
        for fold, (train_idx, test_idx) in enumerate(splits)
    ]
    print(f"  Total jobs: {len(all_jobs)}  (n_jobs={n_jobs})")

    def _job_device(job_idx):
        if n_gpus > 0:
            return f'cuda:{job_idx % n_gpus}'
        return device

    raw = Parallel(n_jobs=n_jobs, prefer='processes')(
        delayed(_layer_classification_fold)(
            ln, fold, train_idx, test_idx, layers[ln], y_cls,
            width, height, n_iters, _job_device(i))
        for i, (ln, fold, train_idx, test_idx) in enumerate(all_jobs)
    )

    # Reassemble per-layer
    from collections import defaultdict
    fold_map = defaultdict(dict)
    for ln, fold, loss in raw:
        fold_map[ln][fold] = loss

    for ln in todo:
        fold_losses = [fold_map[ln][f] for f in range(n_folds)]
        mean_loss = np.mean(fold_losses)
        print(f"  {ln}: mean CE={mean_loss:.4f}  "
              f"(folds: {[f'{l:.3f}' for l in fold_losses]})")
        layer_fold_losses[ln] = fold_losses

    _save_layer_checkpoint(
        checkpoint_path, layer_fold_losses, ordered,
        grid=(width, height), n_samples=len(pos),
        n_episodes=len(np.unique(groups)), metric='ce_loss')

    return layer_fold_losses, ordered


def plot_layer_comparison(layer_fold_values, ordered, save_dir, metric='ce_loss'):
    """Horizontal boxplot: one box per layer, ordered early → late.

    Args:
        metric: 'ce_loss' (lower = better) or 'r2' (higher = better).
    """
    display_order = [ln for ln in ordered if ln in layer_fold_values]
    n = len(display_order)
    if n == 0:
        print("  No layer results to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, max(4, n * 0.5)))

    data = [layer_fold_values[ln] for ln in display_order]
    labels = [ln.replace('/', '/\n') for ln in display_order]

    bp = ax.boxplot(data, vert=False, patch_artist=True,
                    labels=labels, widths=0.6)

    section_colors = {
        'enc/cnn': '#0055cc',
        'enc/mlp': '#0099ff',
        'enc/tok': '#44ccff',
        'dyn/sto': '#ff9900',
        'dyn/det': '#ff5500',
        'pol/mlp': '#33aa00',
        'val/mlp': '#996600',
    }
    for patch, ln in zip(bp['boxes'], display_order):
        color = '#888888'
        for prefix, c in section_colors.items():
            if ln.startswith(prefix):
                color = c
                break
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    if metric == 'r2':
        xlabel = 'R² (higher = better spatial decoding)'
    else:
        xlabel = 'Cross-entropy loss (lower = better spatial decoding)'
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title('Per-Layer Position Decoding', fontsize=13)
    ax.grid(True, axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=8)

    for i, ln in enumerate(display_order, start=1):
        mean_v = np.mean(layer_fold_values[ln])
        ax.text(mean_v, i, f' {mean_v:.3f}', va='center', fontsize=7, color='black')

    fig.tight_layout()
    out = save_dir / 'layer_comparison.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Decode agent position from world model states')
    parser.add_argument('--data', required=True, help='Path to trajectory pkl directory')
    parser.add_argument('--save', default=None, help='Output directory for plots/results')
    parser.add_argument('--mode', default='standard', choices=['standard', 'layers'],
                        help='"standard": decode deter/stoch (existing). '
                             '"layers": decode from every model layer and produce '
                             'a comparison boxplot. (default: standard)')
    parser.add_argument('--method', default='both', choices=['classification', 'regression', 'both'])
    parser.add_argument('--n_iters', type=int, default=5000,
                        help='Training iterations for classification decoder')
    parser.add_argument('--repr', default='all',
                        choices=['deter', 'stoch', 'combined', 'all'],
                        help='Which representation to decode (default: all three)')
    parser.add_argument('--per_neuron', action='store_true', default=True,
                        help='Run per-neuron R² analysis (regression only)')
    parser.add_argument('--no_per_neuron', dest='per_neuron', action='store_false')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='Save trained decoder models (for dream_decode.py)')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Parallel workers for CV folds and per-neuron '
                             'analysis. Use -1 for all CPUs. (default: 1)')
    parser.add_argument('--device', default='cpu',
                        help='Torch device for classification decoder '
                             '(cpu, cuda, cuda:0, etc.). With n_jobs>1 and '
                             'device=cuda, folds are round-robin distributed '
                             'across all available GPUs. (default: cpu)')
    parser.add_argument('--ridge_layers', action='store_true', default=False,
                        help='[--mode layers] Use Ridge regression (closed-form, '
                             'no GPU, fast) instead of the PyTorch classifier. '
                             'Metric becomes R² instead of cross-entropy loss. '
                             'Recommended for all runs. (default: False)')
    parser.add_argument('--max_samples', type=int, default=10000,
                        help='[--mode layers --ridge_layers] Subsample to at most '
                             'this many timesteps before fitting (stratified by '
                             'episode). Eliminates the O(N) cost scaling. '
                             'Set to 0 to use all data. (default: 10000)')
    parser.add_argument('--max_dims', type=int, default=256,
                        help='[--mode layers --ridge_layers] Reduce layer features '
                             'to this many dims via truncated PCA before Ridge. '
                             'Eliminates the O(D²) cost scaling. '
                             'Set to 0 to disable. (default: 256)')
    parser.add_argument('--n_cv_folds', type=int, default=5,
                        help='[--mode layers --ridge_layers] Number of CV folds. '
                             'Uses KFold, NOT leave-one-episode-out — LOGO with '
                             'many episodes creates O(N_eps × N_layers) jobs '
                             'which dominates runtime. (default: 5)')
    parser.add_argument('--resume', default=None,
                        help='[--mode layers] Path to a partial '
                             'layer_decode_checkpoint.pkl to resume from. '
                             'Already-completed layers are skipped. '
                             'If omitted, checkpoints are auto-saved to '
                             '<save>/layer_decode_checkpoint.pkl.')
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

    # ---- Layer-wise decoding mode ----
    if args.mode == 'layers':
        print("\n=== Per-Layer Position Decoding ===")
        layers, pos, groups = prepare_data_layers(episodes)
        print(f"  Layers found: {sorted(layers.keys())}")
        print(f"  pos: {pos.shape}  groups: {groups.shape}")
        if metadata and 'area' in metadata:
            width, height = metadata['area']
        else:
            width = int(pos[:, 0].max()) + 1
            height = int(pos[:, 1].max()) + 1
        print(f"  Grid: {width}x{height}")

        checkpoint_path = (Path(args.resume) if args.resume
                           else save_dir / 'layer_decode_checkpoint.pkl')

        if args.ridge_layers:
            print(f"  Using Ridge regression (fast, closed-form). "
                  f"Metric: R² (higher = better).\n"
                  f"  max_samples={args.max_samples}  max_dims={args.max_dims}")
            layer_fold_values, ordered = decode_layers_ridge(
                layers, pos, groups,
                n_jobs=args.n_jobs,
                checkpoint_path=checkpoint_path,
                max_samples=args.max_samples,
                max_dims=args.max_dims,
                n_cv_folds=args.n_cv_folds)
            metric = 'r2'
        else:
            if not HAS_TORCH:
                print("Error: torch required for classifier layer decoding. "
                      "Add --ridge_layers to use Ridge regression instead.")
                raise SystemExit(1)
            n_iters_layers = min(args.n_iters, 500)
            print(f"  Using classification decoder "
                  f"(n_iters={n_iters_layers}, device={args.device}). "
                  f"Metric: CE loss (lower = better).\n"
                  f"  Tip: add --ridge_layers for a ~100x faster scan.")
            layer_fold_values, ordered = decode_layers(
                layers, pos, groups, width, height,
                n_iters=n_iters_layers, n_jobs=args.n_jobs, device=args.device,
                checkpoint_path=checkpoint_path)
            metric = 'ce_loss'

        plot_layer_comparison(layer_fold_values, ordered, save_dir, metric=metric)

        results_file = save_dir / 'layer_decode_results.pkl'
        with open(results_file, 'wb') as f:
            pickle.dump({
                'layer_fold_values': layer_fold_values,
                'ordered': ordered,
                'grid': (width, height),
                'n_samples': len(pos),
                'n_episodes': len(np.unique(groups)),
                'metric': metric,
            }, f)
        print(f"\nResults saved to {results_file}")
        print("Done.")
        raise SystemExit(0)

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

    all_representations = {'deter': deter, 'stoch': stoch, 'combined': combined}
    representations = {args.repr: all_representations[args.repr]} if args.repr != 'all' else all_representations
    print(f"  Representations: {list(representations.keys())}")
    print(f"  Parallelism: n_jobs={args.n_jobs}  device={args.device}")

    # ---- Ridge regression ----
    if args.method in ('regression', 'both'):
        print("\n=== Ridge Regression Decoding ===")
        reg_results = {}
        for name, X in representations.items():
            print(f"\n--- {name} ({X.shape[1]} dims) ---")
            res = ridge_decode_cv(X, pos, groups, n_jobs=args.n_jobs)
            reg_results[name] = res
            print(f"  Overall R²={res['overall_r2']:.4f}  "
                  f"R²_x={res['r2_x']:.4f}  R²_y={res['r2_y']:.4f}  "
                  f"MAE={res['overall_mae']:.3f}")

        plot_regression_summary(reg_results, 'all', save_dir, pos)

        # Per-neuron analysis
        if args.per_neuron:
            r2_deter = r2_stoch = None
            if 'deter' in representations:
                print("\n--- Per-neuron R² (deter) ---")
                r2_deter = per_neuron_r2(deter, pos, groups, n_jobs=args.n_jobs)
                r2_mean_d = r2_deter.mean(axis=1)
                top5_d = np.argsort(r2_mean_d)[::-1][:5]
                for j in top5_d:
                    print(f"  deter[{j}]: R²_x={r2_deter[j,0]:.4f}  "
                          f"R²_y={r2_deter[j,1]:.4f}  mean={r2_mean_d[j]:.4f}")

            if 'stoch' in representations:
                print("\n--- Per-neuron R² (stoch) ---")
                r2_stoch = per_neuron_r2(stoch, pos, groups, n_jobs=args.n_jobs)
                r2_mean_s = r2_stoch.mean(axis=1)
                top5_s = np.argsort(r2_mean_s)[::-1][:5]
                for j in top5_s:
                    print(f"  stoch[{j}]: R²_x={r2_stoch[j,0]:.4f}  "
                          f"R²_y={r2_stoch[j,1]:.4f}  mean={r2_mean_s[j]:.4f}")

            if r2_deter is not None and r2_stoch is not None:
                plot_per_neuron(r2_deter, r2_stoch, save_dir)
            if r2_deter is not None:
                plot_top_neurons(r2_deter, 'deter', pos, deter, groups, save_dir)
            if r2_stoch is not None:
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
                    X, pos, groups, width, height, n_iters=args.n_iters,
                    n_jobs=args.n_jobs, device=args.device)
                print(f"  Mean Manhattan error: {errors.mean():.3f} "
                      f"(shuffle: {shuffle.mean():.3f})")
                plot_classification_summary(errors, shuffle, name, save_dir)
                # Probability heatmap for each episode (deter only to avoid clutter)
                if name == 'deter':
                    for ep_idx in range(len(np.unique(groups))):
                        plot_decoder_probmap(
                            proba, pos, groups, name, save_dir,
                            n_steps=12, episode=ep_idx)
                        # Probability field overlaid on rendered world map
                        plot_probmap_on_world(
                            proba, pos, groups, metadata, name,
                            save_dir, n_steps=12, episode=ep_idx)

    # Save trained decoder models (for use by dream_decode.py)
    if args.save_model:
        print("\n=== Saving Decoder Models ===")
        model_meta_base = {
            'grid': (width, height),
            'n_samples': len(pos),
            'n_episodes': len(np.unique(groups)),
        }

        if args.method in ('regression', 'both'):
            for name, X in representations.items():
                meta = {**model_meta_base, 'repr_name': name,
                        'n_features': X.shape[1], 'type': 'ridge'}
                model = train_full_ridge(X, pos)
                save_decoder_model(model, meta, save_dir / f'ridge_{name}.pkl')

        if args.method in ('classification', 'both') and HAS_TORCH:
            for name, X in representations.items():
                meta = {**model_meta_base, 'repr_name': name,
                        'n_features': X.shape[1], 'n_units': X.shape[1],
                        'n_classes': width * height, 'width': width,
                        'height': height, 'type': 'classifier'}
                clf = LinearClassifier(X.shape[1], width * height)
                print(f"  Training full classifier on {name}...")
                clf.fit(X,
                        np.ravel_multi_index(
                            (pos.astype(int).clip(0, [width-1, height-1])[:, 0],
                             pos.astype(int).clip(0, [width-1, height-1])[:, 1]),
                            (width, height)),
                        n_iters=args.n_iters, verbose=False)
                save_classifier_model(clf, meta, save_dir / f'classifier_{name}.pkl')

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
            if r2_deter is not None:
                save_data['per_neuron_r2_deter'] = r2_deter
            if r2_stoch is not None:
                save_data['per_neuron_r2_stoch'] = r2_stoch
    with open(results_file, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nResults saved to {results_file}")
    print("Done.")
