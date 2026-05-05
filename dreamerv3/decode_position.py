"""
Position decoder for DreamerV3 world model representations.

Trains linear classification decoders to predict agent (x, y) position from
world model latent states (deter, stoch, or both), following the approach in
the pRNN repo (LinearDecoder.py). Uses cross-entropy over grid cells.

Usage:
  python dreamerv3/decode_position.py \
    --data ./logdir/crafter_small_1m/trajectories \
    --save ./logdir/crafter_small_1m/decoder_results

  # Parallel on a GPU cluster (8 CPU workers, classification on CUDA):
  python dreamerv3/decode_position.py \
    --data ./logdir/crafter_small_1m/trajectories \
    --save ./logdir/crafter_small_1m/decoder_results \
    --n_jobs 8 --device cuda

Parallelism:
  --n_jobs N      - Parallel workers. All (layer, fold) pairs are submitted as
                    one flat job pool for maximum parallelism. Use -1 for all
                    CPUs. (default: 1)
  --device DEV    - Torch device for classification decoder (cpu, cuda, cuda:0).
                    With n_jobs>1 and device=cuda, jobs are round-robin
                    distributed across all available GPUs. (default: cpu)

Layer-wise decoding (--mode layers):
  --resume PATH   - Resume from a partial checkpoint. Already-finished layers
                    are skipped. Auto-checkpoint is always saved to
                    <save>/layer_decode_checkpoint.pkl.

  --max_samples N - Subsample to N timesteps before fitting (eliminates O(N)
                    scaling). Default 10000. Set 0 to use all data.
"""

import argparse
import gc
import pickle
from pathlib import Path

from run_info import log_run_info

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import LeaveOneGroupOut

# ---------------------------------------------------------------------------
# Data loading (reuses plot_trajectories.py conventions)
# ---------------------------------------------------------------------------

def load_episodes(data_path, max_episodes=0):
    """Load trajectory episodes from data_path.

    Args:
        data_path: Directory containing episode_*.pkl and/or all_episodes.pkl.
        max_episodes: If > 0, load at most this many episodes using individual
            episode files (avoids loading the full all_episodes.pkl into memory).
            If 0, load all episodes (prefers all_episodes.pkl when available).
    """
    data_path = Path(data_path)
    ep_files = sorted(data_path.glob('episode_*.pkl'))

    # When max_episodes is set and individual files exist, load them directly
    # to avoid the memory cost of deserializing the combined pickle.
    if max_episodes > 0 and ep_files:
        metadata = _load_metadata_only(data_path)
        ep_files = ep_files[:max_episodes]
        episodes = []
        for ep_file in ep_files:
            with open(ep_file, 'rb') as f:
                episodes.append(pickle.load(f))
        return episodes, metadata

    all_file = data_path / 'all_episodes.pkl'
    if all_file.exists():
        with open(all_file, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, dict) and 'episodes' in data:
            metadata = {k: v for k, v in data.items() if k != 'episodes'}
            if max_episodes > 0:
                data['episodes'] = data['episodes'][:max_episodes]
            return data['episodes'], metadata
        if isinstance(data, list) and max_episodes > 0:
            data = data[:max_episodes]
        return data, None
    episodes = []
    for ep_file in ep_files:
        with open(ep_file, 'rb') as f:
            episodes.append(pickle.load(f))
    return episodes, None


def _load_metadata_only(data_path):
    """Try to extract metadata from all_episodes.pkl without loading episodes.

    Falls back to loading the first episode file to check for metadata keys,
    or returns None if nothing is available.
    """
    # Check for a small metadata sidecar first
    meta_file = data_path / 'metadata.pkl'
    if meta_file.exists():
        with open(meta_file, 'rb') as f:
            return pickle.load(f)
    # Peek at all_episodes.pkl using incremental unpickling is not feasible
    # for generic pickles, so just return None — callers use sensible defaults.
    return None


def reload_layer_from_files(episode_indices, layer_name):
    """Reload a single layer's activations from individual episode files.

    Produces the same output as _prepare_single_layer(episodes, layer_name)
    but reads from disk instead of from episode dicts in memory, so only one
    layer's worth of data is resident at a time.

    Args:
        episode_indices: List of (episode_index, episode_file) tuples.
        layer_name: Layer name (e.g. 'enc/cnn0') — reads act/{layer_name}.

    Returns:
        X, pos, groups — same as _prepare_single_layer.
    """
    act_key = f'act/{layer_name}'
    all_x, all_pos, all_groups = [], [], []
    for i, ep_file in episode_indices:
        with open(ep_file, 'rb') as f:
            ep = pickle.load(f)
        arr = ep.get(act_key)
        p = ep.get('player_pos')
        if arr is None or p is None or len(arr) == 0:
            continue
        p = np.array(p, dtype=np.float32)
        a = np.array(arr, dtype=np.float32)
        T = min(len(p), len(a))
        if a.ndim > 2:
            a = a[:T].reshape(T, -1)
        else:
            a = a[:T]
        all_pos.append(p[:T])
        all_x.append(a)
        all_groups.append(np.full(T, i))
        del ep
    if not all_pos:
        raise ValueError(f"No valid episodes for layer {layer_name}")
    return np.concatenate(all_x), np.concatenate(all_pos), np.concatenate(all_groups)


def filter_stuck_episodes(episodes, min_bbox_area):
    """Remove episodes where the agent's bounding box area is below threshold.

    Bounding box = (max_x - min_x) * (max_y - min_y) in tiles.
    """
    kept = []
    for i, ep in enumerate(episodes):
        p = ep.get('player_pos')
        if p is None or len(p) == 0:
            continue
        dx = p[:, 0].max() - p[:, 0].min()
        dy = p[:, 1].max() - p[:, 1].min()
        bbox = dx * dy
        if bbox >= min_bbox_area:
            kept.append(ep)
        else:
            print(f"  Filtering episode {i+1}: bbox area={bbox:.0f} < {min_bbox_area}")
    return kept


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

    def fit(self, X, y_idx, batch_frac=0.75, n_iters=5000, verbose=True,
            patience=500, smooth_window=200, min_delta=1e-4,
            X_val=None, y_cls_val=None, pos_int_val=None,
            width=None, height=None, manhattan_patience=20,
            manhattan_check_every=100):
        """Train on (X, y_idx) where y_idx are integer class labels.
        Records self.loss_history (list of floats, one per iteration).

        When val data is provided (X_val, pos_int_val, width, height):
          Manhattan-based early stopping — every manhattan_check_every iters,
          compute Manhattan distance on validation set. Stop when no
          improvement for manhattan_patience checks.

        When val data is NOT provided:
          Run for exactly n_iters (no early stopping)."""
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y_idx, dtype=torch.long, device=self.device)
        N = X_t.shape[0]
        batch_size = max(1, int(batch_frac * N))
        self.loss_history = []
        self.model.train()

        has_val = (X_val is not None and pos_int_val is not None
                   and width is not None and height is not None)
        best_manhattan = float('inf')
        wait = 0

        for step in range(n_iters):
            idx = torch.randint(N, (batch_size,), device=self.device)
            logits = self.model(X_t[idx])
            loss = self.loss_fn(logits, y_t[idx])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.loss_history.append(loss.item())
            if verbose and (step % 1000 == 0 or step == n_iters - 1):
                print(f"  [{step:>5d}/{n_iters}] loss={loss.item():.4f}")
            # Manhattan-based early stopping on validation set
            if has_val and manhattan_patience > 0 and step > 0 and step % manhattan_check_every == 0:
                pred_cls = self.predict(X_val)
                pred_xy = np.stack(np.unravel_index(pred_cls, (width, height)), axis=1)
                val_manhattan = np.sum(np.abs(pred_xy - pos_int_val), axis=1).mean()
                self.model.train()  # switch back to train mode after predict
                if val_manhattan < best_manhattan - 1e-3:
                    best_manhattan = val_manhattan
                    wait = 0
                else:
                    wait += 1
                    if wait >= manhattan_patience:
                        if verbose:
                            print(f"  Early stop at step {step}, "
                                  f"val_manhattan={val_manhattan:.3f}")
                        break

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
                         width, height, n_iters, device, patience=500):
    """Run a single classification fold. Returns (fold, test_idx, err, shuf,
    pred_xy, proba) — designed to be called in parallel."""
    print(f"  Classification fold {fold+1} "
          f"(train={len(train_idx)}, test={len(test_idx)}, device={device})")
    clf = LinearClassifier(X.shape[1], width * height, device=device)
    clf.fit(X[train_idx], y_cls[train_idx], n_iters=n_iters, verbose=False,
            patience=patience)
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
                          n_jobs=1, device='cpu', patience=500):
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
            width, height, n_iters, devices[fold], patience=patience)
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
# Layer decoder save / load / eval
# ---------------------------------------------------------------------------

def save_layer_decoders(layers, pos, width, height, ordered, save_dir,
                        n_iters=50000, device='cpu', n_jobs=1, patience=500,
                        ep_file_index=None, train_mask=None):
    """Retrain one classifier decoder per layer on ALL data and save to disk.

    Saves:
      <save_dir>/layer_decoders/<layer_safe_name>.pkl  (one per layer)
      <save_dir>/layer_decoders/manifest.pkl           (metadata)

    Args:
        ep_file_index: If provided, reload layers one-at-a-time from files
            to avoid OOM. ``layers`` may be empty/None in this case but
            must still provide layer_sizes via a prior pass.
        train_mask: Boolean mask to select training samples when reloading
            from ep_file_index (needed when holdout split was applied).
    """
    import gc as _gc
    out_dir = Path(save_dir) / 'layer_decoders'
    out_dir.mkdir(parents=True, exist_ok=True)

    pos_int = pos.astype(int)
    pos_int[:, 0] = np.clip(pos_int[:, 0], 0, width - 1)
    pos_int[:, 1] = np.clip(pos_int[:, 1], 0, height - 1)
    y_cls = np.ravel_multi_index(
        (pos_int[:, 0], pos_int[:, 1]), (width, height))

    # Record layer sizes before we potentially discard layers dict
    layer_sizes = {}
    if layers:
        layer_sizes = {ln: layers[ln].shape[1] for ln in ordered
                       if ln in layers}

    def _get_layer(ln):
        """Get layer data, reloading from files if needed."""
        if layers and ln in layers:
            return layers[ln]
        if ep_file_index is not None:
            X, _, _ = reload_layer_from_files(ep_file_index, ln)
            if train_mask is not None:
                X = X[train_mask]
            return X
        raise KeyError(f"Layer {ln} not in layers dict and no ep_file_index")

    layer_files = {}

    if not HAS_TORCH:
        raise RuntimeError("torch required to save classifier layer decoders")
    for i, ln in enumerate(ordered):
        safe = ln.replace('/', '_')
        X = _get_layer(ln)
        if ln not in layer_sizes:
            layer_sizes[ln] = X.shape[1]
        clf = LinearClassifier(X.shape[1], width * height, device=device)
        print(f"  [{i+1}/{len(ordered)}] Training full classifier on "
              f"{ln} ({X.shape[1]} dims)...")
        clf.fit(X, y_cls, n_iters=n_iters, verbose=False, patience=patience)
        fname = f'{safe}.pkl'
        path = out_dir / fname
        meta = {
            'layer_name': ln, 'n_units': X.shape[1],
            'n_classes': width * height, 'width': width, 'height': height,
            'type': 'classifier', 'grid': (width, height),
        }
        save_classifier_model(clf, meta, path)
        layer_files[ln] = fname
        print(f"  Saved classifier decoder: {ln} → {path}")
        del X, clf
        _gc.collect()

    manifest = {
        'ordered': ordered,
        'grid': (width, height),
        'metric': 'manhattan',
        'decoder_type': 'classifier',
        'layer_files': layer_files,
        'layer_sizes': layer_sizes,
        'n_train_samples': len(pos),
    }
    manifest_path = out_dir / 'manifest.pkl'
    with open(manifest_path, 'wb') as f:
        pickle.dump(manifest, f)
    print(f"  Saved manifest to {manifest_path}")


def eval_layer_decoders(from_model_dir, layers, pos, width, height):
    """Load saved layer decoders and evaluate on new data.

    Args:
        from_model_dir: Path to layer_decoders/ directory containing manifest.pkl
        layers: dict {layer_name: np.ndarray (N, D)} from new trajectories
        pos: np.ndarray (N, 2) ground-truth positions
        width, height: grid dimensions

    Returns:
        layer_values: dict {layer_name: [metric_value]}
        ordered: list of layer names
        metric: 'manhattan'
    """
    from_dir = Path(from_model_dir)
    manifest_path = from_dir / 'manifest.pkl'
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.pkl in {from_dir}")
    with open(manifest_path, 'rb') as f:
        manifest = pickle.load(f)

    ordered = manifest['ordered']
    metric = manifest['metric']
    layer_files = manifest['layer_files']
    train_grid = manifest['grid']

    if train_grid != (width, height):
        print(f"  WARNING: grid mismatch — trained on {train_grid}, "
              f"evaluating on ({width}, {height})")

    # Only eval layers present in both saved models and new data
    available = [ln for ln in ordered if ln in layers and ln in layer_files]
    skipped = [ln for ln in ordered if ln not in layers]
    if skipped:
        print(f"  Skipping {len(skipped)} layers not in new data: {skipped}")

    pos_int = pos.astype(int)
    pos_int[:, 0] = np.clip(pos_int[:, 0], 0, width - 1)
    pos_int[:, 1] = np.clip(pos_int[:, 1], 0, height - 1)

    layer_values = {}
    print(f"  Evaluating {len(available)} layers (classifier)...")

    for ln in available:
        fpath = from_dir / layer_files[ln]
        X = layers[ln]
        clf, meta = load_classifier_model(fpath)
        clf.model.eval()
        pred_cls = clf.predict(X)
        w, h = meta['width'], meta['height']
        pred_xy = np.stack(np.unravel_index(pred_cls, (w, h)), axis=1)
        manhattan = np.sum(np.abs(pred_xy - pos_int), axis=1).mean()
        layer_values[ln] = [float(manhattan)]
        print(f"  {ln}: decode error={manhattan:.3f} tiles")

    return layer_values, available, metric


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

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
    fig.savefig(save_dir / f'classification_{layer_name}.svg', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved classification_{layer_name}.svg")


def _compute_tile_stats(pos, pred, width, height):
    """Compute per-tile occupancy, mean error, and per-sample arrays."""
    pos_int = np.clip(pos.astype(int), 0, [width - 1, height - 1])
    pred_int = np.clip(np.round(pred).astype(int), 0, [width - 1, height - 1])
    sample_err = np.sum(np.abs(pred_int - pos_int), axis=1).astype(float)

    tile_visits = np.zeros((width, height), dtype=float)
    for x, y in pos_int:
        tile_visits[x, y] += 1

    per_sample_occ = tile_visits[pos_int[:, 0], pos_int[:, 1]]

    tile_error_sum = np.zeros((width, height), dtype=float)
    tile_error_count = np.zeros((width, height), dtype=float)
    for i in range(len(pos_int)):
        x, y = pos_int[i]
        tile_error_sum[x, y] += sample_err[i]
        tile_error_count[x, y] += 1
    tile_mean_err = np.full((width, height), np.nan)
    visited = tile_error_count > 0
    tile_mean_err[visited] = tile_error_sum[visited] / tile_error_count[visited]

    return {
        'tile_visits': tile_visits,
        'tile_mean_err': tile_mean_err,
        'sample_err': sample_err,
        'per_sample_occ': per_sample_occ,
    }


def _get_world_img(metadata, width, height):
    """Load and flip the Crafter world map image. Returns (img, extent) or (None, extent)."""
    world_img = None
    tile_size = 8
    try:
        from plot_trajectories import _render_crafter_world
        world_img, _, tile_size = _render_crafter_world(metadata, tile_size)
    except Exception:
        try:
            import sys, os
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from plot_trajectories import _render_crafter_world
            world_img, _, tile_size = _render_crafter_world(metadata, tile_size)
        except Exception:
            pass
    extent = [-0.5, width - 0.5, -0.5, height - 0.5]
    img_lower = world_img[::-1] if world_img is not None else None
    return img_lower, extent


def _plot_occupancy_row(axes, stats, width, height, world_img, world_extent,
                        repr_name, method, row_label):
    """Draw one row of the occupancy-vs-error figure (3 panels)."""
    ax_heat, ax_err_map, ax_scatter = axes
    tile_visits = stats['tile_visits']
    tile_mean_err = stats['tile_mean_err']
    sample_err = stats['sample_err']
    per_sample_occ = stats['per_sample_occ']

    # Panel A: occupancy
    if world_img is not None:
        ax_heat.imshow(world_img, alpha=0.25, extent=world_extent,
                       origin='lower', aspect='equal')
    occ_plot = tile_visits.copy()
    occ_plot[occ_plot == 0] = np.nan
    im = ax_heat.imshow(occ_plot.T, origin='lower', cmap='hot',
                        aspect='equal', interpolation='nearest',
                        extent=world_extent, alpha=0.8)
    cbar = plt.colorbar(im, ax=ax_heat, shrink=0.8)
    cbar.set_label('Visit count')
    ax_heat.set_xlabel('X')
    ax_heat.set_ylabel('Y')
    ax_heat.set_title(f'Tile occupancy ({row_label})')
    ax_heat.set_xlim(world_extent[0], world_extent[1])
    ax_heat.set_ylim(world_extent[2], world_extent[3])

    # Panel B: mean error per tile
    if world_img is not None:
        ax_err_map.imshow(world_img, alpha=0.25, extent=world_extent,
                          origin='lower', aspect='equal')
    im2 = ax_err_map.imshow(tile_mean_err.T, origin='lower', cmap='RdYlGn_r',
                            aspect='equal', interpolation='nearest',
                            extent=world_extent, alpha=0.8)
    cbar2 = plt.colorbar(im2, ax=ax_err_map, shrink=0.8)
    cbar2.set_label('Mean Manhattan error (tiles)')
    ax_err_map.set_xlabel('X')
    ax_err_map.set_ylabel('Y')
    ax_err_map.set_title(f'Mean decode error per tile ({row_label})')
    ax_err_map.set_xlim(world_extent[0], world_extent[1])
    ax_err_map.set_ylim(world_extent[2], world_extent[3])

    # Panel C: scatter
    ax_scatter.scatter(per_sample_occ, sample_err, s=4, alpha=0.15,
                       edgecolors='none', rasterized=True, color='grey')
    occ_vals = np.sort(np.unique(per_sample_occ))
    mean_err_per_occ = np.array([sample_err[per_sample_occ == v].mean()
                                 for v in occ_vals])
    sem_err_per_occ = np.array([sample_err[per_sample_occ == v].std()
                                / np.sqrt((per_sample_occ == v).sum())
                                for v in occ_vals])
    ax_scatter.errorbar(occ_vals, mean_err_per_occ, yerr=sem_err_per_occ,
                        fmt='o-', markersize=4, linewidth=1.5, capsize=2,
                        color='#2196F3', label='mean ± SEM', zorder=5)
    # Median + IQR
    median_per_occ = np.array([np.median(sample_err[per_sample_occ == v])
                               for v in occ_vals])
    q25_per_occ = np.array([np.percentile(sample_err[per_sample_occ == v], 25)
                            for v in occ_vals])
    q75_per_occ = np.array([np.percentile(sample_err[per_sample_occ == v], 75)
                            for v in occ_vals])
    ax_scatter.fill_between(occ_vals, q25_per_occ, q75_per_occ,
                            alpha=0.2, color='#FF9800', zorder=4)
    ax_scatter.plot(occ_vals, median_per_occ, 's--', markersize=3,
                    linewidth=1.2, color='#FF9800', label='median ± IQR',
                    zorder=6)
    ax_scatter.legend(fontsize=8)
    ax_scatter.set_xlabel('Tile occupancy (visit count)')
    ax_scatter.set_ylabel('Manhattan error (tiles)')
    ax_scatter.set_title(f'Decode error vs occupancy ({row_label}, {repr_name}, {method})')


def _plot_diff_row(axes, test_stats, train_stats, width, height,
                   world_img, world_extent, repr_name, method):
    """Draw the train-minus-test difference row (3 panels)."""
    ax_occ, ax_err, ax_scatter = axes

    # -- Panel A: occupancy difference (% of each set's total) --
    test_occ = test_stats['tile_visits']
    train_occ = train_stats['tile_visits']
    # Normalize to percentage of total visits
    test_pct = test_occ / max(test_occ.sum(), 1) * 100
    train_pct = train_occ / max(train_occ.sum(), 1) * 100
    occ_diff = train_pct - test_pct
    # Mask tiles never visited by either set
    either_visited = (test_occ > 0) | (train_occ > 0)
    occ_diff_plot = np.full((width, height), np.nan)
    occ_diff_plot[either_visited] = occ_diff[either_visited]

    if world_img is not None:
        ax_occ.imshow(world_img, alpha=0.25, extent=world_extent,
                      origin='lower', aspect='equal')
    vmax = max(abs(np.nanmin(occ_diff_plot)), abs(np.nanmax(occ_diff_plot)), 0.01)
    im = ax_occ.imshow(occ_diff_plot.T, origin='lower', cmap='RdBu_r',
                       aspect='equal', interpolation='nearest',
                       extent=world_extent, alpha=0.8,
                       vmin=-vmax, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax_occ, shrink=0.8)
    cbar.set_label('Occupancy % (train - test)')
    ax_occ.set_xlabel('X')
    ax_occ.set_ylabel('Y')
    ax_occ.set_title('Occupancy difference (train - test)')
    ax_occ.set_xlim(world_extent[0], world_extent[1])
    ax_occ.set_ylim(world_extent[2], world_extent[3])

    # -- Panel B: mean error difference per tile --
    test_err = test_stats['tile_mean_err']
    train_err = train_stats['tile_mean_err']
    both_visited = np.isfinite(test_err) & np.isfinite(train_err)
    err_diff = np.full((width, height), np.nan)
    err_diff[both_visited] = train_err[both_visited] - test_err[both_visited]

    if world_img is not None:
        ax_err.imshow(world_img, alpha=0.25, extent=world_extent,
                      origin='lower', aspect='equal')
    vmax_e = max(abs(np.nanmin(err_diff)), abs(np.nanmax(err_diff)), 0.01)
    im2 = ax_err.imshow(err_diff.T, origin='lower', cmap='RdBu_r',
                        aspect='equal', interpolation='nearest',
                        extent=world_extent, alpha=0.8,
                        vmin=-vmax_e, vmax=vmax_e)
    cbar2 = plt.colorbar(im2, ax=ax_err, shrink=0.8)
    cbar2.set_label('Mean error diff (train - test)')
    ax_err.set_xlabel('X')
    ax_err.set_ylabel('Y')
    ax_err.set_title('Mean decode error difference (train - test)')
    ax_err.set_xlim(world_extent[0], world_extent[1])
    ax_err.set_ylim(world_extent[2], world_extent[3])

    # -- Panel C: test error vs normalized occupancy difference per tile --
    # x = train% - test% for each tile; y = test decode error at that tile
    # Negative x = tile overrepresented in test; positive = in train
    test_occ = test_stats['tile_visits']
    train_occ = train_stats['tile_visits']
    test_pct = test_occ / max(test_occ.sum(), 1) * 100
    train_pct = train_occ / max(train_occ.sum(), 1) * 100
    occ_diff = train_pct - test_pct  # (width, height)

    test_err_tile = test_stats['tile_mean_err']
    test_visited = np.isfinite(test_err_tile)
    tile_x = occ_diff[test_visited]
    tile_y = test_err_tile[test_visited]

    ax_scatter.scatter(tile_x, tile_y, s=12, alpha=0.5, edgecolors='none',
                       rasterized=True, color='grey', zorder=3)
    ax_scatter.axvline(0, color='k', linewidth=0.5, linestyle=':', alpha=0.5)

    # Binned mean ± SEM and median ± IQR
    n_bins = min(15, max(3, len(tile_x) // 5))
    bin_edges = np.linspace(tile_x.min(), tile_x.max(), n_bins + 1)
    bin_centers, bin_mean, bin_sem, bin_med, bin_q25, bin_q75 = [], [], [], [], [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (tile_x >= lo) & (tile_x < hi)
        else:
            mask = (tile_x >= lo) & (tile_x <= hi)
        if mask.sum() < 2:
            continue
        vals = tile_y[mask]
        bin_centers.append((lo + hi) / 2)
        bin_mean.append(vals.mean())
        bin_sem.append(vals.std() / np.sqrt(len(vals)))
        bin_med.append(np.median(vals))
        bin_q25.append(np.percentile(vals, 25))
        bin_q75.append(np.percentile(vals, 75))
    bin_centers = np.array(bin_centers)
    bin_mean = np.array(bin_mean)
    bin_sem = np.array(bin_sem)
    bin_med = np.array(bin_med)
    bin_q25 = np.array(bin_q25)
    bin_q75 = np.array(bin_q75)

    if len(bin_centers) > 0:
        ax_scatter.errorbar(bin_centers, bin_mean, yerr=bin_sem,
                            fmt='o-', markersize=4, linewidth=1.5, capsize=2,
                            color='#2196F3', label='mean ± SEM', zorder=5)
        ax_scatter.fill_between(bin_centers, bin_q25, bin_q75,
                                alpha=0.2, color='#FF9800', zorder=4)
        ax_scatter.plot(bin_centers, bin_med, 's--', markersize=3,
                        linewidth=1.2, color='#FF9800',
                        label='median ± IQR', zorder=6)
    ax_scatter.legend(fontsize=7)
    ax_scatter.set_xlabel('Occupancy difference % (train - test)')
    ax_scatter.set_ylabel('Test Manhattan error (tiles)')
    ax_scatter.set_title(f'Test error vs occupancy bias ({repr_name}, {method})')


def plot_occupancy_vs_error(pos, pred, width, height, save_dir,
                            repr_name='deter', method='classification',
                            metadata=None,
                            train_pos=None, train_pred=None):
    """Occupancy-vs-error figure.

    Row 1: held-out / test data.
    Row 2 (optional): training data.
    Row 3 (optional): train-minus-test difference (occupancy %, error, overlay).
    """
    has_train = train_pos is not None and train_pred is not None
    nrows = 3 if has_train else 1

    fig, axes = plt.subplots(
        nrows, 3, figsize=(18, 5.5 * nrows),
        gridspec_kw={'width_ratios': [1, 1, 1.15]})
    if nrows == 1:
        axes = axes[np.newaxis, :]  # ensure 2D

    world_img, world_extent = _get_world_img(metadata, width, height)

    test_stats = _compute_tile_stats(pos, pred, width, height)
    _plot_occupancy_row(axes[0], test_stats, width, height,
                        world_img, world_extent, repr_name, method, 'test')

    if has_train:
        train_stats = _compute_tile_stats(train_pos, train_pred, width, height)
        _plot_occupancy_row(axes[1], train_stats, width, height,
                            world_img, world_extent, repr_name, method, 'train')
        _plot_diff_row(axes[2], test_stats, train_stats, width, height,
                       world_img, world_extent, repr_name, method)

    fig.tight_layout()
    fname = f'occupancy_vs_error_{repr_name}.svg'
    fig.savefig(save_dir / fname, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")


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
    fig.savefig(save_dir / f'probmap_{layer_name}_ep{episode+1}.svg', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved probmap_{layer_name}_ep{episode+1}.svg")


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
    out = save_dir / f'probmap_world_{layer_name}_ep{episode + 1}.svg'
    fig.savefig(out, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close(fig)
    print(f"  Saved {out.name}")


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


def _prepare_single_layer(episodes, layer_name):
    """Extract (X, pos, groups) for one layer only — much lower peak memory
    than prepare_data_layers which loads all layers simultaneously.

    Returns:
        X:      np.ndarray (N, D)
        pos:    np.ndarray (N, 2)
        groups: np.ndarray (N,)
    """
    all_x, all_pos, all_groups = [], [], []
    act_key = f'act/{layer_name}'
    for i, ep in enumerate(episodes):
        if 'player_pos' not in ep:
            continue
        arr = ep.get(act_key)
        if arr is None or len(arr) == 0:
            continue
        p = np.array(ep['player_pos'], dtype=np.float32)
        a = np.array(arr, dtype=np.float32)
        T = min(len(p), len(a))
        if a.ndim > 2:
            a = a[:T].reshape(T, -1)
        else:
            a = a[:T]
        all_pos.append(p[:T])
        all_x.append(a)
        all_groups.append(np.full(T, i))
    if not all_pos:
        raise ValueError(f"No valid episodes for layer {layer_name}")
    return np.concatenate(all_x), np.concatenate(all_pos), np.concatenate(all_groups)


def _layer_classification_fold(layer_name, fold, train_idx, test_idx, X,
                                y_cls, pos_int, width, height, n_iters, device,
                                patience=500):
    """Run one fold for a single layer; return (layer_name, fold, per-timestep Manhattan array, loss_history)."""
    # Split training data into 90% train / 10% val for Manhattan early stopping
    rng = np.random.RandomState(42 + fold)
    n_train = len(train_idx)
    n_val = max(1, int(0.1 * n_train))
    perm = rng.permutation(n_train)
    val_sub = train_idx[perm[:n_val]]
    train_sub = train_idx[perm[n_val:]]

    clf = LinearClassifier(X.shape[1], width * height, device=device)
    clf.fit(X[train_sub], y_cls[train_sub], n_iters=n_iters, verbose=False,
            X_val=X[val_sub], y_cls_val=y_cls[val_sub],
            pos_int_val=pos_int[val_sub], width=width, height=height,
            manhattan_patience=patience)
    pred_cls = clf.predict(X[test_idx])
    pred_xy = np.stack(np.unravel_index(pred_cls, (width, height)), axis=1)
    manhattan_per_timestep = np.sum(np.abs(pred_xy - pos_int[test_idx]), axis=1).astype(np.float32)
    return layer_name, fold, manhattan_per_timestep, clf.loss_history


def _layer_classification_holdout(layer_name, X_train, y_cls_train,
                                   pos_int_train, X_test, pos_int_test,
                                   width, height, n_iters, device,
                                   patience=500):
    """Train on train data with internal val split, eval on held-out test. No CV."""
    # Split training data into 90% train / 10% val for Manhattan early stopping
    rng = np.random.RandomState(42)
    n = len(X_train)
    n_val = max(1, int(0.1 * n))
    perm = rng.permutation(n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    clf = LinearClassifier(X_train.shape[1], width * height, device=device)
    clf.fit(X_train[train_idx], y_cls_train[train_idx], n_iters=n_iters,
            verbose=False,
            X_val=X_train[val_idx], y_cls_val=y_cls_train[val_idx],
            pos_int_val=pos_int_train[val_idx], width=width, height=height,
            manhattan_patience=patience)
    pred_cls = clf.predict(X_test)
    pred_xy = np.stack(np.unravel_index(pred_cls, (width, height)), axis=1)
    manhattan_per_timestep = np.sum(np.abs(pred_xy - pos_int_test), axis=1).astype(np.float32)
    # Train predictions for occupancy plot
    train_pred_cls = clf.predict(X_train)
    train_pred_xy = np.stack(np.unravel_index(train_pred_cls, (width, height)), axis=1)
    # Also get probability maps for visualization
    proba_flat = clf.predict_proba(X_test)  # (N_test, width*height)
    proba_maps = proba_flat.reshape(-1, width, height)
    return layer_name, manhattan_per_timestep, clf.loss_history, pred_xy, proba_maps, train_pred_xy, clf


def decode_layers_classification_holdout(
        layers_train, pos_train, layers_test, pos_test,
        width, height, n_iters=50000, n_jobs=1, device='cpu', patience=500,
        max_samples=0, checkpoint_path=None):
    """Train classifier on train set, evaluate on held-out test set. No CV.

    Args:
        max_samples: Subsample training timesteps to at most this many
            (stratified by episode via _subsample_layers). 0 = use all.
        checkpoint_path: Auto-save partial results here after each batch
            of layers. On restart, already-finished layers are skipped.

    Returns (layer_values, ordered, loss_histories) where
    layer_values = {layer: np.ndarray} (per-timestep Manhattan errors).
    """
    if not HAS_TORCH:
        raise RuntimeError("torch required for classification holdout")

    ordered = _ordered_layers(layers_train)

    # Resume: load partial results
    layer_values = {}
    loss_histories = {}
    if checkpoint_path and Path(checkpoint_path).exists():
        with open(checkpoint_path, 'rb') as f:
            prev = pickle.load(f)
        saved_metric = prev.get('metric', 'manhattan')
        if saved_metric != 'manhattan':
            print(f"  WARNING: checkpoint has metric='{saved_metric}', "
                  f"expected 'manhattan'. Ignoring checkpoint.")
        else:
            layer_values = prev.get('layer_fold_values', {})
            print(f"  Resumed from checkpoint: {len(layer_values)} layers done")

    todo = [ln for ln in ordered if ln not in layer_values]
    print(f"  Layers to process: {len(todo)} / {len(ordered)}")
    if not todo:
        return layer_values, ordered, loss_histories, {}, {}, {}

    # Subsample training data if requested
    if max_samples > 0 and len(pos_train) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(pos_train), max_samples, replace=False)
        idx.sort()
        print(f"  Subsampled train: {len(pos_train)} → {len(idx)} timesteps")
        layers_train = {ln: arr[idx] for ln, arr in layers_train.items()}
        pos_train = pos_train[idx]

    pos_int_train = pos_train.astype(int)
    pos_int_train[:, 0] = np.clip(pos_int_train[:, 0], 0, width - 1)
    pos_int_train[:, 1] = np.clip(pos_int_train[:, 1], 0, height - 1)
    y_cls_train = np.ravel_multi_index(
        (pos_int_train[:, 0], pos_int_train[:, 1]), (width, height))

    pos_int_test = pos_test.astype(int)
    pos_int_test[:, 0] = np.clip(pos_int_test[:, 0], 0, width - 1)
    pos_int_test[:, 1] = np.clip(pos_int_test[:, 1], 0, height - 1)

    if HAS_TORCH and device.startswith('cuda') and n_jobs != 1:
        import torch as _torch
        n_gpus = _torch.cuda.device_count()
    else:
        n_gpus = 0

    def _job_device(job_idx):
        if n_gpus > 0:
            return f'cuda:{job_idx % n_gpus}'
        return device

    print(f"  Layers: {len(ordered)}, train={len(pos_train)}, "
          f"test={len(pos_test)} samples")

    # Run layers sequentially — each is fast on GPU, avoids CUDA subprocess
    # spawn overhead that dominates with Parallel(prefer='processes').
    layer_pred_xy = {}
    layer_train_pred_xy = {}
    layer_proba = {}
    layer_clfs = {}
    for i, ln in enumerate(todo):
        print(f"  [{i+1}/{len(todo)}] Training {ln}...")
        ln_out, manhattan_arr, loss_hist, pred_xy, proba_maps, train_pred_xy, clf = \
            _layer_classification_holdout(
                ln, layers_train[ln], y_cls_train, pos_int_train,
                layers_test[ln], pos_int_test, width, height,
                n_iters, _job_device(i), patience=patience)
        layer_values[ln] = manhattan_arr
        loss_histories[ln] = [loss_hist]
        layer_pred_xy[ln] = pred_xy
        layer_train_pred_xy[ln] = train_pred_xy
        layer_proba[ln] = proba_maps
        layer_clfs[ln] = clf
        print(f"  {ln}: decode error={np.mean(manhattan_arr):.3f} tiles "
              f"({len(loss_hist)} iters)")

        # Checkpoint after each layer
        _save_layer_checkpoint(
            checkpoint_path, layer_values, ordered,
            grid=(width, height), n_samples=len(pos_train),
            n_episodes=0, metric='manhattan')

    return layer_values, ordered, loss_histories, layer_pred_xy, layer_proba, layer_train_pred_xy, layer_clfs


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


def decode_layers(layers, pos, groups, width, height, n_iters=500,
                  n_jobs=1, device='cpu', checkpoint_path=None, patience=500,
                  use_kfold=False, n_cv_folds=5, max_samples=0):
    """Run classification CV for every layer.

    By default uses LOGO (leave-one-episode-out). Pass use_kfold=True to
    use KFold instead (much faster with many episodes).

    All (layer, fold) pairs are submitted as one flat parallel job pool.
    Supports checkpointing via checkpoint_path.

    Returns dict {layer_name: [fold_loss, ...]}
    """
    if not HAS_TORCH:
        raise RuntimeError("torch required for layer decoding")

    # Subsample training data if requested
    if max_samples > 0 and len(pos) > max_samples:
        layers, pos, groups = _subsample_layers(
            layers, pos, groups, max_samples)

    pos_int = pos.astype(int)
    pos_int[:, 0] = np.clip(pos_int[:, 0], 0, width - 1)
    pos_int[:, 1] = np.clip(pos_int[:, 1], 0, height - 1)
    y_cls = np.ravel_multi_index(
        (pos_int[:, 0], pos_int[:, 1]), (width, height))

    if use_kfold:
        from sklearn.model_selection import KFold
        n_episodes = len(np.unique(groups))
        n_folds = min(n_cv_folds, n_episodes)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = list(kf.split(pos))
        print(f"  CV: {n_folds}-fold KFold  (n_episodes={n_episodes})")
    else:
        logo = LeaveOneGroupOut()
        splits = list(logo.split(pos, y_cls, groups))
        n_folds = len(splits)
        print(f"  CV: LOGO  ({n_folds} folds = {n_folds} episodes)")

    ordered = _ordered_layers(layers)

    # GPU round-robin across all (layer, fold) jobs
    if HAS_TORCH and device.startswith('cuda') and n_jobs != 1:
        import torch as _torch
        n_gpus = _torch.cuda.device_count()
    else:
        n_gpus = 0

    # Resume: load partial results
    layer_fold_manhattan = {}
    if checkpoint_path and Path(checkpoint_path).exists():
        with open(checkpoint_path, 'rb') as f:
            prev = pickle.load(f)
        saved_metric = prev.get('metric', 'manhattan')
        if saved_metric != 'manhattan':
            print(f"  WARNING: checkpoint has metric='{saved_metric}', expected 'manhattan'. "
                  f"Ignoring checkpoint to avoid mixing metrics.")
        else:
            layer_fold_manhattan = prev.get('layer_fold_values', {})
            print(f"  Resumed from checkpoint: {len(layer_fold_manhattan)} layers done")

    todo = [ln for ln in ordered if ln not in layer_fold_manhattan]
    print(f"  Layers to process: {len(todo)} / {len(ordered)}")
    if not todo:
        return layer_fold_manhattan, ordered, {}

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
            ln, fold, train_idx, test_idx, layers[ln], y_cls, pos_int,
            width, height, n_iters, _job_device(i), patience=patience)
        for i, (ln, fold, train_idx, test_idx) in enumerate(all_jobs)
    )

    # Reassemble per-layer: concatenate per-timestep arrays across folds
    from collections import defaultdict
    fold_map = defaultdict(dict)
    loss_map = defaultdict(dict)
    for ln, fold, manhattan_arr, loss_hist in raw:
        fold_map[ln][fold] = manhattan_arr
        loss_map[ln][fold] = loss_hist

    for ln in todo:
        fold_arrays = [fold_map[ln][f] for f in range(n_folds)]
        all_manhattan = np.concatenate(fold_arrays)
        mean_manhattan = np.mean(all_manhattan)
        per_fold_means = [f'{np.mean(a):.3f}' for a in fold_arrays]
        print(f"  {ln}: mean decode error={mean_manhattan:.3f} tiles  "
              f"(folds: {per_fold_means})")
        layer_fold_manhattan[ln] = all_manhattan

    # Collect loss histories: {layer: [[fold0_losses], [fold1_losses], ...]}
    layer_loss_histories = {}
    for ln in todo:
        layer_loss_histories[ln] = [loss_map[ln][f] for f in range(n_folds)]

    _save_layer_checkpoint(
        checkpoint_path, layer_fold_manhattan, ordered,
        grid=(width, height), n_samples=len(pos),
        n_episodes=len(np.unique(groups)), metric='manhattan')

    return layer_fold_manhattan, ordered, layer_loss_histories


def plot_layer_probmap_on_world(layer_proba, pos_test, groups_test, metadata,
                                layer_name, save_dir, n_frames=12,
                                step_stride=16, episode=None, tile_size=8):
    """3x4 grid: decoded P(pos) heatmap on the real Crafter world for one layer.

    Picks ``n_frames`` timesteps spaced ``step_stride`` apart from one test
    episode. Each panel shows the world map background, semi-transparent
    probability heatmap, trajectory up to that timestep, and true / argmax
    decoded markers.

    Args:
        layer_proba: (N_test, width, height) probability maps for this layer.
        pos_test:    (N_test, 2) true positions.
        groups_test: (N_test,) episode IDs.
        metadata:    dict with 'area' and env info (for world rendering).
        layer_name:  e.g. 'dyn/deter'.
        save_dir:    Path to save directory.
        n_frames:    number of panels (default 12 → 3×4).
        step_stride: timestep spacing between panels (default 16).
        episode:     episode index to plot (None → first episode with enough steps).
        tile_size:   pixels per tile in the rendered world image.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from plot_trajectories import _render_crafter_world

    world_img, env_seed, tile_size = _render_crafter_world(metadata, tile_size)
    if world_img is None:
        print("  Could not render world for layer probmap overlay — skipping")
        return
    img_h, img_w = world_img.shape[:2]

    # Pick episode
    unique_eps = np.unique(groups_test)
    if episode is None:
        for ep in unique_eps:
            if (groups_test == ep).sum() >= n_frames * step_stride:
                episode = ep
                break
        if episode is None:
            episode = unique_eps[0]

    ep_mask = groups_test == episode
    ep_proba = layer_proba[ep_mask]
    ep_pos = pos_test[ep_mask]
    T = len(ep_proba)
    if T == 0:
        print(f"  No test data for episode {episode}, skipping layer probmap")
        return

    # Timestep indices: n_frames steps spaced step_stride apart
    step_indices = np.arange(0, min(T, n_frames * step_stride), step_stride)
    if len(step_indices) > n_frames:
        step_indices = step_indices[:n_frames]
    n_panels = len(step_indices)

    def pos_to_px(p):
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

    ncols = min(4, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows),
                             facecolor='#1a1a1a')
    axes = np.atleast_2d(axes)
    cmap = plt.cm.magma

    for idx, ax in enumerate(axes.flat):
        ax.set_facecolor('#1a1a1a')
        if idx >= n_panels:
            ax.axis('off')
            continue
        t = step_indices[idx]
        p_grid = ep_proba[t]  # (width, height)

        ax.imshow((world_img * 0.6).astype(np.uint8))

        p_up = np.repeat(np.repeat(p_grid, tile_size, axis=0),
                         tile_size, axis=1)
        p_up = p_up.T[::-1]
        p_max = p_up.max()
        p_norm = p_up / p_max if p_max > 0 else p_up
        overlay_rgba = cmap(p_norm)
        overlay_rgba[..., 3] = p_norm * 0.85
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
        ax.set_ylim(y_hi, y_lo)
        ax.set_title(f't={t}', fontsize=9, fontweight='bold', color='white',
                     bbox=dict(facecolor='black', alpha=0.6, pad=2))
        ax.axis('off')
        if idx == 0:
            ax.legend(fontsize=7, loc='upper left', facecolor='black',
                      edgecolor='grey', labelcolor='white')

    fig.suptitle(
        f'Decoder P(pos) on world — {layer_name}, '
        f'episode {episode + 1} (seed={env_seed})',
        fontsize=13, color='white')
    fig.tight_layout()
    out = save_dir / f'layer_probmap_world_{layer_name.replace("/", "_")}.svg'
    fig.savefig(out, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close(fig)
    print(f"  Saved {out.name}")


def plot_layer_error_histogram(layer_fold_values, save_dir,
                               layers=None):
    """Histogram of per-timestep Manhattan decoding error for selected layers.

    Args:
        layer_fold_values: {layer_name: np.ndarray of per-timestep errors}
        save_dir: Path to save directory.
        layers: list of layer names to plot. If None, picks dyn/deter,
                dyn/stoch, first enc/cnn layer, and last pol layer.
    """
    if layers is None:
        all_layers = list(layer_fold_values.keys())
        pick = []
        # dyn/deter
        for ln in all_layers:
            if ln == 'dyn/deter':
                pick.append(ln)
                break
        # dyn/stoch
        for ln in all_layers:
            if ln == 'dyn/stoch':
                pick.append(ln)
                break
        # first enc/cnn layer
        enc_cnn = sorted([ln for ln in all_layers if ln.startswith('enc/cnn')])
        if enc_cnn:
            pick.append(enc_cnn[0])
        # last pol layer
        pol = sorted([ln for ln in all_layers if ln.startswith('pol/')])
        if pol:
            pick.append(pol[-1])
        layers = pick

    layers = [ln for ln in layers if ln in layer_fold_values]
    if not layers:
        print("  No matching layers for error histogram — skipping")
        return

    n = len(layers)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5), squeeze=False)

    colors = {'dyn/deter': '#ff5500', 'dyn/stoch': '#ff9900'}
    default_colors = ['#0055cc', '#33aa00', '#996600', '#44ccff']

    for i, ln in enumerate(layers):
        ax = axes[0, i]
        errs = layer_fold_values[ln]
        color = colors.get(ln, default_colors[i % len(default_colors)])
        ax.hist(errs, bins=40, color=color, alpha=0.75, edgecolor='white',
                linewidth=0.5)
        mean_err = np.mean(errs)
        median_err = np.median(errs)
        ax.axvline(mean_err, color='black', linestyle='--', linewidth=1.5,
                   label=f'mean={mean_err:.2f}')
        ax.axvline(median_err, color='grey', linestyle=':', linewidth=1.5,
                   label=f'median={median_err:.2f}')
        ax.set_title(ln, fontsize=10, fontweight='bold')
        ax.set_xlabel('Manhattan error (tiles)')
        if i == 0:
            ax.set_ylabel('Count')
        ax.legend(fontsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Per-Timestep Decoding Error Distribution', fontsize=13)
    fig.tight_layout()
    out = save_dir / 'layer_error_histogram.svg'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out.name}")


def plot_layer_comparison(layer_fold_values, ordered, save_dir, metric='ce_loss',
                          layer_sizes=None):
    """Horizontal boxplot: one box per layer, ordered early → late.

    Args:
        metric: 'ce_loss' (lower = better) or 'r2' (higher = better).
        layer_sizes: optional dict {layer_name: int} of original neuron counts,
                     appended to y-axis tick labels as "(D units)".
    """
    display_order = [ln for ln in ordered if ln in layer_fold_values]
    n = len(display_order)
    if n == 0:
        print("  No layer results to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, max(4, n * 0.5)))

    data = [layer_fold_values[ln] for ln in display_order]
    if layer_sizes:
        labels = [
            (ln_nl + f" ({layer_sizes[ln]})" if ln in layer_sizes else ln_nl)
            for ln, ln_nl in ((ln, ln.replace('/', '/\n')) for ln in display_order)
        ]
    else:
        labels = [ln.replace('/', '/\n') for ln in display_order]

    bp = ax.boxplot(data, vert=False, patch_artist=True,
                    tick_labels=labels, widths=0.6, showfliers=False)

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
        xlabel = 'R² over timesteps (higher = better spatial decoding)'
    elif metric == 'manhattan':
        xlabel = 'Decode error over timesteps (tiles, lower = better)'
    else:
        xlabel = 'Cross-entropy loss over timesteps (lower = better spatial decoding)'
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title('Per-Layer Position Decoding\n'
                 'orange line = median, black number = mean',
                 fontsize=13)
    ax.grid(True, axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=8)

    for i, ln in enumerate(display_order, start=1):
        mean_v = np.mean(layer_fold_values[ln])
        ax.text(mean_v, i, f' {mean_v:.3f}', va='center', fontsize=7, color='black')

    fig.tight_layout()
    out = save_dir / 'layer_comparison.svg'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


def plot_layer_loss_curves(loss_histories, ordered, save_dir):
    """Plot training loss curves for each layer's classification decoder.

    Args:
        loss_histories: {layer_name: [[fold0_losses], [fold1_losses], ...]}
        ordered: list of layer names in display order.
        save_dir: Path to save directory.
    """
    display_order = [ln for ln in ordered if ln in loss_histories]
    n = len(display_order)
    if n == 0:
        return

    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             squeeze=False)

    for idx, ln in enumerate(display_order):
        r, c = idx // ncols, idx % ncols
        ax = axes[r, c]
        fold_curves = loss_histories[ln]
        for fi, curve in enumerate(fold_curves):
            ax.plot(curve, alpha=0.5, linewidth=0.8, label=f'fold {fi}')
        # Plot mean across folds
        min_len = min(len(c) for c in fold_curves)
        arr = np.array([c[:min_len] for c in fold_curves])
        ax.plot(arr.mean(axis=0), color='black', linewidth=1.5, label='mean')
        ax.set_title(ln, fontsize=9)
        ax.set_xlabel('iter')
        ax.set_ylabel('CE loss')
        ax.set_yscale('log')

    # Turn off unused axes
    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].axis('off')

    fig.suptitle('Decoder Training Loss Curves', fontsize=13)
    fig.tight_layout()
    out = save_dir / 'layer_loss_curves.svg'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Decode agent position from world model states')
    parser.add_argument('--data', default=None, help='Path to trajectory pkl directory')
    parser.add_argument('--save', default=None, help='Output directory for plots/results')
    parser.add_argument('--mode', default='standard', choices=['standard', 'layers'],
                        help='"standard": decode deter/stoch (existing). '
                             '"layers": decode from every model layer and produce '
                             'a comparison boxplot. (default: standard)')
    parser.add_argument('--n_iters', type=int, default=5000,
                        help='Training iterations for classification decoder')
    parser.add_argument('--repr', default='all',
                        choices=['deter', 'stoch', 'combined', 'all'],
                        help='Which representation to decode (default: all three)')
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
    parser.add_argument('--kfold_layers', action='store_true', default=False,
                        help='[--mode layers] Use KFold CV instead of LOGO '
                             '(leave-one-episode-out) for classification. '
                             'With 100 episodes LOGO creates 100 folds; KFold '
                             'uses --n_cv_folds (default 5). (default: False)')
    parser.add_argument('--max_samples', type=int, default=10000,
                        help='[--mode layers] Subsample training timesteps to '
                             'at most this many before fitting (random). '
                             'Set to 0 to use all data. (default: 10000)')
    parser.add_argument('--n_cv_folds', type=int, default=5,
                        help='[--mode layers] Number of CV folds. '
                             'Uses KFold, NOT leave-one-episode-out — LOGO with '
                             'many episodes creates O(N_eps × N_layers) jobs '
                             'which dominates runtime. (default: 5)')
    parser.add_argument('--test_data', default=None,
                        help='[--mode layers] Path to a held-out trajectory '
                             'directory. Decoder is trained on --data and '
                             'evaluated on --test_data with no CV. Works with '
                             'classification. '
                             'Fastest option; requires a second trajectory set.')
    parser.add_argument('--holdout_frac', type=float, default=0.2,
                        help='[--mode layers] Fraction of episodes to hold out '
                             'as test set (e.g. 0.2 = 80/20 split). When >0 '
                             'and --test_data is not set, episodes are auto-split '
                             'into train/test, giving 1 classifier per layer '
                             'instead of N_episodes LOGO folds. Set to 0 for '
                             'CV mode. (default: 0.2)')
    parser.add_argument('--patience', type=int, default=500,
                        help='Early stopping patience (in iters) for '
                             'classification decoder. Training stops when '
                             'smoothed loss has not improved for this many '
                             'iters. Set to 0 to disable. (default: 500)')
    parser.add_argument('--max_iters', type=int, default=None,
                        help='Max training iterations (overrides --n_iters). '
                             'When --patience>0, this is the upper bound; '
                             'training may stop earlier. If not set, defaults '
                             'to --n_iters value.')
    parser.add_argument('--resume', default=None,
                        help='[--mode layers] Path to a partial '
                             'layer_decode_checkpoint.pkl to resume from. '
                             'Already-completed layers are skipped. '
                             'If omitted, checkpoints are auto-saved to '
                             '<save>/layer_decode_checkpoint.pkl.')
    parser.add_argument('--from_model', default=None,
                        help='[--mode layers] Path to a saved layer_decoders/ '
                             'directory (from --save_model). Loads pretrained '
                             'decoders and evaluates on --data without training.')
    parser.add_argument('--max_episodes', type=int, default=0,
                        help='Max episodes to load (0=all). When set, loads '
                             'individual episode files to avoid OOM on large '
                             'all_episodes.pkl.')
    parser.add_argument('--min_bbox', type=float, default=0,
                        help='Minimum bounding-box area (tiles²) per episode. '
                             'Episodes where (max_x-min_x)*(max_y-min_y) < this '
                             'are excluded as "stuck". 0 = no filtering. '
                             '(default: 0)')
    parser.add_argument('--from_results', default=None,
                        help='Path to an existing layer_decode_results.pkl '
                             '(or layer_decode_checkpoint.pkl). Regenerates '
                             'plots (layer_comparison, error_histogram) from '
                             'saved per-timestep errors without retraining. '
                             'Requires --save for output directory.')
    args = parser.parse_args()
    # Resolve max_iters: if set, override n_iters
    if args.max_iters is not None:
        args.n_iters = args.max_iters

    # ---- Plot-only from saved results ----
    if args.from_results:
        results_path = Path(args.from_results)
        save_dir = Path(args.save) if args.save else results_path.parent
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(results_path, 'rb') as f:
            saved = pickle.load(f)
        layer_fold_values = saved['layer_fold_values']
        ordered = saved['ordered']
        metric = saved.get('metric', 'manhattan')
        print(f"Loaded {len(ordered)} layers from {results_path.name} "
              f"(metric={metric})")
        plot_layer_comparison(layer_fold_values, ordered, save_dir,
                              metric=metric)
        if metric == 'manhattan':
            plot_layer_error_histogram(layer_fold_values, save_dir)
        print("Done.")
        raise SystemExit(0)

    if not args.data:
        parser.error("--data is required unless using --from_results")
    data_path = Path(args.data)
    save_dir = Path(args.save) if args.save else data_path.parent / 'decoder_results'
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load data — lightweight approach for layer mode to avoid OOM.
    # For layer mode: load episode files list, discover layers from first file,
    # then reload one layer at a time from individual files.
    # For standard mode: load episodes fully (only needs deter/stoch).
    print("Loading trajectory data...")

    # Build episode file list and lightweight metadata
    ep_files = sorted(data_path.glob('episode_*.pkl'))
    if args.max_episodes > 0:
        ep_files = ep_files[:args.max_episodes]

    if args.mode == 'layers' and ep_files:
        # --- Lightweight loading for layer mode ---
        # Discover layers from first file
        with open(ep_files[0], 'rb') as f:
            first_ep = pickle.load(f)
        all_layer_keys = set(
            k[len('act/'):] for k in first_ep if k.startswith('act/'))
        del first_ep

        # Load only pos/metadata from each file
        lightweight_eps = []
        for ep_file in ep_files:
            with open(ep_file, 'rb') as f:
                ep = pickle.load(f)
            light = {k: ep[k] for k in ['player_pos'] if k in ep}
            light['_source_file'] = ep_file
            lightweight_eps.append(light)
            del ep
        gc.collect()
        print(f"  {len(lightweight_eps)} episodes loaded (lightweight)")

        metadata = _load_metadata_only(data_path)

        if args.min_bbox > 0:
            n_before = len(lightweight_eps)
            lightweight_eps = filter_stuck_episodes(lightweight_eps, args.min_bbox)
            print(f"  Bbox filter: {n_before} → {len(lightweight_eps)} episodes "
                  f"(min_bbox={args.min_bbox})")

        # Build file index from surviving episodes
        ep_file_index = [(i, ep['_source_file'])
                         for i, ep in enumerate(lightweight_eps)]

        # Get pos/groups from first layer via file reload
        if not all_layer_keys:
            raise ValueError("No 'act/*' keys in episodes.")
        print(f"  Layers found: {sorted(all_layer_keys)}")
        _first_ln = sorted(all_layer_keys)[0]
        _, pos, groups = reload_layer_from_files(ep_file_index, _first_ln)
        print(f"  pos: {pos.shape}  groups: {groups.shape}")
        if metadata and 'area' in metadata:
            width, height = metadata['area']
        else:
            width = int(pos[:, 0].max()) + 1
            height = int(pos[:, 1].max()) + 1
        print(f"  Grid: {width}x{height}")

        episodes = None  # not used in layer mode

    else:
        # --- Standard mode: full load (only needs deter/stoch) ---
        episodes, metadata = load_episodes(data_path, max_episodes=args.max_episodes)
        print(f"  {len(episodes)} episodes loaded")
        if metadata:
            print(f"  Metadata: {metadata}")
        if args.min_bbox > 0:
            n_before = len(episodes)
            episodes = filter_stuck_episodes(episodes, args.min_bbox)
            print(f"  Bbox filter: {n_before} → {len(episodes)} episodes "
                  f"(min_bbox={args.min_bbox})")
        ep_file_index = None  # not used in standard mode

    # ---- Layer-wise decoding mode ----
    if args.mode == 'layers':
        print("\n=== Per-Layer Position Decoding ===")

        # --- Eval-only from saved decoders ---
        if args.from_model:
            print(f"\n  Loading pretrained decoders from {args.from_model}")
            layers = {}
            for ln in sorted(all_layer_keys):
                X_ln, _, _ = reload_layer_from_files(ep_file_index, ln)
                layers[ln] = X_ln
            layer_fold_values, ordered, metric = eval_layer_decoders(
                args.from_model, layers, pos, width, height)
            layer_sizes = {ln: layers[ln].shape[1] for ln in ordered
                           if ln in layers}
            plot_layer_comparison(layer_fold_values, ordered, save_dir,
                                  metric=metric, layer_sizes=layer_sizes)
            results_file = save_dir / 'layer_decode_results.pkl'
            with open(results_file, 'wb') as f:
                pickle.dump({
                    'layer_fold_values': layer_fold_values,
                    'ordered': ordered,
                    'grid': (width, height),
                    'n_samples': len(pos),
                    'n_episodes': len(np.unique(groups)),
                    'metric': metric,
                    'from_model': str(args.from_model),
                }, f)
            print(f"\nResults saved to {results_file}")
            log_run_info(save_dir, 'decode_position', vars(args),
                         extra={'sub_mode': 'layers/from_model',
                                'n_layers': len(ordered),
                                'n_samples': len(pos),
                                'grid': (width, height)})
            print("Done.")
            raise SystemExit(0)

        checkpoint_path = (Path(args.resume) if args.resume
                           else save_dir / 'layer_decode_checkpoint.pkl')
        loss_histories = {}

        # --- Determine train/test split ---
        has_test = False
        test_ep_file_index = None
        train_mask = None
        if args.test_data:
            print("Loading held-out test trajectories...")
            test_data_path = Path(args.test_data)
            test_ep_files = sorted(test_data_path.glob('episode_*.pkl'))
            # Discover common layers from first test file
            with open(test_ep_files[0], 'rb') as f:
                first_test = pickle.load(f)
            test_layer_keys = set(
                k[len('act/'):] for k in first_test if k.startswith('act/'))
            del first_test
            common = all_layer_keys & test_layer_keys
            all_layer_keys = common
            # Build test file index (lightweight — just file paths)
            test_ep_file_index = [(i, f) for i, f in enumerate(test_ep_files)]
            print(f"  {len(test_ep_files)} test episodes found")
            # Get test pos/groups from first common layer
            _first_test_ln = sorted(common)[0]
            _, pos_test, groups_test = reload_layer_from_files(
                test_ep_file_index, _first_test_ln)
            has_test = True
        elif args.holdout_frac > 0:
            ep_ids = np.unique(groups)
            n_eps = len(ep_ids)
            n_test = max(1, int(round(n_eps * args.holdout_frac)))
            n_train = n_eps - n_test
            rng = np.random.RandomState(42)
            rng.shuffle(ep_ids)
            test_eps = set(ep_ids[:n_test].tolist())
            train_mask = np.array([g not in test_eps for g in groups])
            test_mask = ~train_mask
            print(f"  Auto holdout split: {n_train} train / {n_test} test episodes "
                  f"({train_mask.sum()} / {test_mask.sum()} timesteps)")
            pos_test = pos[test_mask]
            groups_test = groups[test_mask]
            pos = pos[train_mask]
            has_test = True

        # Helper: load one layer, apply train/test split
        def _load_layer(ln):
            """Load a single layer's activations from files, return (X_train, X_test) or (X, None)."""
            import gc as _gc
            if args.test_data and test_ep_file_index is not None:
                X_train, _, _ = reload_layer_from_files(ep_file_index, ln)
                X_test, _, _ = reload_layer_from_files(test_ep_file_index, ln)
                return X_train, X_test
            elif args.holdout_frac > 0 and has_test:
                X_full, _, _ = reload_layer_from_files(ep_file_index, ln)
                X_train = X_full[train_mask]
                X_test = X_full[test_mask]
                del X_full
                _gc.collect()
                return X_train, X_test
            else:
                X_full, _, _ = reload_layer_from_files(ep_file_index, ln)
                return X_full, None

        # Build layers dicts lazily for the decode functions.
        # For holdout/test modes, build both train and test dicts one layer
        # at a time so only one layer's activations are in memory at once
        # during the dict-building phase.
        import gc as _gc
        ordered = _ordered_layers(all_layer_keys)
        layers = {}
        layers_test_dict = {} if has_test else None
        for ln in ordered:
            X_tr, X_te = _load_layer(ln)
            mem_mb = X_tr.nbytes / 1e6
            print(f"  Loaded {ln}: {X_tr.shape} ({mem_mb:.0f} MB)")
            layers[ln] = X_tr
            if X_te is not None:
                layers_test_dict[ln] = X_te

        if has_test:
            if not HAS_TORCH:
                print("Error: torch required for classifier layer decoding.")
                raise SystemExit(1)
            n_iters_layers = args.n_iters
            pat = args.patience
            ms = args.max_samples
            print(f"  Mode: Classification holdout. No CV.  "
                  f"max_iters={n_iters_layers}, patience={pat}, "
                  f"max_samples={ms}, device={args.device}")
            layer_fold_values, ordered, loss_histories, \
                layer_pred_xy, layer_proba, layer_train_pred_xy, layer_clfs = \
                decode_layers_classification_holdout(
                    layers, pos, layers_test_dict, pos_test, width, height,
                    n_iters=n_iters_layers, n_jobs=args.n_jobs,
                    device=args.device, patience=pat,
                    max_samples=ms, checkpoint_path=checkpoint_path)
            metric = 'manhattan'
        else:
            if not HAS_TORCH:
                print("Error: torch required for classifier layer decoding.")
                raise SystemExit(1)
            n_iters_layers = args.n_iters
            pat = args.patience
            ms = args.max_samples
            use_kf = args.kfold_layers
            cv_desc = (f"{args.n_cv_folds}-fold KFold" if use_kf
                       else f"LOGO ({len(np.unique(groups))} folds)")
            print(f"  Using classification decoder "
                  f"(max_iters={n_iters_layers}, patience={pat}, "
                  f"max_samples={ms}, device={args.device}). "
                  f"CV: {cv_desc}. "
                  f"Metric: mean Manhattan decode error (tiles, lower = better).\n"
                  f"  Tip: use default --holdout_frac 0.2 for faster runs.")
            layer_fold_values, ordered, loss_histories = decode_layers(
                layers, pos, groups, width, height,
                n_iters=n_iters_layers, n_jobs=args.n_jobs, device=args.device,
                checkpoint_path=checkpoint_path, patience=pat,
                use_kfold=use_kf, n_cv_folds=args.n_cv_folds,
                max_samples=ms)
            metric = 'manhattan'
            layer_pred_xy = {}
            layer_train_pred_xy = {}
            layer_proba = {}
            layer_clfs = {}

        layer_sizes = {ln: arr.shape[1] for ln, arr in layers.items()}
        plot_layer_comparison(layer_fold_values, ordered, save_dir, metric=metric,
                              layer_sizes=layer_sizes)
        # Error histogram (classification holdout only)
        if metric == 'manhattan' and layer_fold_values:
            plot_layer_error_histogram(layer_fold_values, save_dir)
        # Probmap-on-world for dyn/deter (classification holdout only)
        if layer_proba and has_test and metadata:
            probmap_layer = 'dyn/deter' if 'dyn/deter' in layer_proba else None
            if probmap_layer is None and layer_proba:
                probmap_layer = next(iter(layer_proba))
            if probmap_layer:
                plot_layer_probmap_on_world(
                    layer_proba[probmap_layer], pos_test, groups_test,
                    metadata, probmap_layer, save_dir)
        # Occupancy vs error for dyn/deter (holdout modes only)
        if has_test and layer_pred_xy:
            occ_layer = 'dyn/deter' if 'dyn/deter' in layer_pred_xy else None
            if occ_layer:
                train_pred_ln = layer_train_pred_xy.get(occ_layer, None)
                # pos may be larger than train_pred if subsampling was used
                train_pos_ln = None
                if train_pred_ln is not None:
                    if len(pos) == len(train_pred_ln):
                        train_pos_ln = pos
                    else:
                        # Subsampled: skip train overlay (shapes don't match)
                        train_pred_ln = None
                plot_occupancy_vs_error(
                    pos_test, layer_pred_xy[occ_layer], width, height,
                    save_dir, repr_name=occ_layer.replace('/', '_'),
                    method='classification', metadata=metadata,
                    train_pos=train_pos_ln, train_pred=train_pred_ln)
        print("\n>>> Plots saved. Decoding evaluation complete. <<<")
        if loss_histories:
            plot_layer_loss_curves(loss_histories, ordered, save_dir)

        # Save reusable per-layer decoders if requested
        if args.save_model:
            if layer_clfs:
                # Holdout mode: save the already-trained classifiers directly
                # (no retraining needed)
                print("\n=== Saving Layer Decoders (from holdout training) ===")
                out_dir = save_dir / 'layer_decoders'
                out_dir.mkdir(parents=True, exist_ok=True)
                layer_files = {}
                for ln in ordered:
                    if ln not in layer_clfs:
                        continue
                    clf = layer_clfs[ln]
                    safe = ln.replace('/', '_')
                    fname = f'{safe}.pkl'
                    meta = {
                        'layer_name': ln,
                        'n_units': clf.model[0].in_features,
                        'n_classes': width * height,
                        'width': width, 'height': height,
                        'type': 'classifier', 'grid': (width, height),
                    }
                    save_classifier_model(clf, meta, out_dir / fname)
                    layer_files[ln] = fname
                layer_sizes = {ln: layer_clfs[ln].model[0].in_features
                               for ln in ordered if ln in layer_clfs}
                manifest = {
                    'ordered': ordered,
                    'grid': (width, height),
                    'metric': 'manhattan',
                    'decoder_type': 'classifier',
                    'layer_files': layer_files,
                    'layer_sizes': layer_sizes,
                    'n_train_samples': len(pos),
                }
                with open(out_dir / 'manifest.pkl', 'wb') as f:
                    pickle.dump(manifest, f)
                print(f"  Saved manifest to {out_dir / 'manifest.pkl'}")
            else:
                # CV mode: no single classifier per layer, retrain on full data
                print("\n=== Saving Layer Decoders (retrain on full data) ===")
                print("  Freeing decode results to save memory...")
                layers = None
                layers_test_dict = None
                layer_pred_xy = None
                layer_train_pred_xy = None
                layer_proba = None
                loss_histories = None
                gc.collect()
                save_layer_decoders(
                    None, pos, width, height, ordered, save_dir,
                    n_iters=(args.n_iters if args.max_iters is not None
                             else max(args.n_iters, 50000)),
                    device=args.device, n_jobs=args.n_jobs,
                    patience=args.patience,
                    ep_file_index=ep_file_index,
                    train_mask=train_mask)

        results_file = save_dir / 'layer_decode_results.pkl'
        save_payload = {
            'layer_fold_values': layer_fold_values,
            'ordered': ordered,
            'grid': (width, height),
            'n_samples': len(pos),
            'n_episodes': len(np.unique(groups)),
            'metric': metric,
        }
        with open(results_file, 'wb') as f:
            pickle.dump(save_payload, f)
        print(f"\nResults saved to {results_file}")
        log_run_info(save_dir, 'decode_position', vars(args),
                     extra={'sub_mode': 'layers',
                            'n_layers': len(ordered),
                            'n_samples': len(pos),
                            'grid': (width, height),
                            'metric': metric})
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

    # ---- Classification (pRNN-style) ----
    if not HAS_TORCH:
        print("\nError: torch required for classification decoder")
        raise SystemExit(1)

    print("\n=== Classification Decoding (pRNN-style) ===")
    for name, X in representations.items():
        print(f"\n--- {name} ({X.shape[1]} dims) ---")
        if args.holdout_frac > 0:
            # Single train/test split (fast, no CV folds)
            unique_eps = np.unique(groups)
            rng = np.random.RandomState(42)
            rng.shuffle(unique_eps)
            n_test = max(1, int(args.holdout_frac * len(unique_eps)))
            test_eps = set(unique_eps[:n_test])
            train_mask = np.array([g not in test_eps for g in groups])
            test_mask = ~train_mask
            print(f"  Holdout split: {train_mask.sum()} train, "
                  f"{test_mask.sum()} test "
                  f"({n_test}/{len(unique_eps)} episodes)")

            pos_int = pos.astype(int)
            pos_int[:, 0] = np.clip(pos_int[:, 0], 0, width - 1)
            pos_int[:, 1] = np.clip(pos_int[:, 1], 0, height - 1)
            y_cls = np.ravel_multi_index(
                (pos_int[:, 0], pos_int[:, 1]), (width, height))

            # Train with internal 90/10 val for Manhattan early stopping
            X_tr, y_tr, pi_tr = (X[train_mask], y_cls[train_mask],
                                 pos_int[train_mask])
            rng2 = np.random.RandomState(42)
            n_val = max(1, int(0.1 * train_mask.sum()))
            perm = rng2.permutation(train_mask.sum())
            val_idx, tr_idx = perm[:n_val], perm[n_val:]

            clf = LinearClassifier(X.shape[1], width * height,
                                   device=args.device)
            clf.fit(X_tr[tr_idx], y_tr[tr_idx], n_iters=args.n_iters,
                    verbose=False,
                    X_val=X_tr[val_idx], y_cls_val=y_tr[val_idx],
                    pos_int_val=pi_tr[val_idx],
                    width=width, height=height,
                    manhattan_patience=args.patience)

            pred_cls = clf.predict(X[test_mask])
            pred_xy = np.stack(
                np.unravel_index(pred_cls, (width, height)), axis=1)
            true_xy = pos_int[test_mask]
            errors = np.sum(np.abs(pred_xy - true_xy), axis=1)
            shuffle = np.sum(np.abs(
                np.column_stack([
                    np.random.randint(0, width, len(true_xy)),
                    np.random.randint(0, height, len(true_xy))
                ]) - true_xy), axis=1)
            test_pos = pos[test_mask]

            # Train predictions for occupancy plot
            train_pred_cls = clf.predict(X[train_mask])
            train_pred_xy = np.stack(
                np.unravel_index(train_pred_cls, (width, height)), axis=1)
            train_pos_cls = pos[train_mask]
        else:
            errors, shuffle, pred_xy, true_xy, proba = classification_decode(
                X, pos, groups, width, height, n_iters=args.n_iters,
                n_jobs=args.n_jobs, device=args.device,
                patience=args.patience)
            test_pos = pos
            train_pos_cls = None
            train_pred_xy = None

        print(f"  Mean Manhattan error: {errors.mean():.3f} "
              f"(shuffle: {shuffle.mean():.3f})")
        plot_classification_summary(errors, shuffle, name, save_dir)
        plot_occupancy_vs_error(test_pos, pred_xy, width, height,
                               save_dir, repr_name=name,
                               method='classification',
                               metadata=metadata,
                               train_pos=train_pos_cls,
                               train_pred=train_pred_xy)
        # Probability heatmap for each episode (deter only, LOGO only)
        if name == 'deter' and args.holdout_frac <= 0:
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
                    n_iters=args.n_iters, verbose=False,
                    patience=args.patience)
            save_classifier_model(clf, meta, save_dir / f'classifier_{name}.pkl')

    # Save numerical results
    results_file = save_dir / 'decode_results.pkl'
    save_data = {
        'representations': list(representations.keys()),
        'n_samples': len(pos),
        'n_episodes': len(np.unique(groups)),
        'grid': (width, height),
    }
    with open(results_file, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nResults saved to {results_file}")
    log_run_info(save_dir, 'decode_position', vars(args),
                 extra={'sub_mode': 'standard',
                        'representations': list(representations.keys()),
                        'n_samples': len(pos),
                        'n_episodes': len(np.unique(groups)),
                        'grid': (width, height)})
    print("Done.")
