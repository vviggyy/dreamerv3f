"""
Neural manifold analysis: sRSA, Isomap, and SW distance.

Compares wake (real trajectory) and dream (imagination) activations to assess
whether the world model's dreamed representations occupy the same neural
manifold as wake representations.

Metrics:
  - sRSA: Spearman rank correlation between pairwise position distances and
    pairwise neural cosine distances during wake — measures spatial structure.
  - Isomap: 2D manifold visualization of combined wake+dream activations.
  - SW Distance: Median minimum cosine distance from each dream point to its
    nearest wake neighbor — quantifies manifold proximity.

Usage:
  # Wake vs dream (from dream_decode output):
  MPLBACKEND=Agg python dreamerv3/manifold_analysis.py \
    --data ./logdir/my_run/trajectories \
    --dream_data ./logdir/my_run/dream_results/dream_results.pkl \
    --save ./logdir/my_run/manifold_results

  # Wake vs wake control (second trajectory set):
  MPLBACKEND=Agg python dreamerv3/manifold_analysis.py \
    --data ./logdir/my_run/trajectories_train \
    --dream_data ./logdir/my_run/trajectories_test \
    --save ./logdir/my_run/manifold_results

  # Filter layers:
  MPLBACKEND=Agg python dreamerv3/manifold_analysis.py \
    --data ./logdir/my_run/trajectories \
    --dream_data ./logdir/my_run/dream_results/dream_results.pkl \
    --save ./logdir/my_run/manifold_results \
    --layers dyn/deter dyn/stoch
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import spearmanr
from sklearn.manifold import Isomap

from run_info import log_run_info
from decode_position import load_episodes, _prepare_single_layer, filter_stuck_episodes

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

DREAM_LAYERS = ['dyn/deter', 'dyn/stoch']


def load_wake_activations(data_path, layer_name, max_samples, min_bbox=0):
    """Load wake activations and positions for a single layer.

    Returns:
        X: (N, D) activations
        pos: (N, 2) positions
    """
    episodes, metadata = load_episodes(data_path)
    if min_bbox > 0:
        episodes = filter_stuck_episodes(episodes, min_bbox)
    X, pos, _groups = _prepare_single_layer(episodes, layer_name)

    # Subsample if needed
    if max_samples > 0 and len(X) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), max_samples, replace=False)
        X, pos = X[idx], pos[idx]

    return X.astype(np.float32), pos.astype(np.float32), metadata


def load_dream_activations(dream_path, layer_name):
    """Load dream activations from dream_results.pkl or a trajectory directory.

    If dream_path points to a .pkl file, loads dream_deter/dream_stoch from it.
    If dream_path is a directory, treats it as a second wake trajectory set and
    loads activations using _prepare_single_layer.

    Returns:
        X: (M, D) activations (flattened from (N, H, D) if from dream pkl)
    """
    dream_path = Path(dream_path)

    # Case 1: pkl file from dream_decode
    if dream_path.suffix == '.pkl' or (dream_path.is_file() and not dream_path.is_dir()):
        with open(dream_path, 'rb') as f:
            data = pickle.load(f)

        key_map = {'dyn/deter': 'dream_deter', 'dyn/stoch': 'dream_stoch'}
        pkl_key = key_map.get(layer_name)
        if pkl_key is None or pkl_key not in data:
            raise ValueError(
                f"Layer '{layer_name}' not found in dream pkl. "
                f"Available: {[k for k in data if k.startswith('dream_') and not k.endswith('shape')]}")
        arr = np.array(data[pkl_key], dtype=np.float32)
        # Reshape from (N, H, D) or (N, H, S, C) to (N*H, D)
        if arr.ndim == 4:
            N, H = arr.shape[:2]
            arr = arr.reshape(N * H, -1)
        elif arr.ndim == 3:
            N, H, D = arr.shape
            arr = arr.reshape(N * H, D)
        return arr

    # Case 2: directory — treat as second wake trajectory set
    if dream_path.is_dir():
        episodes, _ = load_episodes(dream_path)
        X, _pos, _groups = _prepare_single_layer(episodes, layer_name)
        return X.astype(np.float32)

    raise ValueError(f"dream_data path not found or unrecognized: {dream_path}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_srsa(X, pos):
    """Spearman RSA: rank correlation between neural and spatial distances.

    Args:
        X: (N, D) activations
        pos: (N, 2) positions

    Returns:
        rho: Spearman correlation coefficient
        pval: p-value
        neural_dists: condensed neural distance vector
        spatial_dists: condensed spatial distance vector
    """
    neural_dists = pdist(X, metric='cosine')
    spatial_dists = pdist(pos, metric='euclidean')
    rho, pval = spearmanr(neural_dists, spatial_dists)
    return rho, pval, neural_dists, spatial_dists


def compute_sw_distance(X_wake, X_dream):
    """Median minimum cosine distance from each dream point to nearest wake point.

    Args:
        X_wake: (N, D) wake activations
        X_dream: (M, D) dream activations

    Returns:
        median_sw: median min cosine distance
        min_dists: (M,) per-dream-point minimum distances
    """
    # Compute in chunks to avoid OOM for large arrays
    chunk_size = 2000
    M = len(X_dream)
    min_dists = np.empty(M, dtype=np.float32)

    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        dists = cdist(X_dream[start:end], X_wake, metric='cosine')
        min_dists[start:end] = dists.min(axis=1)

    median_sw = float(np.median(min_dists))
    return median_sw, min_dists


def compute_isomap(X_wake, X_dream, n_neighbors=150, max_total=8000):
    """Fit Isomap on concatenated wake+dream, return 2D embedding.

    Args:
        X_wake: (N, D)
        X_dream: (M, D)
        n_neighbors: k-NN neighbors for Isomap
        max_total: max combined samples (subsample dream if needed)

    Returns:
        emb_wake: (N, 2)
        emb_dream: (M', 2)
        dream_idx: indices into original X_dream that were kept
    """
    N = len(X_wake)

    # Subsample dream if combined would be too large
    max_dream = max_total - N
    if max_dream <= 0:
        max_dream = min(len(X_dream), 2000)
    dream_idx = np.arange(len(X_dream))
    if len(X_dream) > max_dream:
        rng = np.random.RandomState(42)
        dream_idx = rng.choice(len(X_dream), max_dream, replace=False)

    X_combined = np.vstack([X_wake, X_dream[dream_idx]])

    # Adjust n_neighbors if dataset is small
    n_neighbors = min(n_neighbors, len(X_combined) - 1)

    iso = Isomap(n_neighbors=n_neighbors, n_components=2, metric='cosine')
    emb = iso.fit_transform(X_combined)

    emb_wake = emb[:N]
    emb_dream = emb[N:]
    return emb_wake, emb_dream, dream_idx


# ---------------------------------------------------------------------------
# Plotting (dark theme)
# ---------------------------------------------------------------------------

def _setup_dark_style():
    """Dark theme consistent with repo plotting style."""
    plt.rcParams.update({
        'figure.facecolor': '#1a1a1a',
        'axes.facecolor': '#1a1a1a',
        'axes.edgecolor': '#555555',
        'axes.labelcolor': 'white',
        'text.color': 'white',
        'xtick.color': '#cccccc',
        'ytick.color': '#cccccc',
        'grid.color': '#333333',
        'savefig.facecolor': '#1a1a1a',
    })


def _pos_to_hue(pos):
    """Map (x, y) positions to angular hue via arctan2 (pRNN style)."""
    cx = pos[:, 0].mean()
    cy = pos[:, 1].mean()
    angles = np.arctan2(pos[:, 1] - cy, pos[:, 0] - cx)
    # Normalize to [0, 1]
    hue = (angles + np.pi) / (2 * np.pi)
    return hue


def plot_isomap_position(emb_wake, pos_wake, layer_name, save_dir):
    """Isomap colored by position angle (arctan2 hue)."""
    _setup_dark_style()
    hue = _pos_to_hue(pos_wake)

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    sc = ax.scatter(emb_wake[:, 0], emb_wake[:, 1], c=hue, cmap='hsv',
                    s=4, alpha=0.6, edgecolors='none')
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label('Position angle (arctan2)', fontsize=9)
    ax.set_xlabel('Isomap 1', fontsize=10)
    ax.set_ylabel('Isomap 2', fontsize=10)
    ax.set_title(f'{layer_name} — Isomap colored by position', fontsize=11,
                 fontweight='bold')

    out = save_dir / f'{layer_name.replace("/", "_")}_isomap_position.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out.name}")


def plot_isomap_wakedream(emb_wake, emb_dream, layer_name, save_dir):
    """Isomap with wake (white) vs dream (red) overlay."""
    _setup_dark_style()

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    ax.scatter(emb_wake[:, 0], emb_wake[:, 1], c='white', s=4, alpha=0.3,
               edgecolors='none', label='wake', zorder=2)
    ax.scatter(emb_dream[:, 0], emb_dream[:, 1], c='#ff4444', s=6, alpha=0.5,
               edgecolors='none', label='dream', zorder=3)
    ax.legend(fontsize=9, loc='upper right', facecolor='#333333',
              edgecolor='grey', labelcolor='white')
    ax.set_xlabel('Isomap 1', fontsize=10)
    ax.set_ylabel('Isomap 2', fontsize=10)
    ax.set_title(f'{layer_name} — Wake vs Dream manifold', fontsize=11,
                 fontweight='bold')

    out = save_dir / f'{layer_name.replace("/", "_")}_isomap_wakedream.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out.name}")


def plot_srsa(neural_dists, spatial_dists, rho, layer_name, save_dir,
              max_points=50000):
    """2D histogram of neural vs spatial distances with sRSA value."""
    _setup_dark_style()

    # Subsample for plotting if needed
    if len(neural_dists) > max_points:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(neural_dists), max_points, replace=False)
        nd, sd = neural_dists[idx], spatial_dists[idx]
    else:
        nd, sd = neural_dists, spatial_dists

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    h = ax.hist2d(sd, nd, bins=100, cmap='inferno', norm=matplotlib.colors.LogNorm())
    fig.colorbar(h[3], ax=ax, shrink=0.8, label='count')
    ax.set_xlabel('Spatial distance (Euclidean)', fontsize=10)
    ax.set_ylabel('Neural distance (cosine)', fontsize=10)
    ax.set_title(f'{layer_name} — sRSA = {rho:.4f}', fontsize=11,
                 fontweight='bold')

    out = save_dir / f'{layer_name.replace("/", "_")}_srsa.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out.name}")


def plot_sw_histogram(min_dists, median_sw, layer_name, save_dir):
    """Histogram of per-dream minimum cosine distances to wake."""
    _setup_dark_style()

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.hist(min_dists, bins=80, color='#ff6666', edgecolor='#cc3333', alpha=0.8)
    ax.axvline(median_sw, color='cyan', linestyle='--', linewidth=2,
               label=f'median = {median_sw:.4f}')
    ax.set_xlabel('Min cosine distance to wake', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(f'{layer_name} — SW distance distribution', fontsize=11,
                 fontweight='bold')
    ax.legend(fontsize=9, facecolor='#333333', edgecolor='grey',
              labelcolor='white')

    out = save_dir / f'{layer_name.replace("/", "_")}_swdist.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out.name}")


def plot_summary(results, save_dir):
    """Multi-panel summary figure across all layers."""
    _setup_dark_style()

    layers = list(results.keys())
    n_layers = len(layers)
    if n_layers == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: sRSA bar chart
    ax = axes[0, 0]
    srsa_vals = [results[l]['srsa_rho'] for l in layers]
    bars = ax.bar(range(n_layers), srsa_vals, color='#44aaff', edgecolor='#2277cc')
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([l.replace('/', '\n') for l in layers], fontsize=8)
    ax.set_ylabel('Spearman rho', fontsize=10)
    ax.set_title('sRSA (spatial structure)', fontsize=11, fontweight='bold')
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.5)

    # Panel 2: SW distance bar chart
    ax = axes[0, 1]
    sw_vals = [results[l]['sw_median'] for l in layers]
    ax.bar(range(n_layers), sw_vals, color='#ff6666', edgecolor='#cc3333')
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([l.replace('/', '\n') for l in layers], fontsize=8)
    ax.set_ylabel('Median min cosine dist', fontsize=10)
    ax.set_title('SW distance (manifold proximity)', fontsize=11,
                 fontweight='bold')

    # Panel 3: SW distance box plot
    ax = axes[1, 0]
    sw_data = [results[l]['sw_min_dists'] for l in layers]
    bp = ax.boxplot(sw_data, labels=[l.replace('/', '\n') for l in layers],
                    patch_artist=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor('#ff6666')
        patch.set_alpha(0.6)
    for element in ['whiskers', 'caps', 'medians']:
        for item in bp[element]:
            item.set_color('white')
    ax.set_ylabel('Min cosine dist to wake', fontsize=10)
    ax.set_title('SW distance distribution', fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', labelsize=8)

    # Panel 4: text summary
    ax = axes[1, 1]
    ax.axis('off')
    lines = ['Layer Summary', '─' * 40]
    for l in layers:
        r = results[l]
        lines.append(f"{l}:")
        lines.append(f"  sRSA ρ = {r['srsa_rho']:.4f} (p={r['srsa_pval']:.2e})")
        lines.append(f"  SW median = {r['sw_median']:.4f}")
        lines.append(f"  wake N = {r['n_wake']}, dream N = {r['n_dream']}")
    ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes,
            fontsize=9, fontfamily='monospace', verticalalignment='top',
            color='white')

    fig.suptitle('Neural Manifold Analysis', fontsize=14, fontweight='bold',
                 color='white')
    fig.tight_layout()
    out = save_dir / 'manifold_summary.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Neural manifold analysis (sRSA, Isomap, SW distance)')
    parser.add_argument('--data', required=True,
                        help='Wake trajectory directory (eval_trajectory output)')
    parser.add_argument('--dream_data', required=True,
                        help='Dream results pkl (from dream_decode) or second '
                             'wake trajectory directory')
    parser.add_argument('--save', required=True,
                        help='Output directory')
    parser.add_argument('--layers', nargs='+', default=DREAM_LAYERS,
                        help='Layers to analyze (default: dyn/deter dyn/stoch)')
    parser.add_argument('--max_wake_samples', type=int, default=4000,
                        help='Max wake samples (0=all, default: 4000)')
    parser.add_argument('--max_dream_samples', type=int, default=0,
                        help='Max dream samples for SW distance (0=all)')
    parser.add_argument('--n_neighbors', type=int, default=150,
                        help='Isomap k-NN neighbors (default: 150)')
    parser.add_argument('--metric', default='cosine',
                        help='Distance metric for neural space (default: cosine)')
    parser.add_argument('--min_bbox', type=int, default=0,
                        help='Min bounding box area to filter stuck episodes')
    parser.add_argument('--no_isomap', action='store_true',
                        help='Skip Isomap (faster)')
    args = parser.parse_args()

    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    output_files = []

    for layer_name in args.layers:
        print(f"\n{'='*60}")
        print(f"Layer: {layer_name}")
        print(f"{'='*60}")

        # --- Load wake data ---
        print("Loading wake activations...")
        try:
            X_wake, pos_wake, metadata = load_wake_activations(
                args.data, layer_name, args.max_wake_samples,
                min_bbox=args.min_bbox)
        except (ValueError, KeyError) as e:
            print(f"  SKIP {layer_name}: {e}")
            continue
        print(f"  Wake: {X_wake.shape[0]} samples, dim={X_wake.shape[1]}")

        # --- Load dream data ---
        print("Loading dream activations...")
        try:
            X_dream = load_dream_activations(args.dream_data, layer_name)
        except (ValueError, KeyError) as e:
            print(f"  SKIP {layer_name}: {e}")
            continue

        if args.max_dream_samples > 0 and len(X_dream) > args.max_dream_samples:
            rng = np.random.RandomState(43)
            idx = rng.choice(len(X_dream), args.max_dream_samples, replace=False)
            X_dream = X_dream[idx]
        print(f"  Dream: {X_dream.shape[0]} samples, dim={X_dream.shape[1]}")

        # --- sRSA on wake ---
        print("Computing sRSA...")
        rho, pval, neural_dists, spatial_dists = compute_srsa(X_wake, pos_wake)
        print(f"  sRSA rho={rho:.4f}, p={pval:.2e}")

        # --- SW distance ---
        print("Computing SW distance...")
        median_sw, min_dists = compute_sw_distance(X_wake, X_dream)
        print(f"  SW median={median_sw:.4f}, "
              f"mean={min_dists.mean():.4f}, "
              f"std={min_dists.std():.4f}")

        # --- Isomap ---
        emb_wake, emb_dream = None, None
        if not args.no_isomap:
            print("Computing Isomap...")
            try:
                emb_wake, emb_dream, dream_idx = compute_isomap(
                    X_wake, X_dream, n_neighbors=args.n_neighbors)
                print(f"  Isomap: {len(emb_wake)} wake + {len(emb_dream)} dream points")
            except Exception as e:
                print(f"  Isomap failed: {e}")

        # --- Store results ---
        layer_results = {
            'srsa_rho': float(rho),
            'srsa_pval': float(pval),
            'sw_median': float(median_sw),
            'sw_mean': float(min_dists.mean()),
            'sw_std': float(min_dists.std()),
            'sw_min_dists': min_dists,
            'n_wake': len(X_wake),
            'n_dream': len(X_dream),
        }
        if emb_wake is not None:
            layer_results['isomap_wake'] = emb_wake
            layer_results['isomap_dream'] = emb_dream
            layer_results['pos_wake'] = pos_wake
        all_results[layer_name] = layer_results

        # --- Plots ---
        print("Generating plots...")
        plot_srsa(neural_dists, spatial_dists, rho, layer_name, save_dir)
        plot_sw_histogram(min_dists, median_sw, layer_name, save_dir)
        slug = layer_name.replace('/', '_')
        output_files.extend([f'{slug}_srsa.png', f'{slug}_swdist.png'])

        if emb_wake is not None:
            plot_isomap_position(emb_wake, pos_wake, layer_name, save_dir)
            plot_isomap_wakedream(emb_wake, emb_dream, layer_name, save_dir)
            output_files.extend([
                f'{slug}_isomap_position.png',
                f'{slug}_isomap_wakedream.png',
            ])

    if not all_results:
        print("\nNo layers successfully analyzed.")
        return

    # --- Summary plot ---
    plot_summary(all_results, save_dir)
    output_files.append('manifold_summary.png')

    # --- Save results pkl ---
    results_file = save_dir / 'manifold_results.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(all_results, f)
    output_files.append('manifold_results.pkl')
    print(f"\nResults saved to {results_file}")

    # --- Log provenance ---
    log_run_info(
        save_dir=save_dir,
        stage='manifold_analysis',
        args=vars(args),
        outputs=output_files,
        extra={
            'layers': list(all_results.keys()),
            'srsa': {l: all_results[l]['srsa_rho'] for l in all_results},
            'sw_median': {l: all_results[l]['sw_median'] for l in all_results},
        },
    )

    # Print summary table
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Layer':<15} {'sRSA ρ':>10} {'SW median':>12} {'N_wake':>8} {'N_dream':>8}")
    print(f"{'─'*15} {'─'*10} {'─'*12} {'─'*8} {'─'*8}")
    for l in all_results:
        r = all_results[l]
        print(f"{l:<15} {r['srsa_rho']:>10.4f} {r['sw_median']:>12.4f} "
              f"{r['n_wake']:>8d} {r['n_dream']:>8d}")


if __name__ == '__main__':
    main()
