"""
Trajectory visualization script for interpretability analysis.

Usage:
  python dreamerv3/plot_trajectories.py --data ./logdir/crafter_run1/trajectories
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_episodes(data_path):
    """Load all episodes from trajectory data."""
    data_path = Path(data_path)

    # Try loading all_episodes.pkl first
    all_file = data_path / 'all_episodes.pkl'
    if all_file.exists():
        with open(all_file, 'rb') as f:
            return pickle.load(f)

    # Otherwise load individual episode files
    episodes = []
    for ep_file in sorted(data_path.glob('episode_*.pkl')):
        with open(ep_file, 'rb') as f:
            episodes.append(pickle.load(f))
    return episodes


def plot_trajectories(episodes, save_path=None):
    """Plot all episode trajectories overlaid on a single plot."""
    fig, ax = plt.subplots(figsize=(10, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(episodes)))

    for i, ep in enumerate(episodes):
        pos = ep['player_pos']
        x, y = pos[:, 0], pos[:, 1]

        # Plot trajectory line
        ax.plot(x, y, '-', color=colors[i], alpha=0.7, linewidth=1.5,
                label=f"Ep {ep['episode']} (r={ep['total_reward']:.0f})")

        # Mark start and end
        ax.scatter(x[0], y[0], c='green', s=100, marker='o', zorder=5,
                   edgecolor='white', linewidth=2)
        ax.scatter(x[-1], y[-1], c='red', s=100, marker='X', zorder=5,
                   edgecolor='white', linewidth=2)

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Agent Trajectories Across Episodes')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory plot to {save_path}")

    plt.show()


def plot_trajectory_heatmap(episodes, save_path=None):
    """Plot position heatmap across all episodes."""
    # Collect all positions
    all_pos = np.concatenate([ep['player_pos'] for ep in episodes], axis=0)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Create 2D histogram
    h, xedges, yedges = np.histogram2d(
        all_pos[:, 0], all_pos[:, 1], bins=50)

    im = ax.imshow(h.T, origin='lower', cmap='hot',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   aspect='equal')

    plt.colorbar(im, ax=ax, label='Visit Count')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Position Visitation Heatmap')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved heatmap to {save_path}")

    plt.show()


def plot_activation_by_position(episodes, unit_idx=0, layer='deter', save_path=None):
    """Plot activation of a specific unit as a function of position."""
    if layer not in episodes[0]:
        print(f"Layer '{layer}' not found in data. Available: {list(episodes[0].keys())}")
        return

    # Collect positions and activations
    all_pos = []
    all_act = []

    for ep in episodes:
        pos = ep['player_pos']
        act = ep[layer]

        # Flatten stoch if needed
        if layer == 'stoch' and act.ndim == 3:
            act = act.reshape(act.shape[0], -1)

        all_pos.append(pos)
        all_act.append(act[:, unit_idx])

    all_pos = np.concatenate(all_pos, axis=0)
    all_act = np.concatenate(all_act, axis=0)

    fig, ax = plt.subplots(figsize=(10, 10))

    scatter = ax.scatter(all_pos[:, 0], all_pos[:, 1], c=all_act,
                         cmap='viridis', s=10, alpha=0.7)

    plt.colorbar(scatter, ax=ax, label=f'{layer}[{unit_idx}] activation')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'{layer}[{unit_idx}] Activation by Position')
    ax.set_aspect('equal')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved activation plot to {save_path}")

    plt.show()


def find_spatial_units(episodes, layer='deter', top_k=10):
    """Find units most correlated with position."""
    if layer not in episodes[0]:
        print(f"Layer '{layer}' not found in data.")
        return None

    # Collect all positions and activations
    all_pos = np.concatenate([ep['player_pos'] for ep in episodes], axis=0)
    all_act = np.concatenate([ep[layer] for ep in episodes], axis=0)

    # Flatten stoch if needed
    if layer == 'stoch' and all_act.ndim == 3:
        all_act = all_act.reshape(all_act.shape[0], -1)

    n_units = all_act.shape[1]

    # Compute correlation with x and y
    x_corr = np.array([np.corrcoef(all_pos[:, 0], all_act[:, i])[0, 1]
                       for i in range(n_units)])
    y_corr = np.array([np.corrcoef(all_pos[:, 1], all_act[:, i])[0, 1]
                       for i in range(n_units)])

    # Combined spatial correlation (max of abs correlations)
    spatial_score = np.maximum(np.abs(x_corr), np.abs(y_corr))

    # Get top-k spatial units
    top_indices = np.argsort(spatial_score)[::-1][:top_k]

    print(f"\n=== Top {top_k} Spatially-Tuned Units ({layer}) ===")
    for rank, idx in enumerate(top_indices):
        print(f"  {rank+1}. Unit {idx}: x_corr={x_corr[idx]:.3f}, "
              f"y_corr={y_corr[idx]:.3f}, score={spatial_score[idx]:.3f}")

    return top_indices, x_corr, y_corr


def main():
    parser = argparse.ArgumentParser(description='Plot DreamerV3 trajectories')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to trajectory data directory')
    parser.add_argument('--plot', type=str, default='all',
                        choices=['trajectories', 'heatmap', 'activation', 'spatial', 'all'],
                        help='Type of plot to generate')
    parser.add_argument('--unit', type=int, default=0,
                        help='Unit index for activation plot')
    parser.add_argument('--layer', type=str, default='deter',
                        choices=['deter', 'stoch'],
                        help='Layer to analyze')
    parser.add_argument('--save', type=str, default=None,
                        help='Directory to save plots')
    args = parser.parse_args()

    print(f"Loading episodes from {args.data}")
    episodes = load_episodes(args.data)
    print(f"Loaded {len(episodes)} episodes")

    save_dir = Path(args.save) if args.save else None
    if save_dir:
        save_dir.mkdir(exist_ok=True)

    if args.plot in ('trajectories', 'all'):
        save_path = save_dir / 'trajectories.png' if save_dir else None
        plot_trajectories(episodes, save_path)

    if args.plot in ('heatmap', 'all'):
        save_path = save_dir / 'heatmap.png' if save_dir else None
        plot_trajectory_heatmap(episodes, save_path)

    if args.plot in ('spatial', 'all'):
        find_spatial_units(episodes, args.layer)

    if args.plot in ('activation', 'all'):
        save_path = save_dir / f'{args.layer}_{args.unit}_activation.png' if save_dir else None
        plot_activation_by_position(episodes, args.unit, args.layer, save_path)


if __name__ == '__main__':
    main()
