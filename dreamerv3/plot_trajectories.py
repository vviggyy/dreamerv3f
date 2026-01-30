"""
Trajectory visualization script for interpretability analysis.

Usage:
  python dreamerv3/plot_trajectories.py --data ./trajectories --plot all --save ./plots

Plot types:
  trajectories  - Overlay all episode paths
  heatmap       - 2D visitation frequency
  activation    - Unit activation by position
  spatial       - Find spatially-tuned units
  world         - Trajectories on stitched world view (from observations)
  fullworld     - Trajectories on full Crafter world (requires env_seed)
  all           - Generate all plots
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
import numpy as np


def load_episodes(data_path):
    """Load all episodes from trajectory data.

    Returns:
        episodes: List of episode dicts
        metadata: Dict with env_seed, task, etc (or None for old format)
    """
    data_path = Path(data_path)
    metadata = None

    # Try loading all_episodes.pkl first
    all_file = data_path / 'all_episodes.pkl'
    if all_file.exists():
        with open(all_file, 'rb') as f:
            data = pickle.load(f)
        # Handle new format with metadata
        if isinstance(data, dict) and 'episodes' in data:
            metadata = {k: v for k, v in data.items() if k != 'episodes'}
            return data['episodes'], metadata
        # Old format - just list of episodes
        return data, None

    # Otherwise load individual episode files
    episodes = []
    for ep_file in sorted(data_path.glob('episode_*.pkl')):
        with open(ep_file, 'rb') as f:
            episodes.append(pickle.load(f))
    return episodes, None


def _filter_valid_episodes(episodes, context=''):
    """Filter out episodes with missing or empty player_pos. Returns valid list."""
    valid = []
    for i, ep in enumerate(episodes):
        pp = ep.get('player_pos')
        if pp is None or len(pp) == 0:
            print(f"WARNING: Skipping episode {ep.get('episode', i)} — "
                  f"empty player_pos{' in ' + context if context else ''}")
            continue
        valid.append(ep)
    return valid


def plot_trajectories(episodes, save_path=None):
    """Plot all episode trajectories overlaid on a single plot."""
    episodes = _filter_valid_episodes(episodes, 'plot_trajectories')
    if not episodes:
        print("No valid episodes to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(episodes)))

    for i, ep in enumerate(episodes):
        pos = ep['player_pos']
        x, y = pos[:, 0], pos[:, 1]

        # Plot trajectory line
        ax.plot(x, y, '-', color=colors[i], alpha=0.7, linewidth=1.5,
                label=f"Ep {ep['episode']} (r={ep['total_reward']:.0f})")

        # Mark start and end
        ax.scatter(x[0], y[0], c='green', s=30, marker='o', zorder=5,
                   edgecolor='none')
        ax.scatter(x[-1], y[-1], c='red', s=15, marker='o', zorder=5,
                   edgecolor='none')

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
    episodes = _filter_valid_episodes(episodes, 'plot_trajectory_heatmap')
    if not episodes:
        print("No valid episodes for heatmap.")
        return

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
    episodes = _filter_valid_episodes(episodes, 'plot_activation_by_position')
    if not episodes:
        print("No valid episodes for activation plot.")
        return

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


def plot_world_overlay(episodes, tile_size=7, save_path=None):
    """Plot trajectories overlaid on stitched world view from observations."""
    episodes = _filter_valid_episodes(episodes, 'plot_world_overlay')
    if not episodes:
        print("No valid episodes for world overlay.")
        return

    # Collect all positions and images
    all_pos = []
    all_imgs = []
    for ep in episodes:
        for i in range(len(ep['player_pos'])):
            all_pos.append(ep['player_pos'][i])
            all_imgs.append(ep['image'][i])

    all_pos = np.array(all_pos)
    all_imgs = np.array(all_imgs)
    obs_size = all_imgs[0].shape[0]  # Usually 64

    # Get position bounds
    x_min, x_max = int(all_pos[:, 0].min()), int(all_pos[:, 0].max())
    y_min, y_max = int(all_pos[:, 1].min()), int(all_pos[:, 1].max())

    # Build world canvas by averaging overlapping observations
    canvas_h = (y_max - y_min + 1) * tile_size + obs_size
    canvas_w = (x_max - x_min + 1) * tile_size + obs_size
    world_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    world_count = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    for pos, img in zip(all_pos, all_imgs):
        x_idx = int(pos[0] - x_min) * tile_size
        y_idx = int(pos[1] - y_min) * tile_size
        # Flip y for image coordinates
        y_idx = canvas_h - y_idx - obs_size

        # Compute valid region bounds
        y_end = min(y_idx + obs_size, canvas_h)
        x_end = min(x_idx + obs_size, canvas_w)
        y_start = max(y_idx, 0)
        x_start = max(x_idx, 0)

        img_y_start = y_start - y_idx
        img_x_start = x_start - x_idx
        img_y_end = img_y_start + (y_end - y_start)
        img_x_end = img_x_start + (x_end - x_start)

        world_img[y_start:y_end, x_start:x_end] += img[img_y_start:img_y_end, img_x_start:img_x_end].astype(np.float32)
        world_count[y_start:y_end, x_start:x_end] += 1

    # Average overlapping regions
    world_count = np.maximum(world_count, 1)
    world_img = (world_img / world_count[:, :, None]).astype(np.uint8)

    # Plot with trajectory overlay
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(world_img)

    # Convert positions to pixel coordinates on canvas
    colors = plt.cm.tab10(np.linspace(0, 1, len(episodes)))
    for i, ep in enumerate(episodes):
        pos = ep['player_pos']
        px = (pos[:, 0] - x_min) * tile_size + obs_size // 2
        py = canvas_h - ((pos[:, 1] - y_min) * tile_size + obs_size // 2)
        ax.plot(px, py, '-', color=colors[i], linewidth=2, alpha=0.8,
                label=f"Ep {ep['episode']} (r={ep['total_reward']:.0f})")
        ax.plot(px[0], py[0], 'o', color='lime', markersize=4, zorder=5)
        ax.plot(px[-1], py[-1], 'o', color='red', markersize=3, zorder=5)

    ax.set_title('Agent Trajectories on World View')
    ax.legend(loc='upper right', fontsize=8)
    ax.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved world overlay to {save_path}")

    plt.show()


def _render_crafter_world(metadata=None, tile_size=8):
    """Render the full Crafter world map image.

    Returns (world_img, env_seed, tile_size) or (None, None, None) on failure.
    """
    try:
        import crafter
    except ImportError:
        print("Crafter not installed, skipping fullworld plot")
        return None, None, tile_size

    env_seed = metadata.get('env_seed') if metadata else None
    if env_seed is None:
        print("No env_seed in metadata, using seed=42 (world may not match)")
        env_seed = 42

    area = tuple(metadata.get('area', (64, 64))) if metadata else (64, 64)
    env = crafter.Env(area=area, view=(9, 9), size=(64, 64), seed=env_seed)
    env.reset()

    world = env._world
    textures = env._textures
    mat_map = world._mat_map
    mat_names = world._mat_names

    world_img = np.zeros((mat_map.shape[0] * tile_size, mat_map.shape[1] * tile_size, 3), dtype=np.uint8)
    for x in range(mat_map.shape[0]):
        for y in range(mat_map.shape[1]):
            mat_id = mat_map[x, y]
            mat_name = mat_names.get(mat_id, 'unknown')
            if mat_name:
                texture = textures.get(mat_name, (tile_size, tile_size))
                if texture.shape[-1] == 4:
                    texture = texture[..., :3]
                px, py = x * tile_size, y * tile_size
                world_img[px:px+tile_size, py:py+tile_size] = texture

    for obj in world.objects:
        texture = textures.get(obj.texture, (tile_size, tile_size))
        if texture.shape[-1] == 4:
            alpha = texture[..., 3:].astype(np.float32) / 255
            rgb = texture[..., :3].astype(np.float32)
            px, py = int(obj.pos[0]) * tile_size, int(obj.pos[1]) * tile_size
            if 0 <= px < world_img.shape[0] - tile_size and 0 <= py < world_img.shape[1] - tile_size:
                current = world_img[px:px+tile_size, py:py+tile_size].astype(np.float32)
                blended = alpha * rgb + (1 - alpha) * current
                world_img[px:px+tile_size, py:py+tile_size] = blended.astype(np.uint8)

    world_img = world_img.transpose(1, 0, 2)[::-1]
    return world_img, env_seed, tile_size


def plot_fullworld_overlay(episodes, metadata=None, tile_size=8, save_path=None):
    """Plot trajectories on full Crafter world map."""
    episodes = _filter_valid_episodes(episodes, 'plot_fullworld_overlay')
    if not episodes:
        print("No valid episodes for fullworld overlay.")
        return

    world_img, env_seed, tile_size = _render_crafter_world(metadata, tile_size)
    if world_img is None:
        return

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.imshow(world_img)

    colors = plt.cm.tab10(np.linspace(0, 1, len(episodes)))
    for i, ep in enumerate(episodes):
        pos = ep['player_pos']
        px = pos[:, 0] * tile_size + tile_size // 2
        py = world_img.shape[0] - (pos[:, 1] * tile_size + tile_size // 2)
        ax.plot(px, py, '-', color=colors[i], linewidth=2, alpha=0.8,
                label=f"Ep {ep['episode']} (r={ep['total_reward']:.0f})")
        ax.plot(px[0], py[0], 'o', color='lime', markersize=4, zorder=5)
        ax.plot(px[-1], py[-1], 'o', color='red', markersize=3, zorder=5)

    ax.set_title(f'Agent Trajectories on Full World (seed={env_seed})')
    ax.legend(loc='upper right', fontsize=8)
    ax.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved fullworld overlay to {save_path}")

    plt.show()


def animate_trajectories(episodes, save_path=None, fps=30, trail_length=40):
    """Animate agent tracing trajectories sequentially with a fading trail.

    Each episode is drawn one after another. The current position is shown as
    a bright dot with a trailing line that fades and thins behind it. Previous
    episodes remain as faint traces so context accumulates.
    """
    episodes = _filter_valid_episodes(episodes, 'animate_trajectories')
    if not episodes:
        print("No valid episodes for animation.")
        return

    # Build a single timeline: list of (x, y, episode_idx) per frame
    timeline_x, timeline_y, timeline_ep = [], [], []
    ep_boundaries = [0]  # frame index where each episode starts
    for idx, ep in enumerate(episodes):
        pos = ep['player_pos']
        timeline_x.extend(pos[:, 0].tolist())
        timeline_y.extend(pos[:, 1].tolist())
        timeline_ep.extend([idx] * len(pos))
        ep_boundaries.append(len(timeline_x))

    timeline_x = np.array(timeline_x)
    timeline_y = np.array(timeline_y)
    timeline_ep = np.array(timeline_ep)
    total_frames = len(timeline_x)

    # Figure setup
    fig, ax = plt.subplots(figsize=(10, 10))
    pad = 2
    ax.set_xlim(timeline_x.min() - pad, timeline_x.max() + pad)
    ax.set_ylim(timeline_y.min() - pad, timeline_y.max() + pad)
    ax.set_aspect('equal')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.grid(True, alpha=0.3)
    title = ax.set_title('')

    # Artists
    trail_color = np.array([0.2, 0.6, 1.0])  # blue trail
    ghost_lines = []  # completed episode traces
    trail_col = LineCollection([], linewidths=[], colors=[], capstyle='round')
    ax.add_collection(trail_col)
    head_dot, = ax.plot([], [], 'o', color='white', markersize=7,
                        zorder=10, markeredgecolor=trail_color, markeredgewidth=2)
    spawn_dot, = ax.plot([], [], 'o', color='lime', markersize=5, zorder=9)

    current_ep_start = 0

    def _update(frame):
        nonlocal current_ep_start, ghost_lines

        ep_idx = int(timeline_ep[frame])

        # Detect episode transition — freeze previous episode as ghost
        if frame > 0 and timeline_ep[frame] != timeline_ep[frame - 1]:
            prev_start = current_ep_start
            prev_end = frame
            gx = timeline_x[prev_start:prev_end]
            gy = timeline_y[prev_start:prev_end]
            ghost, = ax.plot(gx, gy, '-', color='grey', alpha=0.25,
                             linewidth=0.8, zorder=1)
            ghost_lines.append(ghost)
            current_ep_start = frame

        # Build fading trail for current episode
        seg_start = max(current_ep_start, frame - trail_length)
        xs = timeline_x[seg_start:frame + 1]
        ys = timeline_y[seg_start:frame + 1]

        if len(xs) >= 2:
            points = np.column_stack([xs, ys]).reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            n_seg = len(segments)
            # Fade: oldest segment = 0, newest = 1
            t = np.linspace(0, 1, n_seg)
            alphas = t ** 1.5  # ease-in for a smooth fade
            widths = 1.0 + 3.0 * t  # thin -> thick
            colors = np.zeros((n_seg, 4))
            colors[:, :3] = trail_color
            colors[:, 3] = alphas
            trail_col.set_segments(segments)
            trail_col.set_linewidths(widths)
            trail_col.set_colors(colors)
        else:
            trail_col.set_segments([])

        # Head dot
        head_dot.set_data([timeline_x[frame]], [timeline_y[frame]])

        # Spawn marker for current episode
        spawn_dot.set_data([timeline_x[current_ep_start]],
                           [timeline_y[current_ep_start]])

        ep_num = ep_idx + 1
        step_in_ep = frame - current_ep_start
        r = episodes[ep_idx]['total_reward']
        title.set_text(f'Episode {ep_num}/{len(episodes)}  '
                       f'step {step_in_ep}  reward={r:.0f}')

        return trail_col, head_dot, spawn_dot, title

    # Sub-sample frames if there are too many (keep it under ~60s at target fps)
    max_frames = fps * 60
    if total_frames > max_frames:
        step = total_frames // max_frames
        frames = list(range(0, total_frames, step))
    else:
        frames = list(range(total_frames))

    anim = animation.FuncAnimation(
        fig, _update, frames=frames, blit=False, interval=1000 // fps)

    if save_path:
        writer = animation.PillowWriter(fps=fps)
        anim.save(str(save_path), writer=writer, dpi=100)
        print(f"Saved trajectory animation to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def animate_fullworld_trajectories(episodes, metadata=None, tile_size=8,
                                   save_path=None, fps=30, trail_length=40):
    """Animate agent on the full Crafter world map with a fading trail."""
    episodes = _filter_valid_episodes(episodes, 'animate_fullworld_trajectories')
    if not episodes:
        print("No valid episodes for fullworld animation.")
        return

    world_img, env_seed, tile_size = _render_crafter_world(metadata, tile_size)
    if world_img is None:
        return

    img_h = world_img.shape[0]

    # Convert all positions to pixel coords up front
    def to_px(pos):
        px = pos[:, 0] * tile_size + tile_size // 2
        py = img_h - (pos[:, 1] * tile_size + tile_size // 2)
        return px, py

    # Build timeline in pixel space
    timeline_x, timeline_y, timeline_ep = [], [], []
    for idx, ep in enumerate(episodes):
        px, py = to_px(ep['player_pos'])
        timeline_x.extend(px.tolist())
        timeline_y.extend(py.tolist())
        timeline_ep.extend([idx] * len(px))

    timeline_x = np.array(timeline_x)
    timeline_y = np.array(timeline_y)
    timeline_ep = np.array(timeline_ep)
    total_frames = len(timeline_x)

    if total_frames == 0:
        print("No player_pos data found in episodes, skipping animation.")
        return

    # Figure — crop to visited region with some padding
    pad_px = tile_size * 6
    x_lo = max(0, timeline_x.min() - pad_px)
    x_hi = min(world_img.shape[1], timeline_x.max() + pad_px)
    y_lo = max(0, timeline_y.min() - pad_px)
    y_hi = min(img_h, timeline_y.max() + pad_px)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(world_img)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_hi, y_lo)  # imshow has y-down
    ax.axis('off')
    title = ax.set_title('', color='white', fontsize=12, fontweight='bold',
                         bbox=dict(facecolor='black', alpha=0.7, pad=4))

    trail_color = np.array([1.0, 0.85, 0.0])  # gold trail on the map
    ghost_lines = []
    trail_col = LineCollection([], linewidths=[], colors=[], capstyle='round')
    ax.add_collection(trail_col)
    head_dot, = ax.plot([], [], 'o', color='white', markersize=8,
                        zorder=10, markeredgecolor=trail_color,
                        markeredgewidth=2.5)
    spawn_dot, = ax.plot([], [], 'o', color='lime', markersize=6, zorder=9)

    current_ep_start = 0

    def _update(frame):
        nonlocal current_ep_start, ghost_lines

        ep_idx = int(timeline_ep[frame])

        if frame > 0 and timeline_ep[frame] != timeline_ep[frame - 1]:
            prev_start = current_ep_start
            gx = timeline_x[prev_start:frame]
            gy = timeline_y[prev_start:frame]
            ghost, = ax.plot(gx, gy, '-', color='cyan', alpha=0.18,
                             linewidth=0.8, zorder=2)
            ghost_lines.append(ghost)
            current_ep_start = frame

        seg_start = max(current_ep_start, frame - trail_length)
        xs = timeline_x[seg_start:frame + 1]
        ys = timeline_y[seg_start:frame + 1]

        if len(xs) >= 2:
            points = np.column_stack([xs, ys]).reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            n_seg = len(segments)
            t = np.linspace(0, 1, n_seg)
            alphas = 0.15 + 0.85 * (t ** 1.5)
            widths = 1.0 + 3.5 * t
            colors = np.zeros((n_seg, 4))
            colors[:, :3] = trail_color
            colors[:, 3] = alphas
            trail_col.set_segments(segments)
            trail_col.set_linewidths(widths)
            trail_col.set_colors(colors)
        else:
            trail_col.set_segments([])

        head_dot.set_data([timeline_x[frame]], [timeline_y[frame]])
        spawn_dot.set_data([timeline_x[current_ep_start]],
                           [timeline_y[current_ep_start]])

        ep_num = ep_idx + 1
        step_in_ep = frame - current_ep_start
        r = episodes[ep_idx]['total_reward']
        title.set_text(f'Episode {ep_num}/{len(episodes)}  '
                       f'step {step_in_ep}  reward={r:.0f}')

        return trail_col, head_dot, spawn_dot, title

    max_frames = fps * 60
    if total_frames > max_frames:
        step = total_frames // max_frames
        frames = list(range(0, total_frames, step))
    else:
        frames = list(range(total_frames))

    anim = animation.FuncAnimation(
        fig, _update, frames=frames, blit=False, interval=1000 // fps)

    if save_path:
        writer = animation.PillowWriter(fps=fps)
        anim.save(str(save_path), writer=writer, dpi=60)
        print(f"Saved fullworld trajectory animation to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def find_spatial_units(episodes, layer='deter', top_k=10):
    """Find units most correlated with position."""
    episodes = _filter_valid_episodes(episodes, 'find_spatial_units')
    if not episodes:
        print("No valid episodes for spatial analysis.")
        return None

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
                        choices=['trajectories', 'heatmap', 'activation', 'spatial', 'world', 'fullworld', 'animate', 'animate_world', 'all'],
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
    episodes, metadata = load_episodes(args.data)
    print(f"Loaded {len(episodes)} episodes")
    if not episodes:
        print("ERROR: No episodes found. Check that --data points to a "
              "directory containing all_episodes.pkl or episode_*.pkl files.")
        return
    for i, ep in enumerate(episodes):
        pp = ep.get('player_pos')
        shape = pp.shape if pp is not None else 'MISSING'
        print(f"  Episode {i}: player_pos shape={shape}, keys={list(ep.keys())}")
    if metadata:
        print(f"Metadata: {metadata}")

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

    if args.plot in ('world', 'all'):
        save_path = save_dir / 'world_overlay.png' if save_dir else None
        plot_world_overlay(episodes, save_path=save_path)

    if args.plot in ('fullworld', 'all'):
        save_path = save_dir / 'fullworld_overlay.png' if save_dir else None
        plot_fullworld_overlay(episodes, metadata, save_path=save_path)

    if args.plot in ('animate', 'all'):
        save_path = save_dir / 'trajectories.gif' if save_dir else None
        animate_trajectories(episodes, save_path=save_path)

    if args.plot in ('animate_world', 'all'):
        save_path = save_dir / 'fullworld_trajectories.gif' if save_dir else None
        animate_fullworld_trajectories(episodes, metadata, save_path=save_path)


if __name__ == '__main__':
    main()
