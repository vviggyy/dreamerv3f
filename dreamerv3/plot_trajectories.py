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
  worldview     - Two-panel: allocentric world + agent obs (single episode)
  allocentric   - Allocentric world view only, all episodes in sequence (GIF)
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


def _precompute_facings(ep):
    """Return list of (fx, fy) facing per step.

    Uses stored player_facing if available.

    Otherwise derives from the action recorded at each step, since in Crafter
    action[i] is the action chosen at step i (based on obs at step i) and it
    sets the player's facing direction at step i.  Carry forward for non-move
    actions (do, sleep, craft, etc.) which leave facing unchanged.

    Action → facing mapping (world tile coordinates, north = -y):
      move_left  (1) → (-1,  0)   west
      move_right (2) → ( 1,  0)   east
      move_up    (3) → ( 0, -1)   north
      move_down  (4) → ( 0,  1)   south
    """
    n = len(ep['player_pos'])
    if 'player_facing' in ep:
        return [tuple(ep['player_facing'][i]) for i in range(n)]

    _MOVE_FACING = {
        1: (-1,  0),   # move_left  → west
        2: ( 1,  0),   # move_right → east
        3: ( 0, -1),   # move_up    → north
        4: ( 0,  1),   # move_down  → south
    }
    actions = ep.get('action', [])
    facings = []
    last = (0, 1)  # default: south
    for i in range(n):
        # facing[i] is set by action[i-1] (the action that produced image[i])
        prev_act = int(actions[i - 1]) if (i > 0 and i - 1 < len(actions)) else 0
        if prev_act in _MOVE_FACING:
            last = _MOVE_FACING[prev_act]
        facings.append(last)
    return facings


def animate_worldview_agentview(
        episodes, metadata=None, tile_size=8, window_tiles=17,
        egocentric_view=0, view_half=4, step_ms=200, trail_length=30,
        episode_idx=0, save_path=None):
    """Animate a two-panel view: allocentric world map (left) + agent image (right).

    Animates a single episode (selected by episode_idx). Every timestep is
    shown — no subsampling. step_ms controls how long each frame is displayed.

    Left panel: full world map cropped and centered on the agent with:
      - fading trail of recent positions
      - translucent grey FOV box (symmetric 9x9 for standard view, or
        directional V×V when egocentric_view > 0)
      - white agent dot with a direction arrow

    Right panel: the stored observation image for that step.

    Parameters
    ----------
    step_ms : int
        Milliseconds per timestep (e.g. 500 = 2 timesteps/sec). Controls
        playback speed only; every step is shown.
    episode_idx : int
        Which episode to animate (0-indexed).
    egocentric_view : int
        If > 0, draw a directional FOV box (V-1 tiles forward, V//2 each side).
        If 0, draw a symmetric (2*view_half+1)^2 FOV box centered on agent.
    view_half : int
        Half-width of the symmetric FOV box in tiles (default 4 → 9×9 tiles).
    window_tiles : int
        Width/height of the left-panel world-map window in tiles.
    """
    episodes = _filter_valid_episodes(episodes, 'animate_worldview_agentview')
    if not episodes:
        print("No valid episodes for worldview animation.")
        return

    if episode_idx >= len(episodes):
        print(f"episode_idx {episode_idx} out of range ({len(episodes)} episodes).")
        return
    ep = episodes[episode_idx]

    world_img, env_seed, tile_size = _render_crafter_world(metadata, tile_size)
    if world_img is None:
        return

    # _render_crafter_world produces south-at-top (due to transpose+flip).
    # Crafter observations have north-at-top (conventional game orientation).
    # Flip the world map so both panels share the same orientation.
    world_img = world_img[::-1].copy()

    img_h, img_w = world_img.shape[:2]
    win_px = window_tiles * tile_size  # left-panel pixel size (square)
    half_win = win_px // 2

    facings = _precompute_facings(ep)
    n_steps = len(ep['player_pos'])

    # Figure layout: left = world, right = agent obs
    obs_h, obs_w = ep['image'][0].shape[:2]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7),
                             gridspec_kw={'width_ratios': [1, 1]})
    ax_world, ax_obs = axes

    # Left panel: static background image placeholder (updated per frame)
    init_world = np.zeros((win_px, win_px, 3), dtype=np.uint8)
    world_im = ax_world.imshow(init_world, origin='upper',
                               extent=[0, win_px, win_px, 0])
    ax_world.set_xlim(0, win_px)
    ax_world.set_ylim(win_px, 0)  # row 0 at top
    ax_world.axis('off')
    ax_world.set_title('World View (allocentric, centered on agent)',
                       fontsize=10)

    # FOV rectangle patch (translucent grey)
    fov_patch = plt.Polygon(np.zeros((4, 2)), closed=True,
                            facecolor='grey', edgecolor='white',
                            alpha=0.35, linewidth=1.2, zorder=5)
    ax_world.add_patch(fov_patch)

    # Agent dot
    agent_dot, = ax_world.plot([], [], 'o', color='white', markersize=7,
                               markeredgecolor='gold', markeredgewidth=1.5,
                               zorder=10)
    # Facing arrow: FancyArrowPatch for easy per-frame position updates
    from matplotlib.patches import FancyArrowPatch
    facing_arrow = FancyArrowPatch(
        (half_win, half_win), (half_win, half_win - tile_size),
        arrowstyle='->', color='gold', linewidth=2,
        mutation_scale=12, zorder=11)
    ax_world.add_patch(facing_arrow)

    # Trail collection
    trail_col = LineCollection([], linewidths=[], colors=[], capstyle='round',
                               zorder=6)
    ax_world.add_collection(trail_col)

    # Right panel: agent observation
    init_obs = np.zeros((obs_h, obs_w, 3), dtype=np.uint8)
    obs_im = ax_obs.imshow(init_obs, origin='upper',
                           extent=[0, obs_w, obs_h, 0])
    ax_obs.set_xlim(0, obs_w)
    ax_obs.set_ylim(obs_h, 0)
    ax_obs.axis('off')
    ax_obs.set_title('Agent Observation', fontsize=10)

    title = fig.suptitle('', fontsize=11, y=0.98)

    # Action name label below the panels
    _CRAFTER_ACTIONS = [
        'noop', 'move_left', 'move_right', 'move_up', 'move_down',
        'do', 'sleep', 'place_stone', 'place_table', 'place_furnace',
        'place_plant', 'make_wood_pickaxe', 'make_stone_pickaxe',
        'make_iron_pickaxe', 'make_wood_sword', 'make_stone_sword',
        'make_iron_sword',
    ]
    action_label = fig.text(0.5, 0.01, '', ha='center', va='bottom',
                            fontsize=13, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3',
                                      facecolor='#1a1a2e', edgecolor='white',
                                      alpha=0.85))

    def _world_to_img(wx, wy):
        """World tile coords → image pixel (col, row) in the flipped world_img.

        After the [::-1] flip, world_img has north at top (conventional
        game orientation), so image_row = world_y * tile_size.
        """
        col = int(wx * tile_size + tile_size // 2)
        row = int(wy * tile_size + tile_size // 2)
        return col, row

    def _fov_corners(agent_col, agent_row, fx, fy, egocentric_view, view_half):
        """Compute 4 corners of the FOV box in LOCAL window pixel coords.

        After the world_img flip, north is at top (north-at-top convention):
          image-col: east = +col  ✓
          image-row: south = +row (row increases downward = southward)  ✓

        Forward direction in image (dcol, drow) per tile:
          fwd = (fx, fy)   — no sign change; +y (south) = +row ✓
        Right-perpendicular (CW rotation in y-down image space):
          CW of (a,b) = (-b, a)
          fwd = (fx, fy) → rgt = (-fy, fx)

        Local coords: agent is always at (half_win, half_win).
        """
        # forward/right unit vectors in image (col, row) per tile
        fwd = np.array([fx, fy], dtype=float)
        rgt = np.array([-fy, fx], dtype=float)  # CW perp of fwd

        cx, cy = half_win, half_win  # agent in local window

        ts = tile_size

        if egocentric_view > 0:
            # Directional box: V-1 forward, V//2 each side, agent's own tile behind
            V = egocentric_view
            fwd_t = V - 1    # tiles ahead
            side_t = V // 2  # tiles each side
            fwd_ext = (fwd_t + 0.5) * ts   # forward from agent center
            bk_ext = 0.5 * ts              # just the agent's own tile behind
            side_ext = (side_t + 0.5) * ts  # each side
            corners = np.array([
                [cx + side_ext * rgt[0] - bk_ext * fwd[0],
                 cy + side_ext * rgt[1] - bk_ext * fwd[1]],
                [cx + side_ext * rgt[0] + fwd_ext * fwd[0],
                 cy + side_ext * rgt[1] + fwd_ext * fwd[1]],
                [cx - side_ext * rgt[0] + fwd_ext * fwd[0],
                 cy - side_ext * rgt[1] + fwd_ext * fwd[1]],
                [cx - side_ext * rgt[0] - bk_ext * fwd[0],
                 cy - side_ext * rgt[1] - bk_ext * fwd[1]],
            ])
        else:
            # Standard view: centered square, no facing dependence
            half_ext = (view_half + 0.5) * ts
            corners = np.array([
                [cx - half_ext, cy - half_ext],
                [cx + half_ext, cy - half_ext],
                [cx + half_ext, cy + half_ext],
                [cx - half_ext, cy + half_ext],
            ])
        return corners

    def _update(step):
        pos = ep['player_pos'][step]
        facing = facings[step]
        fx, fy = facing

        # Agent pixel position in world_img
        a_col, a_row = _world_to_img(pos[0], pos[1])

        # Crop window centered on agent
        c0 = a_col - half_win  # left edge
        r0 = a_row - half_win  # top edge
        c1 = c0 + win_px
        r1 = r0 + win_px

        # Clip to image bounds + pad
        pad_l = max(0, -c0)
        pad_t = max(0, -r0)
        pad_r = max(0, c1 - img_w)
        pad_b = max(0, r1 - img_h)
        cc0, cr0 = max(0, c0), max(0, r0)
        cc1, cr1 = min(img_w, c1), min(img_h, r1)
        window = world_img[cr0:cr1, cc0:cc1].copy()
        if pad_l or pad_t or pad_r or pad_b:
            window = np.pad(window,
                            [(pad_t, pad_b), (pad_l, pad_r), (0, 0)],
                            mode='constant', constant_values=0)
        world_im.set_data(window)

        # FOV box in local (window) pixel coords
        corners = _fov_corners(a_col, a_row, fx, fy, egocentric_view, view_half)
        fov_patch.set_xy(corners)

        # Agent dot (always at window center)
        agent_dot.set_data([half_win], [half_win])

        # Facing arrow: 1.5-tile-length arrow from agent center
        # North-at-top: fwd = (fx, fy), so row increases southward (+fy = south)
        arrow_len = 1.5 * tile_size
        adx = fx * arrow_len
        ady = fy * arrow_len
        posA = (half_win, half_win)
        posB = (half_win + adx, half_win + ady)
        facing_arrow.set_positions(posA, posB)

        # Trail: past positions in local coords (north-at-top: row = y * ts)
        trail_start = max(0, step - trail_length)
        trail_pos = ep['player_pos'][trail_start:step + 1]
        tpx = trail_pos[:, 0] * tile_size + tile_size // 2 - c0
        tpy = trail_pos[:, 1] * tile_size + tile_size // 2 - r0
        if len(tpx) >= 2:
            pts = np.column_stack([tpx, tpy]).reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            n = len(segs)
            t = np.linspace(0, 1, n)
            alphas = 0.15 + 0.85 * (t ** 1.5)
            widths = 1.0 + 3.5 * t
            colors = np.zeros((n, 4))
            colors[:, :3] = [1.0, 0.85, 0.0]  # gold trail
            colors[:, 3] = alphas
            trail_col.set_segments(segs)
            trail_col.set_linewidths(widths)
            trail_col.set_colors(colors)
        else:
            trail_col.set_segments([])

        # Right panel: agent observation
        obs_im.set_data(ep['image'][step])

        # Action label: show action[step-1] — the action that *produced* image[step]
        if step == 0:
            act_display = 'action: start'
        else:
            act_id = int(ep['action'][step - 1]) if step - 1 < len(ep['action']) else 0
            act_name = (_CRAFTER_ACTIONS[act_id]
                        if act_id < len(_CRAFTER_ACTIONS) else str(act_id))
            # blocked if action[step-1] was a move but pos didn't change
            moved = not np.all(ep['player_pos'][step] == ep['player_pos'][step - 1])
            if not moved and act_name.startswith('move_'):
                act_display = f'action: {act_name}  (blocked)'
            else:
                act_display = f'action: {act_name}'
        action_label.set_text(act_display)

        # Title
        title.set_text(
            f'Ep {episode_idx + 1}  step {step}/{n_steps - 1}  '
            f'reward={ep["total_reward"]:.0f}')

        return (world_im, fov_patch, agent_dot, facing_arrow, trail_col,
                obs_im, title, action_label)

    # Every step shown — no subsampling. step_ms controls playback speed only.
    anim = animation.FuncAnimation(
        fig, _update, frames=n_steps, blit=False, interval=step_ms)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.07)  # room for action label

    if save_path:
        # PillowWriter fps = 1000 / step_ms
        writer = animation.PillowWriter(fps=max(1, round(1000 / step_ms)))
        anim.save(str(save_path), writer=writer, dpi=100)
        print(f"Saved worldview animation to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def animate_allocentric(
        episodes, metadata=None, tile_size=8, window_tiles=17,
        egocentric_view=0, view_half=4, step_ms=200, trail_length=30,
        save_path=None):
    """Animate all episodes one after another — allocentric world view only.

    Same as the left panel of animate_worldview_agentview but loops through
    every episode in sequence. Each episode transitions with a brief pause and
    ghost trace of the previous path.

    Parameters
    ----------
    step_ms : int
        Milliseconds per timestep.
    trail_length : int
        Number of recent steps to show in the fading trail.
    window_tiles : int
        Width/height of the cropped world window in tiles.
    egocentric_view : int
        If > 0, draw a directional FOV box.  0 = symmetric square.
    view_half : int
        Half-width of symmetric FOV box in tiles (default 4 → 9×9).
    """
    episodes = _filter_valid_episodes(episodes, 'animate_allocentric')
    if not episodes:
        print("No valid episodes for allocentric animation.")
        return
    episodes = episodes[:10]

    world_img, env_seed, tile_size = _render_crafter_world(metadata, tile_size)
    if world_img is None:
        return

    # Flip so north is at top (same as animate_worldview_agentview)
    world_img = world_img[::-1].copy()
    img_h, img_w = world_img.shape[:2]
    win_px = window_tiles * tile_size
    half_win = win_px // 2

    # Build flat timeline: list of (ep_idx, step_in_ep)
    timeline = []
    for ep_idx, ep in enumerate(episodes):
        for step in range(len(ep['player_pos'])):
            timeline.append((ep_idx, step))
    total_frames = len(timeline)

    if total_frames == 0:
        print("No frames to animate.")
        return

    # Pre-compute facings per episode
    all_facings = [_precompute_facings(ep) for ep in episodes]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.axis('off')
    title = ax.set_title('', fontsize=11, fontweight='bold')

    init_window = np.zeros((win_px, win_px, 3), dtype=np.uint8)
    world_im = ax.imshow(init_window, origin='upper',
                         extent=[0, win_px, win_px, 0])
    ax.set_xlim(0, win_px)
    ax.set_ylim(win_px, 0)

    trail_col = LineCollection([], linewidths=[], colors=[], capstyle='round',
                               zorder=6)
    ax.add_collection(trail_col)

    ghost_lines = []

    fov_patch = plt.Polygon(np.zeros((4, 2)), closed=True,
                            facecolor='grey', edgecolor='white',
                            alpha=0.35, linewidth=1.2, zorder=5)
    ax.add_patch(fov_patch)

    agent_dot, = ax.plot([], [], 'o', color='white', markersize=7,
                         markeredgecolor='gold', markeredgewidth=1.5, zorder=10)

    from matplotlib.patches import FancyArrowPatch
    facing_arrow = FancyArrowPatch(
        (half_win, half_win), (half_win, half_win - tile_size),
        arrowstyle='->', color='gold', linewidth=2,
        mutation_scale=12, zorder=11)
    ax.add_patch(facing_arrow)

    def _world_to_img(wx, wy):
        col = int(wx * tile_size + tile_size // 2)
        row = int(wy * tile_size + tile_size // 2)
        return col, row

    def _fov_corners_local(fx, fy):
        cx, cy = half_win, half_win
        ts = tile_size
        fwd = np.array([fx, fy], dtype=float)
        rgt = np.array([-fy, fx], dtype=float)
        if egocentric_view > 0:
            V = egocentric_view
            fwd_ext = (V - 1 + 0.5) * ts
            bk_ext = 0.5 * ts
            side_ext = (V // 2 + 0.5) * ts
            corners = np.array([
                [cx + side_ext * rgt[0] - bk_ext * fwd[0],
                 cy + side_ext * rgt[1] - bk_ext * fwd[1]],
                [cx + side_ext * rgt[0] + fwd_ext * fwd[0],
                 cy + side_ext * rgt[1] + fwd_ext * fwd[1]],
                [cx - side_ext * rgt[0] + fwd_ext * fwd[0],
                 cy - side_ext * rgt[1] + fwd_ext * fwd[1]],
                [cx - side_ext * rgt[0] - bk_ext * fwd[0],
                 cy - side_ext * rgt[1] - bk_ext * fwd[1]],
            ])
        else:
            half_ext = (view_half + 0.5) * ts
            corners = np.array([
                [cx - half_ext, cy - half_ext],
                [cx + half_ext, cy - half_ext],
                [cx + half_ext, cy + half_ext],
                [cx - half_ext, cy + half_ext],
            ])
        return corners

    prev_ep_idx = -1

    def _update(frame):
        nonlocal prev_ep_idx, ghost_lines

        ep_idx, step = timeline[frame]
        ep = episodes[ep_idx]
        facings = all_facings[ep_idx]
        n_steps = len(ep['player_pos'])

        # On episode transition: draw ghost of completed episode
        if ep_idx != prev_ep_idx and prev_ep_idx >= 0:
            prev_ep = episodes[prev_ep_idx]
            px_prev = prev_ep['player_pos'][:, 0] * tile_size + tile_size // 2
            py_prev = prev_ep['player_pos'][:, 1] * tile_size + tile_size // 2
            # draw ghost on the world_img coords — approximate placement
            # (ghost is in world space; we draw it at a fixed world offset)
            # Instead, record ghost as world coords for the trail
            ghost_lines.append((prev_ep['player_pos'].copy()))
            trail_col.set_segments([])
        prev_ep_idx = ep_idx

        pos = ep['player_pos'][step]
        fx, fy = facings[step]
        a_col, a_row = _world_to_img(pos[0], pos[1])

        # Crop window
        c0, r0 = a_col - half_win, a_row - half_win
        c1, r1 = c0 + win_px, r0 + win_px
        pad_l = max(0, -c0);  pad_t = max(0, -r0)
        pad_r = max(0, c1 - img_w); pad_b = max(0, r1 - img_h)
        cc0, cr0 = max(0, c0), max(0, r0)
        cc1, cr1 = min(img_w, c1), min(img_h, r1)
        window = world_img[cr0:cr1, cc0:cc1].copy()
        if pad_l or pad_t or pad_r or pad_b:
            window = np.pad(window, [(pad_t, pad_b), (pad_l, pad_r), (0, 0)],
                            mode='constant', constant_values=0)
        world_im.set_data(window)

        # FOV box
        fov_patch.set_xy(_fov_corners_local(fx, fy))

        # Agent dot + arrow (always at window centre)
        agent_dot.set_data([half_win], [half_win])
        arrow_len = 1.5 * tile_size
        facing_arrow.set_positions(
            (half_win, half_win),
            (half_win + fx * arrow_len, half_win + fy * arrow_len))

        # Fading trail (current episode only)
        t_start = max(0, step - trail_length)
        trail_pos = ep['player_pos'][t_start:step + 1]
        tpx = trail_pos[:, 0] * tile_size + tile_size // 2 - c0
        tpy = trail_pos[:, 1] * tile_size + tile_size // 2 - r0
        if len(tpx) >= 2:
            pts = np.column_stack([tpx, tpy]).reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            n = len(segs)
            t = np.linspace(0, 1, n)
            alphas = 0.15 + 0.85 * (t ** 1.5)
            widths = 1.0 + 3.5 * t
            cols = np.zeros((n, 4))
            cols[:, :3] = [1.0, 0.85, 0.0]
            cols[:, 3] = alphas
            trail_col.set_segments(segs)
            trail_col.set_linewidths(widths)
            trail_col.set_colors(cols)
        else:
            trail_col.set_segments([])

        title.set_text(
            f'Ep {ep_idx + 1}/{len(episodes)}  '
            f'step {step}/{n_steps - 1}  '
            f'reward={ep["total_reward"]:.0f}')

        return world_im, fov_patch, agent_dot, facing_arrow, trail_col, title

    plt.tight_layout()
    anim = animation.FuncAnimation(
        fig, _update, frames=total_frames, blit=False, interval=step_ms)

    if save_path:
        writer = animation.PillowWriter(fps=max(1, round(1000 / step_ms)))
        anim.save(str(save_path), writer=writer, dpi=100)
        print(f"Saved allocentric animation to {save_path}")
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
                        choices=['trajectories', 'heatmap', 'activation', 'spatial', 'world', 'fullworld', 'animate', 'animate_world', 'worldview', 'all'],
                        help='Type of plot to generate')
    parser.add_argument('--unit', type=int, default=0,
                        help='Unit index for activation plot')
    parser.add_argument('--layer', type=str, default='deter',
                        choices=['deter', 'stoch'],
                        help='Layer to analyze')
    parser.add_argument('--save', type=str, default=None,
                        help='Directory to save plots')
    parser.add_argument('--egocentric_view', type=int, default=0,
                        help='V for directional FOV box (0 = symmetric 9x9)')
    parser.add_argument('--view_half', type=int, default=4,
                        help='Half-width of symmetric FOV box in tiles (default 4 → 9x9)')
    parser.add_argument('--window_tiles', type=int, default=17,
                        help='Left-panel world window size in tiles (default 17)')
    parser.add_argument('--step_ms', type=int, default=200,
                        help='Milliseconds per timestep for worldview (default 200 = 5 steps/sec)')
    parser.add_argument('--episode_idx', type=int, default=0,
                        help='Which episode to animate for worldview (default 0)')
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

    if args.plot in ('worldview',):
        save_path = save_dir / 'worldview_agentview.gif' if save_dir else None
        animate_worldview_agentview(
            episodes, metadata,
            window_tiles=args.window_tiles,
            egocentric_view=args.egocentric_view,
            view_half=args.view_half,
            step_ms=args.step_ms,
            episode_idx=args.episode_idx,
            save_path=save_path)

    if args.plot in ('allocentric',):
        save_path = save_dir / 'allocentric.gif' if save_dir else None
        animate_allocentric(
            episodes, metadata,
            window_tiles=args.window_tiles,
            egocentric_view=args.egocentric_view,
            view_half=args.view_half,
            step_ms=args.step_ms,
            trail_length=40,
            save_path=save_path)


if __name__ == '__main__':
    main()
