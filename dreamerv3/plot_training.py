"""
plot_training.py — visualize DreamerV3 training progress from logdir.

Reads scores.jsonl and metrics.jsonl written during training.

Plots:
  1. Episode score over training steps (scatter + smoothed)
  2. Cumulative reward over training steps
  3. Per-achievement unlock rate over time (from metrics.jsonl)

Usage:
  MPLBACKEND=Agg python dreamerv3/plot_training.py \
      --logdir ./logdir/crafter_small_1m \
      --save ./logdir/crafter_small_1m/plots \
      --smooth 50
"""

import argparse
import json
import pathlib
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


BLUE   = '#0022ff'
GREEN  = '#33aa00'

ACHIEVEMENT_COLORS = [
    '#0022ff', '#33aa00', '#ff0011', '#ddaa00', '#cc44dd', '#0088aa',
    '#001177', '#117700', '#990022', '#885500', '#553366', '#006666',
    '#7777cc', '#999999', '#990099', '#888800', '#ff00aa', '#444444',
    '#aaaaaa', '#ff6600', '#00aaff', '#ff00ff',
]


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def records_to_series(records, key):
    steps, vals = [], []
    for r in records:
        if 'step' in r and key in r:
            steps.append(r['step'])
            vals.append(r[key])
    order = np.argsort(steps)
    return np.array(steps)[order], np.array(vals)[order]


def smooth(steps, values, window):
    """Moving average, trimming edge artefacts."""
    if window <= 1 or len(values) <= window:
        return steps, values
    kernel = np.ones(window) / window
    s = np.convolve(values, kernel, mode='same')
    half = window // 2
    return steps[half:-half], s[half:-half]


def fmt_millions(x, pos):
    if x >= 1e6:
        return f'{x/1e6:.1f}M'
    elif x >= 1e3:
        return f'{x/1e3:.0f}K'
    return str(int(x))


def style_ax(ax):
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_millions))
    ax.set_xlabel('Training steps', fontsize=11)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_episode_score(ax, steps, scores, smooth_window):
    ax.scatter(steps, scores, alpha=0.25, s=6, color=BLUE, label='Per episode')
    if smooth_window > 1 and len(steps) > smooth_window:
        sx, sy = smooth(steps, scores, smooth_window)
        ax.plot(sx, sy, color=BLUE, linewidth=1.8, label=f'Smoothed (w={smooth_window})')
    ax.set_ylabel('Score (achievements unlocked)', fontsize=11)
    ax.set_title('Episode Score', fontsize=13)
    ax.legend(fontsize=9, framealpha=0.7)
    style_ax(ax)


def plot_cumulative_reward(ax, steps, scores):
    cumulative = np.cumsum(scores)
    ax.plot(steps, cumulative, color=GREEN, linewidth=1.8)
    ax.fill_between(steps, 0, cumulative, alpha=0.12, color=GREEN)
    ax.set_ylabel('Cumulative reward', fontsize=11)
    ax.set_title('Cumulative Reward', fontsize=13)
    style_ax(ax)


def plot_achievement_rates(ax, records, smooth_window, top_n=16):
    ach_pattern = re.compile(r'^epstats/log/achievement_(.+)/sum$')
    all_keys = set()
    for r in records:
        for k in r:
            if ach_pattern.match(k):
                all_keys.add(k)

    if not all_keys:
        ax.text(0.5, 0.5,
                'No achievement data in metrics.jsonl',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10, color='grey')
        ax.set_title('Achievement Unlock Rate', fontsize=13)
        style_ax(ax)
        return

    # Sort by total unlocks descending, keep top_n
    totals = {}
    for key in all_keys:
        _, vals = records_to_series(records, key)
        totals[key] = vals.sum() if len(vals) else 0
    sorted_keys = sorted(all_keys, key=lambda k: -totals[k])[:top_n]

    for i, key in enumerate(sorted_keys):
        steps, vals = records_to_series(records, key)
        if len(steps) == 0:
            continue
        name = ach_pattern.match(key).group(1).replace('_', ' ')
        color = ACHIEVEMENT_COLORS[i % len(ACHIEVEMENT_COLORS)]
        # faint raw, bold smoothed
        ax.plot(steps, vals, alpha=0.2, linewidth=0.7, color=color)
        if smooth_window > 1 and len(steps) > smooth_window:
            sx, sy = smooth(steps, vals, smooth_window)
            ax.plot(sx, sy, linewidth=1.4, color=color, label=name)
        else:
            ax.lines[-1].set_label(name)

    ax.set_ylabel('Avg unlocks / episode', fontsize=11)
    ax.set_title('Per-Achievement Unlock Rate', fontsize=13)
    ax.legend(fontsize=7, framealpha=0.8, ncol=2, loc='upper left')
    style_ax(ax)


def main():
    parser = argparse.ArgumentParser(description='Plot DreamerV3 training progress')
    parser.add_argument('--logdir', required=True,
                        help='Logdir containing scores.jsonl / metrics.jsonl')
    parser.add_argument('--save', default=None,
                        help='Directory to save plot (default: show interactively)')
    parser.add_argument('--smooth', type=int, default=50,
                        help='Smoothing window size (default: 50)')
    parser.add_argument('--no_achievements', action='store_true',
                        help='Skip the per-achievement panel')
    args = parser.parse_args()

    logdir = pathlib.Path(args.logdir)
    scores_path  = logdir / 'scores.jsonl'
    metrics_path = logdir / 'metrics.jsonl'

    ep_steps, ep_scores = np.array([]), np.array([])
    if scores_path.exists():
        print(f'Loading {scores_path}')
        ep_steps, ep_scores = records_to_series(load_jsonl(scores_path), 'episode/score')
        print(f'  {len(ep_steps)} episodes, score [{ep_scores.min():.2f}, {ep_scores.max():.2f}]')
    else:
        print(f'Warning: {scores_path} not found')

    metric_records = []
    if not args.no_achievements:
        if metrics_path.exists():
            print(f'Loading {metrics_path}')
            metric_records = load_jsonl(metrics_path)
            print(f'  {len(metric_records)} metric entries')
        else:
            print(f'Warning: {metrics_path} not found')

    n_panels = 2 if args.no_achievements else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4.5))
    fig.suptitle(f'Training Progress — {logdir.name}', fontsize=14, fontweight='bold')

    def empty_panel(ax, title):
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=11, color='grey')
        ax.set_title(title, fontsize=13)
        style_ax(ax)

    if len(ep_steps):
        plot_episode_score(axes[0], ep_steps, ep_scores, args.smooth)
        plot_cumulative_reward(axes[1], ep_steps, ep_scores)
    else:
        empty_panel(axes[0], 'Episode Score')
        empty_panel(axes[1], 'Cumulative Reward')

    if not args.no_achievements:
        plot_achievement_rates(axes[2], metric_records, args.smooth)

    plt.tight_layout()

    if args.save:
        save_dir = pathlib.Path(args.save)
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / 'training_progress.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'Saved: {out}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
