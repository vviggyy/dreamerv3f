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

from run_info import log_run_info

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
    ax.set_xlabel('Environment steps', fontsize=11)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


LOSS_COLORS = {
    'image': '#0022ff',
    'rew': '#33aa00',
    'con': '#ff0011',
    'dyn': '#cc44dd',
    'rep': '#cc44dd',
    'policy': '#0088aa',
    'value': '#001177',
    'player_pos': '#999999',
}

# Losses that should be drawn with a dashed line (e.g. same color as another)
LOSS_DASHED = {'dyn'}

REWARD_COLORS = {
    'train/rew': ('#33aa00', 'Avg reward'),
    'train/ret': ('#0022ff', 'Return'),
    'train/val': ('#cc44dd', 'Value'),
}


def plot_losses(ax, records, smooth_window):
    """Plot training losses over time."""
    loss_keys = sorted(set(
        k for r in records for k in r
        if k.startswith('train/loss/') and k != 'train/opt/loss'))
    if not loss_keys:
        ax.text(0.5, 0.5, 'No training loss data',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10, color='grey')
        ax.set_title('Training Losses', fontsize=13)
        style_ax(ax)
        return

    for key in loss_keys:
        name = key.split('/')[-1]
        steps, vals = records_to_series(records, key)
        if len(steps) == 0:
            continue
        color = LOSS_COLORS.get(name, '#444444')
        ax.plot(steps, vals, alpha=0.3, linewidth=0.7, color=color)
        if smooth_window > 1 and len(steps) > smooth_window:
            sx, sy = smooth(steps, vals, smooth_window)
            ax.plot(sx, sy, linewidth=1.6, color=color, label=name)
        else:
            ax.lines[-1].set_label(name)

    ax.set_yscale('log')
    ax.set_ylabel('Loss (log scale)', fontsize=11)
    ax.set_title('Training Losses', fontsize=13)
    ax.legend(fontsize=7, framealpha=0.8, ncol=2, loc='upper right')
    style_ax(ax)


def plot_reward_value(ax, records, smooth_window):
    """Plot training reward, return, and value estimates over time."""
    has_data = False
    for key, (color, label) in REWARD_COLORS.items():
        steps, vals = records_to_series(records, key)
        if len(steps) == 0:
            continue
        has_data = True
        ax.plot(steps, vals, alpha=0.3, linewidth=0.7, color=color)
        if smooth_window > 1 and len(steps) > smooth_window:
            sx, sy = smooth(steps, vals, smooth_window)
            ax.plot(sx, sy, linewidth=1.6, color=color, label=label)
        else:
            ax.lines[-1].set_label(label)

    if not has_data:
        ax.text(0.5, 0.5, 'No reward/value data',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10, color='grey')
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('Reward & Value Estimates', fontsize=13)
    ax.legend(fontsize=9, framealpha=0.7)
    style_ax(ax)


def plot_episode_score(ax, steps, scores, smooth_window):
    ax.scatter(steps, scores, alpha=0.25, s=6, color=BLUE, label='Per episode')
    if smooth_window > 1 and len(steps) > smooth_window:
        sx, sy = smooth(steps, scores, smooth_window)
        ax.plot(sx, sy, color=BLUE, linewidth=1.8, label=f'Smoothed (w={smooth_window})')
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_ylabel('Score (achievements unlocked)', fontsize=11)
    ax.set_title('Episode Score', fontsize=13)
    ax.legend(fontsize=9, framealpha=0.7)
    style_ax(ax)


def plot_cumulative_reward(ax, steps, scores):
    cumulative = np.cumsum(scores)
    ax.plot(steps, cumulative, color=GREEN, linewidth=1.8)
    ax.fill_between(steps, 0, cumulative, alpha=0.12, color=GREEN)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_ylabel('Cumulative reward', fontsize=11)
    ax.set_title('Cumulative Reward', fontsize=13)
    style_ax(ax)


def plot_crafter_score(ax, records, smooth_window):
    """Compute and plot the Crafter score (geometric mean of success rates).

    The Crafter score = exp(mean(log(rate_i + eps))) across all 22 achievements,
    where rate_i is the success rate of achievement i over a rolling window.
    """
    ach_pattern = re.compile(r'^episode/achievement_(.+)$')
    ach_keys = sorted(set(
        k for r in records for k in r if ach_pattern.match(k)))

    if not ach_keys:
        ax.text(0.5, 0.5,
                'No per-episode achievement data in scores.jsonl\n'
                '(requires training with updated logger)',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=9, color='grey')
        ax.set_title('Crafter Score', fontsize=13)
        style_ax(ax)
        return

    # Build aligned arrays: steps and per-achievement success (1/0)
    steps = np.array([r['step'] for r in records if 'step' in r and ach_keys[0] in r])
    if len(steps) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=11, color='grey')
        ax.set_title('Crafter Score', fontsize=13)
        style_ax(ax)
        return

    successes = np.zeros((len(steps), len(ach_keys)))
    for j, key in enumerate(ach_keys):
        for i, r in enumerate(
                [r for r in records if 'step' in r and ach_keys[0] in r]):
            successes[i, j] = r.get(key, 0.0)

    # Compute rolling Crafter score
    eps = 1e-6
    window = max(smooth_window, 10)
    if len(steps) < window:
        window = len(steps)
    crafter_scores = np.full(len(steps), np.nan)
    for i in range(window - 1, len(steps)):
        rates = successes[i - window + 1:i + 1].mean(axis=0)  # per-achievement
        crafter_scores[i] = np.exp(np.mean(np.log(rates + eps))) * 100

    valid = ~np.isnan(crafter_scores)
    ax.plot(steps[valid], crafter_scores[valid], color='#cc3300', linewidth=1.8,
            label=f'Crafter score (w={window})')
    ax.set_ylabel('Crafter Score (%)', fontsize=11)
    ax.set_ylim(bottom=0)
    ax.set_title('Crafter Score', fontsize=13)
    ax.legend(fontsize=9, framealpha=0.7)
    style_ax(ax)

    # Print final score
    final = crafter_scores[valid][-1] if valid.any() else 0
    n_ach = len(ach_keys)
    print(f'  Crafter score (last window): {final:.1f}% ({n_ach} achievements)')


def plot_achievement_rates(ax, records, smooth_window):
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

    # Sort by total unlocks descending
    totals = {}
    for key in all_keys:
        _, vals = records_to_series(records, key)
        totals[key] = vals.sum() if len(vals) else 0
    sorted_keys = sorted(all_keys, key=lambda k: -totals[k])

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
    parser.add_argument('--no_losses', action='store_true',
                        help='Skip the loss and reward/value panels')
    args = parser.parse_args()

    logdir = pathlib.Path(args.logdir)
    scores_path  = logdir / 'scores.jsonl'
    metrics_path = logdir / 'metrics.jsonl'

    score_records = []
    ep_steps, ep_scores = np.array([]), np.array([])
    if scores_path.exists():
        print(f'Loading {scores_path}')
        score_records = load_jsonl(scores_path)
        ep_steps, ep_scores = records_to_series(score_records, 'episode/score')
        if len(ep_steps):
            print(f'  {len(ep_steps)} episodes, score [{ep_scores.min():.2f}, {ep_scores.max():.2f}]')
    else:
        print(f'Warning: {scores_path} not found')

    metric_records = []
    if metrics_path.exists():
        print(f'Loading {metrics_path}')
        metric_records = load_jsonl(metrics_path)
        print(f'  {len(metric_records)} metric entries')
    else:
        print(f'Warning: {metrics_path} not found')

    # Check if scores.jsonl has per-episode achievement data
    has_ach_scores = any(
        k.startswith('episode/achievement_') for r in score_records for k in r)
    # Check if metrics.jsonl has training loss data
    has_train = any('train/loss/image' in r for r in metric_records)

    # Row 1: episode score, crafter score, per-achievement unlock rate
    # Row 2: cumulative reward, training losses, reward & value estimates
    top_panels = ['score', 'crafter_score', 'achievements']
    bot_panels = ['cumulative', 'losses', 'reward_value']

    n_cols = 3
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    fig.suptitle(f'Training Progress — {logdir.name}', fontsize=14, fontweight='bold')

    def empty_panel(ax, title):
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=11, color='grey')
        ax.set_title(title, fontsize=13)
        style_ax(ax)

    # Use smoothing window scaled for metric records (fewer data points)
    metric_smooth = max(1, args.smooth // 10)

    def render_panel(ax, panel_type):
        if panel_type == 'score':
            if len(ep_steps):
                plot_episode_score(ax, ep_steps, ep_scores, args.smooth)
            else:
                empty_panel(ax, 'Episode Score')
        elif panel_type == 'cumulative':
            if len(ep_steps):
                plot_cumulative_reward(ax, ep_steps, ep_scores)
            else:
                empty_panel(ax, 'Cumulative Reward')
        elif panel_type == 'losses':
            plot_losses(ax, metric_records, metric_smooth)
        elif panel_type == 'reward_value':
            plot_reward_value(ax, metric_records, metric_smooth)
        elif panel_type == 'crafter_score':
            plot_crafter_score(ax, score_records, args.smooth)
        elif panel_type == 'achievements':
            plot_achievement_rates(ax, metric_records, args.smooth)

    for i, panel_type in enumerate(top_panels):
        render_panel(axes[0, i], panel_type)
    for i, panel_type in enumerate(bot_panels):
        render_panel(axes[1, i], panel_type)

    plt.tight_layout()

    if args.save:
        save_dir = pathlib.Path(args.save)
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / 'training_progress.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'Saved: {out}')
        log_run_info(save_dir, 'plot_training', vars(args))
    else:
        plt.show()


if __name__ == '__main__':
    main()
