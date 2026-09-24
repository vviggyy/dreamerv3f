"""Plot eval-trajectory summary distributions (the eval-side analog of plot_training).

Training progress (plot_training.py) is a "vs training-step" view driven by
scores.jsonl. Eval trajectories are instead N episodes rolled out at a SINGLE
fixed checkpoint, so there is no training-step axis -- the natural analog is a
DISTRIBUTION view across the eval episodes. This reads all_episodes.pkl (written
by eval_trajectory.py) and produces eval_progress.png:

  1. Episode length distribution (steps survived) -- histogram + mean/median
  2. Total reward distribution                    -- histogram + mean/median
  3. Episode length vs total reward               -- scatter
  4. Per-achievement unlock rate across episodes  -- tier-colored bars,
                                                     Crafter score in the title

Usage:
  MPLBACKEND=Agg python dreamerv3/plot_eval.py \
      --data ./logdir/my_run/trajectories \
      --save ./logdir/my_run/plots
"""

import argparse
import pickle
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from run_info import log_run_info
# Reuse the training-plot palette + tech-tree tiers so the two figures match.
from plot_training import ACHIEVEMENT_TIERS, TIER_LABELS, ACHIEVEMENT_COLORS, BLUE, GREEN


def load_episodes(data_path):
    """Load all_episodes.pkl (or a dir containing it). Returns (episodes, meta)."""
    p = pathlib.Path(data_path)
    if p.is_dir():
        p = p / 'all_episodes.pkl'
    if not p.exists():
        raise FileNotFoundError(f'No all_episodes.pkl at {p}')
    with open(p, 'rb') as f:
        d = pickle.load(f)
    if isinstance(d, dict) and 'episodes' in d:
        episodes = d['episodes']
        meta = {k: v for k, v in d.items() if k != 'episodes'}
    else:  # bare list fallback
        episodes = d
        meta = {}
    return episodes, meta


def _style(ax):
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _hist(ax, vals, color, xlabel, title):
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    bins = max(10, min(40, int(np.sqrt(n)) * 2))
    ax.hist(vals, bins=bins, color=color, alpha=0.75, edgecolor='white', linewidth=0.4)
    mean, med = float(np.mean(vals)), float(np.median(vals))
    ax.axvline(mean, color='#ff0011', linewidth=1.6, label=f'mean {mean:.1f}')
    ax.axvline(med, color='#111111', linewidth=1.4, linestyle='--', label=f'median {med:.1f}')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('# episodes', fontsize=11)
    ax.set_title(f'{title}  (n={n})', fontsize=12)
    ax.legend(fontsize=9, frameon=False)
    _style(ax)


def plot_length_hist(ax, lengths):
    _hist(ax, lengths, GREEN, 'Episode length (steps)', 'Episode length distribution')


def plot_reward_hist(ax, rewards):
    _hist(ax, rewards, BLUE, 'Total reward', 'Total reward distribution')


def plot_length_vs_reward(ax, lengths, rewards):
    ax.scatter(lengths, rewards, s=14, alpha=0.5, color='#0088aa', edgecolor='none')
    if len(lengths) > 2:
        r = np.corrcoef(lengths, rewards)[0, 1]
        ax.set_title(f'Length vs reward  (Pearson r={r:.2f})', fontsize=12)
    else:
        ax.set_title('Length vs reward', fontsize=12)
    ax.set_xlabel('Episode length (steps)', fontsize=11)
    ax.set_ylabel('Total reward', fontsize=11)
    _style(ax)


def compute_success_rates(episodes):
    """Fraction of episodes that unlocked each achievement at least once."""
    names = sorted({k for ep in episodes for k in ep.get('final_achievements', {})})
    rates = {}
    for name in names:
        unlocked = [1.0 if ep.get('final_achievements', {}).get(name, 0) > 0 else 0.0
                    for ep in episodes]
        rates[name] = float(np.mean(unlocked)) if unlocked else 0.0
    return rates


def crafter_score(rates):
    """Geometric mean of per-achievement success rates, in percent (Hafner 2021)."""
    if not rates:
        return 0.0
    pcts = np.array(list(rates.values())) * 100.0
    return float(np.exp(np.mean(np.log(1.0 + pcts))) - 1.0)


def plot_achievement_rates(ax, episodes):
    rates = compute_success_rates(episodes)
    if not rates:
        ax.text(0.5, 0.5, 'No achievement data', ha='center', va='center',
                transform=ax.transAxes, color='grey')
        return
    # sort by tier then rate, so related achievements group together
    names = sorted(rates, key=lambda n: (ACHIEVEMENT_TIERS.get(n, 99), rates[n]))
    vals = [rates[n] * 100.0 for n in names]
    tiers = [ACHIEVEMENT_TIERS.get(n, 0) for n in names]
    colors = [ACHIEVEMENT_COLORS[t % len(ACHIEVEMENT_COLORS)] for t in tiers]
    y = np.arange(len(names))
    ax.barh(y, vals, color=colors, alpha=0.85, edgecolor='white', linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Unlock rate across eval episodes (%)', fontsize=11)
    for yi, v in zip(y, vals):
        ax.text(min(v + 1.5, 99), yi, f'{v:.0f}', va='center', fontsize=7, color='#333333')
    cs = crafter_score(rates)
    ax.set_title(f'Achievement unlock rate  (Crafter score {cs:.1f}%)', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='x', alpha=0.3, linewidth=0.5)


def main():
    parser = argparse.ArgumentParser(description='Plot eval-trajectory distributions')
    parser.add_argument('--data', required=True,
                        help='Trajectories dir (or path to all_episodes.pkl)')
    parser.add_argument('--save', default=None,
                        help='Output dir (default: alongside --data)')
    parser.add_argument('--no_achievements', action='store_true',
                        help='Skip the per-achievement panel')
    args = parser.parse_args()

    episodes, meta = load_episodes(args.data)
    if not episodes:
        raise SystemExit('No episodes found.')
    lengths = [ep['length'] for ep in episodes]
    rewards = [ep.get('total_reward', float(np.sum(ep['reward']))) for ep in episodes]

    save_dir = pathlib.Path(args.save) if args.save else pathlib.Path(args.data)
    if save_dir.suffix == '.pkl':
        save_dir = save_dir.parent
    save_dir.mkdir(parents=True, exist_ok=True)

    show_ach = not args.no_achievements and any(ep.get('final_achievements') for ep in episodes)
    if show_ach:
        fig = plt.figure(figsize=(15, 9))
        gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.15], hspace=0.42, wspace=0.28)
        ax_len = fig.add_subplot(gs[0, 0])
        ax_rew = fig.add_subplot(gs[1, 0])
        ax_sc = fig.add_subplot(gs[2, 0])
        ax_ach = fig.add_subplot(gs[:, 1])
        plot_achievement_rates(ax_ach, episodes)
    else:
        fig, (ax_len, ax_rew, ax_sc) = plt.subplots(1, 3, figsize=(16, 4.5))
        fig.subplots_adjust(wspace=0.3)

    plot_length_hist(ax_len, lengths)
    plot_reward_hist(ax_rew, rewards)
    plot_length_vs_reward(ax_sc, lengths, rewards)

    # Title carries the run context (world, area, island/fixed-layout, n episodes).
    bits = [f'{len(episodes)} eval episodes']
    if 'area' in meta:
        area = meta['area']
        bits.append(f'area {area[0]}x{area[1]}' if hasattr(area, '__len__') else f'area {area}')
    if 'world_seed' in meta:
        bits.append(f'world_seed {meta["world_seed"]}')
    if meta.get('island'):
        bits.append('island')
    if meta.get('fixed_layout'):
        bits.append('fixed_layout')
    bits.append(f'len: mean {np.mean(lengths):.0f} / median {np.median(lengths):.0f} '
                f'/ [{min(lengths)}, {max(lengths)}]')
    fig.suptitle('Eval trajectory summary  —  ' + '  |  '.join(bits),
                 fontsize=13, y=0.98)

    out = save_dir / 'eval_progress.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    print(f'Saved {out}')

    log_run_info(save_dir, stage='plot_eval', args=vars(args),
                 outputs=[str(out)],
                 extra={'n_episodes': len(episodes),
                        'length_mean': float(np.mean(lengths)),
                        'length_median': float(np.median(lengths)),
                        'length_min': int(min(lengths)),
                        'length_max': int(max(lengths)),
                        'reward_mean': float(np.mean(rewards))})


if __name__ == '__main__':
    main()
