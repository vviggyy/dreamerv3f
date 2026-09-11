"""Vital dynamics: observational analysis of interoceptive vitals vs behavior.

Reads real eval_trajectory rollouts (which now log `vital_{health,food,drink,
energy}` per step, 0-9) and relates those vitals to the policy's actions and to
achievement-unlock events over time. This is the OBSERVATIONAL complement to
`state_probe.py` — state_probe intervenes on the vitals of a fixed scene and
measures the counterfactual policy shift; here we simply watch what the agent
actually does as its real vitals rise and fall.

Three products:
  1. Per-episode time-series overlay (`vitals_episode_XXX.png`): the four vital
     curves over the episode, vertical markers at each achievement unlock, and
     a bottom lane marking vital-relevant actions ("do" = interact, "sleep").
  2. Action-rate vs vital level (`action_rate_vs_vital.png`): P(action | vital
     == level) pooled across all steps/episodes — the observational analog of
     state_probe's value/action sweep.
  3. Event-triggered vital averages (`event_triggered_vitals.png`): mean vital
     trajectory in a window around each achievement-unlock event (e.g. drink
     jumps at collect_drink, energy at wake_up).

Usage:
    MPLBACKEND=Agg python dreamerv3/vital_dynamics.py \
        --data ./logdir/my_run/trajectories \
        --save ./logdir/my_run/vital_dynamics \
        --max_episode_plots 10

NOTE: crafter has no separate eat/drink actions — "do" (5) is the single
interact used to eat a cow, drink water, attack, or harvest, so the *semantic*
food/drink/attack events are read from achievement unlocks, while "do"/"sleep"
are the only vital-relevant raw actions. See VITAL_ACTIONS / VITAL_EVENTS below.
"""

import argparse
import pickle
from pathlib import Path
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from run_info import log_run_info

VITALS = ['health', 'food', 'drink', 'energy']
VITAL_COLORS = {
    'health': '#d62728', 'food': '#ff7f0e',
    'drink': '#1f77b4', 'energy': '#2ca02c'}
VITAL_MAX = 9  # crafter vitals are ints in [0, 9]

# crafter action index -> name (crafter.constants.actions)
ACTION_NAMES = [
    'noop', 'move_left', 'move_right', 'move_up', 'move_down', 'do', 'sleep',
    'place_stone', 'place_table', 'place_furnace', 'place_plant',
    'make_wood_pickaxe', 'make_stone_pickaxe', 'make_iron_pickaxe',
    'make_wood_sword', 'make_stone_sword', 'make_iron_sword']

# The only raw actions that touch vitals: interact ("do") and rest ("sleep").
VITAL_ACTIONS = {'do': 5, 'sleep': 6}
# Coarse behavior classes for the action-rate-vs-vital panel.
ACTION_CLASSES = {
    'noop': lambda a: a == 0,
    'move': lambda a: a in (1, 2, 3, 4),
    'do': lambda a: a == 5,
    'sleep': lambda a: a == 6,
    'craft/place': lambda a: a >= 7,
}
ACTION_CLASS_COLORS = {
    'noop': '#7f7f7f', 'move': '#9467bd', 'do': '#d62728',
    'sleep': '#1f77b4', 'craft/place': '#2ca02c'}

# Achievement unlocks that reflect a vital change (used for event-triggered avg
# and per-episode markers). collect_drink->drink, eat_*->food, wake_up->energy,
# defeat_*->health (took/dealt damage).
VITAL_EVENTS = [
    'eat_cow', 'eat_plant', 'collect_drink', 'wake_up',
    'defeat_zombie', 'defeat_skeleton']


def load_episodes(data_path):
    """Load eval_trajectory output: either all_episodes.pkl or a dir of
    episode_*.pkl. Returns (episodes_list, metadata_dict)."""
    p = Path(data_path)
    if p.is_file():
        with open(p, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, dict) and 'episodes' in data:
            return data['episodes'], {k: v for k, v in data.items() if k != 'episodes'}
        return [data], {}
    # directory
    all_file = p / 'all_episodes.pkl'
    if all_file.exists():
        with open(all_file, 'rb') as f:
            data = pickle.load(f)
        return data['episodes'], {k: v for k, v in data.items() if k != 'episodes'}
    eps = []
    for ep_file in sorted(p.glob('episode_*.pkl')):
        with open(ep_file, 'rb') as f:
            eps.append(pickle.load(f))
    if not eps:
        raise FileNotFoundError(f'No all_episodes.pkl or episode_*.pkl under {p}')
    return eps, {}


def get_vitals(ep):
    """Return {vital: np.array(T)} for the episode, or None if not recorded."""
    out = {}
    for v in VITALS:
        key = f'vital_{v}'
        if key not in ep or len(ep[key]) == 0:
            return None
        out[v] = np.asarray(ep[key])
    return out


def get_actions(ep):
    return np.asarray(ep.get('action', [])).astype(int).ravel()


def get_unlocks(ep):
    """From the per-step cumulative achievement dicts, return {achievement:
    [first-unlock step, ...]} (each 0->1 or count increase is one event)."""
    ach = ep.get('achievements')
    if not ach:
        return {}
    unlocks = defaultdict(list)
    prev = {}
    for t, counts in enumerate(ach):
        for name, c in counts.items():
            if c > prev.get(name, 0):
                unlocks[name].append(t)
            prev[name] = c
    return dict(unlocks)


# --------------------------------------------------------------------------- #
# 1. Per-episode time-series overlay
# --------------------------------------------------------------------------- #
def plot_episode(ep, save_path, ep_idx):
    vitals = get_vitals(ep)
    if vitals is None:
        return False
    actions = get_actions(ep)
    unlocks = get_unlocks(ep)
    T = len(next(iter(vitals.values())))
    steps = np.arange(T)

    fig, ax = plt.subplots(figsize=(max(8, T / 40), 4.5))
    for v in VITALS:
        ax.plot(steps, vitals[v], color=VITAL_COLORS[v], lw=1.6, label=v)
    ax.set_ylim(-0.5, VITAL_MAX + 0.5)
    ax.set_ylabel('vital level (0-9)')
    ax.set_xlabel('step')
    ax.set_xlim(0, max(1, T - 1))

    # achievement unlock markers (vertical lines + rotated labels)
    for name, when in unlocks.items():
        for t in when:
            is_vital = name in VITAL_EVENTS
            ax.axvline(t, color='k' if is_vital else '0.7',
                       ls='--' if is_vital else ':', lw=0.9, alpha=0.7, zorder=0)
            ax.annotate(name, xy=(t, VITAL_MAX + 0.4), rotation=90, fontsize=6,
                        va='bottom', ha='center',
                        color='k' if is_vital else '0.5')

    # vital-relevant action lane below the axis
    ymarks = {'do': -0.25, 'sleep': -0.42}
    for aname, aidx in VITAL_ACTIONS.items():
        ts = steps[actions == aidx] if len(actions) == T else []
        if len(ts):
            ax.scatter(ts, np.full(len(ts), ymarks[aname]), marker='|',
                       s=40, color=ACTION_CLASS_COLORS.get(aname, 'k'),
                       label=f'action={aname}')

    ax.legend(loc='lower right', fontsize=7, ncol=3, framealpha=0.9)
    ax.set_title(f'Episode {ep.get("episode", ep_idx)} — vitals, achievements, actions',
                 pad=22)
    fig.tight_layout()
    out = Path(save_path) / f'vitals_episode_{ep_idx:03d}.png'
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
# 2. Action-rate vs vital level (pooled across all steps)
# --------------------------------------------------------------------------- #
def compute_action_rate_vs_vital(episodes):
    """For each vital level 0-9 and each action class, P(action class | vital).
    Returns {vital: {'levels': arr, 'counts': arr(n_levels),
                     'rates': {cls: arr(n_levels)}}}."""
    levels = np.arange(VITAL_MAX + 1)
    # accumulate: per vital, per level -> action-class counts + total
    acc = {v: {'total': np.zeros(len(levels)),
               'cls': {c: np.zeros(len(levels)) for c in ACTION_CLASSES}}
           for v in VITALS}
    for ep in episodes:
        vitals = get_vitals(ep)
        actions = get_actions(ep)
        if vitals is None or len(actions) == 0:
            continue
        T = min(len(actions), len(vitals['health']))
        for v in VITALS:
            vv = vitals[v][:T]
            a = actions[:T]
            for lvl in levels:
                mask = vv == lvl
                n = int(mask.sum())
                if n == 0:
                    continue
                acc[v]['total'][lvl] += n
                for cname, fn in ACTION_CLASSES.items():
                    acc[v]['cls'][cname][lvl] += int(np.array(
                        [fn(int(x)) for x in a[mask]]).sum())
    out = {}
    for v in VITALS:
        tot = acc[v]['total']
        with np.errstate(invalid='ignore', divide='ignore'):
            rates = {c: np.where(tot > 0, acc[v]['cls'][c] / tot, np.nan)
                     for c in ACTION_CLASSES}
        out[v] = {'levels': levels, 'counts': tot, 'rates': rates}
    return out


def plot_action_rate_vs_vital(stats, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, v in zip(axes.ravel(), VITALS):
        s = stats[v]
        for cname in ACTION_CLASSES:
            ax.plot(s['levels'], s['rates'][cname], marker='o', ms=3,
                    color=ACTION_CLASS_COLORS[cname], label=cname)
        ax.set_title(f'{v}  (color = {VITAL_COLORS[v]})', color=VITAL_COLORS[v])
        ax.set_xlabel('vital level (0-9)')
        ax.set_ylabel('P(action class | vital level)')
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        # annotate sample size sparsely
        tot = s['counts']
        for lvl in s['levels']:
            if tot[lvl] > 0:
                ax.annotate(f'{int(tot[lvl])}', xy=(lvl, 1.0), fontsize=5,
                            ha='center', va='top', color='0.5')
    axes.ravel()[0].legend(loc='center left', fontsize=8)
    fig.suptitle('Action-class probability vs vital level (n above each level)',
                 fontsize=12)
    fig.tight_layout()
    out = Path(save_path) / 'action_rate_vs_vital.png'
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# 3. Event-triggered vital averages
# --------------------------------------------------------------------------- #
def compute_event_triggered(episodes, window=15):
    """For each vital event, gather vital windows [-W, +W] around each unlock.
    Returns {event: {'offsets': arr, vital: {'mean': arr, 'sem': arr, 'n': int}}}."""
    offsets = np.arange(-window, window + 1)
    # event -> vital -> list of length-(2W+1) arrays (nan-padded at edges)
    buf = {e: {v: [] for v in VITALS} for e in VITAL_EVENTS}
    for ep in episodes:
        vitals = get_vitals(ep)
        if vitals is None:
            continue
        T = len(vitals['health'])
        unlocks = get_unlocks(ep)
        for e in VITAL_EVENTS:
            for t0 in unlocks.get(e, []):
                for v in VITALS:
                    seg = np.full(len(offsets), np.nan)
                    for i, off in enumerate(offsets):
                        t = t0 + off
                        if 0 <= t < T:
                            seg[i] = vitals[v][t]
                    buf[e][v].append(seg)
    out = {}
    for e in VITAL_EVENTS:
        n_events = len(buf[e][VITALS[0]])
        if n_events == 0:
            continue
        entry = {'offsets': offsets, 'n_events': n_events}
        for v in VITALS:
            arr = np.vstack(buf[e][v])  # (n_events, 2W+1)
            with np.errstate(invalid='ignore'):
                mean = np.nanmean(arr, axis=0)
                cnt = np.sum(~np.isnan(arr), axis=0)
                sd = np.nanstd(arr, axis=0)
                sem = np.where(cnt > 0, sd / np.sqrt(np.maximum(cnt, 1)), np.nan)
            entry[v] = {'mean': mean, 'sem': sem}
        out[e] = entry
    return out


def plot_event_triggered(evstats, save_path):
    events = [e for e in VITAL_EVENTS if e in evstats]
    if not events:
        return None
    ncol = min(3, len(events))
    nrow = int(np.ceil(len(events) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3.4 * nrow),
                             squeeze=False)
    for ax, e in zip(axes.ravel(), events):
        s = evstats[e]
        off = s['offsets']
        for v in VITALS:
            m, sem = s[v]['mean'], s[v]['sem']
            ax.plot(off, m, color=VITAL_COLORS[v], lw=1.6, label=v)
            ax.fill_between(off, m - sem, m + sem, color=VITAL_COLORS[v], alpha=0.18)
        ax.axvline(0, color='k', ls='--', lw=0.9)
        ax.set_title(f'{e}  (n={s["n_events"]})', fontsize=10)
        ax.set_xlabel('steps from event')
        ax.set_ylabel('vital level')
        ax.set_ylim(-0.5, VITAL_MAX + 0.5)
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(events):]:
        ax.axis('off')
    axes.ravel()[0].legend(loc='best', fontsize=8)
    fig.suptitle('Event-triggered vital averages (±SEM)', fontsize=12)
    fig.tight_layout()
    out = Path(save_path) / 'event_triggered_vitals.png'
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(
        description='Observational analysis of vitals vs policy actions/events')
    parser.add_argument('--data', type=str, required=True,
                        help='eval_trajectory output (all_episodes.pkl or dir)')
    parser.add_argument('--save', type=str, default=None,
                        help='output dir (default: <data>/vital_dynamics)')
    parser.add_argument('--max_episode_plots', type=int, default=10,
                        help='# per-episode time-series plots (0=skip)')
    parser.add_argument('--event_window', type=int, default=15,
                        help='half-window (steps) for event-triggered averages')
    parser.add_argument('--no_agg', action='store_true',
                        help='skip aggregate stats (per-episode plots only)')
    args = parser.parse_args()

    data_path = Path(args.data)
    save_dir = Path(args.save) if args.save else (
        (data_path if data_path.is_dir() else data_path.parent) / 'vital_dynamics')
    save_dir.mkdir(parents=True, exist_ok=True)

    episodes, meta = load_episodes(args.data)
    have_vitals = sum(1 for ep in episodes if get_vitals(ep) is not None)
    print(f'Loaded {len(episodes)} episodes ({have_vitals} with vitals logged)')
    if have_vitals == 0:
        print('ERROR: no episode has vital_* fields. Regenerate trajectories with '
              'the updated crafter env + eval_trajectory (vitals now logged).')
        return

    outputs = []
    # 1. per-episode plots
    n_plotted = 0
    for i, ep in enumerate(episodes):
        if args.max_episode_plots and n_plotted >= args.max_episode_plots:
            break
        if plot_episode(ep, save_dir, i):
            n_plotted += 1
    print(f'Wrote {n_plotted} per-episode plots')
    if n_plotted:
        outputs.append(str(save_dir / 'vitals_episode_*.png'))

    results = {'metadata': meta, 'n_episodes': len(episodes)}
    if not args.no_agg:
        # 2. action rate vs vital
        ar_stats = compute_action_rate_vs_vital(episodes)
        out2 = plot_action_rate_vs_vital(ar_stats, save_dir)
        results['action_rate_vs_vital'] = ar_stats
        outputs.append(str(out2))
        print(f'Wrote {out2}')
        # 3. event-triggered vitals
        ev_stats = compute_event_triggered(episodes, window=args.event_window)
        out3 = plot_event_triggered(ev_stats, save_dir)
        results['event_triggered'] = ev_stats
        if out3:
            outputs.append(str(out3))
            print(f'Wrote {out3}')
        else:
            print('No vital events found; skipping event-triggered plot')

    res_file = save_dir / 'vital_dynamics_results.pkl'
    with open(res_file, 'wb') as f:
        pickle.dump(results, f)
    outputs.append(str(res_file))
    print(f'Saved results to {res_file}')

    log_run_info(save_dir, 'vital_dynamics', vars(args), outputs,
                 extra={'n_episodes': len(episodes), 'n_with_vitals': have_vitals})


if __name__ == '__main__':
    main()
