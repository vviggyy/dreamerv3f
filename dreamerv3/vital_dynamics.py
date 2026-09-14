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

# Health-drop / death cause labels, inferred from crafter's damage model
# (objects.py): a necessity at 0 (food/drink/energy) causes slow -1 degen via
# _degen_or_regen_health; moving onto lava is instant nonzero->0; a drop while
# all necessities are >0 must be mob combat (zombie melee / skeleton arrow).
HURT_CAUSE_COLORS = {
    'hunger': '#8c564b',      # food == 0
    'thirst': '#17becf',      # drink == 0
    'exhaustion': '#bcbd22',  # energy == 0
    'combat': '#e377c2',      # mob attack (necessities OK)
    'lava': '#ff0000',        # instant nonzero->0 on a move step
    'unknown': '#555555',
}


def classify_health_drops(vitals, actions=None):
    """Every step where health decreased -> (t, delta, cause).

    cause in {hunger, thirst, exhaustion, combat, lava}; multiple simultaneously
    lacking necessities are joined (e.g. 'hunger+thirst'). Heuristic from crafter's
    damage model — see HURT_CAUSE_COLORS. `actions` (per-step action ids) lets us
    flag lava (instant nonzero->0 on a move) vs a same-step combat kill."""
    h = np.asarray(vitals['health']).astype(int)
    food = np.asarray(vitals['food']).astype(int)
    drink = np.asarray(vitals['drink']).astype(int)
    energy = np.asarray(vitals['energy']).astype(int)
    drops = []
    for t in range(1, len(h)):
        d = int(h[t]) - int(h[t - 1])
        if d >= 0:
            continue
        lacking = []
        if food[t - 1] == 0 or food[t] == 0:
            lacking.append('hunger')
        if drink[t - 1] == 0 or drink[t] == 0:
            lacking.append('thirst')
        if energy[t - 1] == 0 or energy[t] == 0:
            lacking.append('exhaustion')
        instant_to_zero = (h[t] == 0 and h[t - 1] >= 2)  # lost >=2 to 0 in one step
        is_move = (actions is not None and t < len(actions)
                   and 1 <= int(actions[t]) <= 4)
        if not lacking and instant_to_zero and (is_move or actions is None):
            cause = 'lava'
        elif lacking:
            cause = '+'.join(lacking)
        else:
            cause = 'combat'
        drops.append((t, d, cause))
    return drops


def cause_of_death(vitals, actions=None):
    """Death-cause label, or 'survived (timeout)' if the episode ended with
    health > 0 (i.e. it hit the step limit rather than dying)."""
    h = np.asarray(vitals['health']).astype(int)
    if len(h) == 0:
        return 'unknown'
    if int(h[-1]) > 0:
        return 'survived (timeout)'
    drops = classify_health_drops(vitals, actions)
    zero_drops = [dc for dc in drops if int(h[dc[0]]) == 0]
    if zero_drops:
        return zero_drops[-1][2]
    return 'unknown'


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
# Cache the rendered world per env_seed — all episodes in a fixed_seed run share
# the same world, so we only pay the (slow) crafter render once.
_WORLD_CACHE = {}


def _get_world_img(meta):
    """Return (world_img, env_seed, tile_size) for this run's world, cached by
    env_seed. Returns (None, None, None) if crafter/render is unavailable."""
    key = (meta or {}).get('env_seed')
    if key in _WORLD_CACHE:
        return _WORLD_CACHE[key]
    try:
        from plot_trajectories import _render_crafter_world
        result = _render_crafter_world(meta or {}, tile_size=8)
    except Exception as e:  # crafter missing / render failure
        print(f'  (world map unavailable: {e})')
        result = (None, None, None)
    _WORLD_CACHE[key] = result
    return result


def _draw_episode_map(ax, ep, meta, drops):
    """Draw the episode's trajectory on the crafter world map (path colored by
    time; start/end + hurt locations marked). Returns True if drawn. NOTE: the
    world image is stock worldgen from env_seed and is NOT fixed_layout-aware, so
    on fixed_layout runs the terrain shown is the natural world, not what the
    agent saw (positions are still correct)."""
    pos = np.asarray(ep.get('player_pos'))
    if pos is None or pos.ndim != 2 or pos.shape[0] == 0:
        return False
    world_img, env_seed, ts = _get_world_img(meta)
    if world_img is None:
        return False
    ax.imshow(world_img)
    px = pos[:, 0] * ts + ts // 2
    py = world_img.shape[0] - (pos[:, 1] * ts + ts // 2)
    ax.plot(px, py, '-', color='white', lw=0.6, alpha=0.45, zorder=2)
    ax.scatter(px, py, c=np.arange(len(px)), cmap='viridis', s=7, zorder=3)
    ax.plot(px[0], py[0], 'o', color='lime', ms=8, mec='k', zorder=5, label='start')
    ax.plot(px[-1], py[-1], 'X', color='red', ms=9, mec='k', zorder=5, label='end')
    for (t, d, cause) in drops:                      # mark where it took damage
        if t < len(px):
            col = HURT_CAUSE_COLORS.get(cause.split('+')[0], 'k')
            ax.plot(px[t], py[t], marker='v', color=col, ms=6, mec='k',
                    mew=0.4, zorder=4)
    ax.set_title(f'trajectory (seed={env_seed})', fontsize=9)
    ax.axis('off')
    ax.legend(loc='upper right', fontsize=6, framealpha=0.85)
    return True


def plot_episode(ep, save_path, ep_idx, meta=None):
    vitals = get_vitals(ep)
    if vitals is None:
        return False
    actions = get_actions(ep)
    unlocks = get_unlocks(ep)
    T = len(next(iter(vitals.values())))
    steps = np.arange(T)
    drops = classify_health_drops(vitals, actions)
    death = cause_of_death(vitals, actions)

    # Figure: wide time-series (left) + square world map (right).
    ts_w = max(8, T / 40)
    fig = plt.figure(figsize=(ts_w + 5.2, 5.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[ts_w, 5.0], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    ax_map = fig.add_subplot(gs[0, 1])

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

    # cause-of-hurt markers on the health curve (one legend entry per cause)
    hy = np.asarray(vitals['health'])
    seen = set()
    for (t, d, cause) in drops:
        col = HURT_CAUSE_COLORS.get(cause.split('+')[0], 'k')
        lbl = f'hurt: {cause}' if cause not in seen else None
        seen.add(cause)
        ax.scatter([t], [hy[t]], marker='v', color=col, s=48, edgecolors='k',
                   linewidths=0.5, zorder=6, label=lbl)

    # vital-relevant action lane below the axis
    ymarks = {'do': -0.25, 'sleep': -0.42}
    for aname, aidx in VITAL_ACTIONS.items():
        tsx = steps[actions == aidx] if len(actions) == T else []
        if len(tsx):
            ax.scatter(tsx, np.full(len(tsx), ymarks[aname]), marker='|',
                       s=40, color=ACTION_CLASS_COLORS.get(aname, 'k'),
                       label=f'action={aname}')

    ax.legend(loc='lower right', fontsize=6.5, ncol=3, framealpha=0.9)

    if not _draw_episode_map(ax_map, ep, meta, drops):
        ax_map.axis('off')
        ax_map.text(0.5, 0.5, 'world map\nunavailable', ha='center', va='center',
                    fontsize=10, color='grey', transform=ax_map.transAxes)

    fig.suptitle(
        f'Episode {ep.get("episode", ep_idx)}  ·  length={T}  ·  outcome: {death}',
        fontsize=12, fontweight='bold', y=1.04)
    out = Path(save_path) / f'vitals_episode_{ep_idx:03d}.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
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


# --------------------------------------------------------------------------- #
# 4. Averaged vitals over time (across episodes; clipped to min episode length)
# --------------------------------------------------------------------------- #
def compute_averaged_vitals(episodes, clip_pct=0):
    """Mean/std of each vital at each step across episodes, clipped to a common
    length so every step averages the SAME episode set (no survivorship bias).

    Episodes vary in length, so we clip to `L`: the minimum episode length
    (clip_pct=0, the default — faithful but sensitive to a single short episode),
    or the clip_pct-th percentile length (e.g. clip_pct=5 ignores the shortest
    5% of episodes so one early death doesn't collapse the window). Only episodes
    with length >= L contribute. Returns (stats, L)."""
    per_ep = {v: [] for v in VITALS}
    lengths = []
    for ep in episodes:
        vit = get_vitals(ep)
        if vit is None:
            continue
        lengths.append(len(vit['health']))
        for v in VITALS:
            per_ep[v].append(np.asarray(vit[v], float))
    if not lengths:
        return None, 0
    lengths = np.asarray(lengths)
    L = int(np.percentile(lengths, clip_pct)) if clip_pct > 0 else int(lengths.min())
    L = max(1, L)
    keep = lengths >= L
    stats = {}
    for v in VITALS:
        arr = np.vstack([a[:L] for a, k in zip(per_ep[v], keep) if k])  # (n_kept, L)
        stats[v] = {'mean': arr.mean(0), 'std': arr.std(0), 'n': arr.shape[0]}
    return stats, L


def plot_averaged_vitals(groups, save_path, fname='averaged_vitals.png'):
    """One subplot per vital; overlays each group (label, stats, L). A single
    group draws a ±1 std band; multiple groups (e.g. training phases) overlay as
    lines distinguished by linestyle (color still encodes the vital)."""
    groups = [g for g in groups if g[1] is not None]
    if not groups:
        return None
    styles = ['-', '--', ':', '-.']
    multi = len(groups) > 1
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, v in zip(axes.ravel(), VITALS):
        for gi, (label, stats, L) in enumerate(groups):
            s = stats[v]
            x = np.arange(len(s['mean']))
            ls = styles[gi % len(styles)]
            lbl = label if multi else f'{v} (n={s["n"]})'
            ax.plot(x, s['mean'], color=VITAL_COLORS[v], ls=ls, lw=1.8, label=lbl)
            if not multi:
                ax.fill_between(x, s['mean'] - s['std'], s['mean'] + s['std'],
                                color=VITAL_COLORS[v], alpha=0.18)
        ax.set_title(v, color=VITAL_COLORS[v])
        ax.set_ylim(-0.5, VITAL_MAX + 0.5)
        ax.set_xlabel('step (clipped to common episode length)')
        ax.set_ylabel('vital level (0-9)')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, framealpha=0.85)
    sub = ('mean ±1 std across episodes' if not multi
           else 'mean across episodes, by phase (linestyle) — color = vital')
    fig.suptitle(f'Averaged vitals over time ({sub})', fontsize=12)
    fig.tight_layout()
    out = Path(save_path) / fname
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# 5. Empirical p(achievement unlock within horizon | vital level)
# --------------------------------------------------------------------------- #
def compute_achievement_rate_vs_vital(episodes, horizon=15):
    """For each vital and level L, P(achievement a unlocks within the next
    `horizon` steps | vital == L), pooled over all steps/episodes. This is the
    OBSERVATIONAL conditional (correlational) — the policy net has no achievement
    head, so this comes from behavior, not a forward pass. Returns {vital:
    {'levels', 'ach', 'P'(n_ach,n_levels), 'counts'(n_levels)}}."""
    levels = np.arange(VITAL_MAX + 1)
    ach_names = sorted({a for ep in episodes for a in get_unlocks(ep)})
    idx = {a: i for i, a in enumerate(ach_names)}
    acc = {v: {'hit': np.zeros((len(ach_names), len(levels))),
               'tot': np.zeros(len(levels))} for v in VITALS}
    for ep in episodes:
        vit = get_vitals(ep)
        if vit is None or not ach_names:
            continue
        T = len(vit['health'])
        unlocks = get_unlocks(ep)
        # future[a, t] = achievement a unlocks at some t0 in [t, t+horizon]
        future = np.zeros((len(ach_names), T), dtype=bool)
        for a, tlist in unlocks.items():
            ai = idx.get(a)
            if ai is None:
                continue
            for t0 in tlist:
                lo = max(0, t0 - horizon)
                future[ai, lo:t0 + 1] = True
        for v in VITALS:
            vv = np.asarray(vit[v])[:T]
            for lvl in levels:
                mask = vv == lvl
                n = int(mask.sum())
                if n == 0:
                    continue
                acc[v]['tot'][lvl] += n
                acc[v]['hit'][:, lvl] += future[:, mask].sum(axis=1)
    out = {}
    for v in VITALS:
        tot = acc[v]['tot']
        with np.errstate(invalid='ignore', divide='ignore'):
            P = np.where(tot > 0, acc[v]['hit'] / tot, np.nan)
        out[v] = {'levels': levels, 'ach': ach_names, 'P': P, 'counts': tot}
    return out


def plot_achievement_rate_vs_vital(stats, save_path, horizon):
    """Heatmap per vital: rows = achievements, cols = vital level, color =
    P(unlock within horizon | vital). Shared color scale across vitals."""
    achs = stats[VITALS[0]]['ach']
    if not achs:
        return None
    vmax = max([np.nanmax(stats[v]['P']) for v in VITALS
                if np.isfinite(stats[v]['P']).any()] + [1e-9])
    fig, axes = plt.subplots(1, len(VITALS), sharey=True,
                             figsize=(3.6 * len(VITALS), max(4.0, 0.32 * len(achs))))
    axes = np.atleast_1d(axes)
    im = None
    for ax, v in zip(axes, VITALS):
        s = stats[v]
        im = ax.imshow(s['P'], aspect='auto', cmap='magma', vmin=0, vmax=vmax)
        ax.set_xticks(range(len(s['levels'])))
        ax.set_xticklabels(s['levels'], fontsize=7)
        ax.set_xlabel('vital level')
        ax.set_title(v, color=VITAL_COLORS[v])
        # sample size under each column
        for j, lvl in enumerate(s['levels']):
            if s['counts'][j] > 0:
                ax.annotate(f'{int(s["counts"][j])}', xy=(j, len(achs) - 0.4),
                            fontsize=5, ha='center', va='top', color='0.6')
    axes[0].set_yticks(range(len(achs)))
    axes[0].set_yticklabels([a.replace('_', ' ') for a in achs], fontsize=6)
    fig.colorbar(im, ax=list(axes), fraction=0.02, pad=0.02,
                 label=f'P(unlock ≤ {horizon} steps | vital)')
    fig.suptitle('p(achievement | vital) — empirical conditional '
                 '(n below each column)', fontsize=12)
    out = Path(save_path) / 'achievement_rate_vs_vital.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
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
    parser.add_argument('--achievement_horizon', type=int, default=15,
                        help='horizon (steps) for p(achievement unlock | vital)')
    parser.add_argument('--avg_clip_pct', type=float, default=5,
                        help='averaged-vitals clip length: percentile of episode '
                             'lengths to clip to (default 5 = ignore the shortest '
                             '5%% so one early death does not collapse the window). '
                             '0 = strict min episode length.')
    parser.add_argument('--phase_data', type=str, nargs='*', default=None,
                        help='labelled trajectory sets for the averaged-vitals '
                             'overlay, e.g. early:./traj_ck1 mid:./traj_ck2 '
                             'late:./traj_ck3 (each label:dir). When given, the '
                             'averaged-vitals plot overlays one curve per phase '
                             'instead of the single --data set.')
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
    # 1. per-episode plots (now with cause-of-death, hurt markers, world map)
    n_plotted = 0
    if args.max_episode_plots:                 # 0 = skip per-episode plots
        for i, ep in enumerate(episodes):
            if n_plotted >= args.max_episode_plots:
                break
            if plot_episode(ep, save_dir, i, meta=meta):
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

        # 4. averaged vitals over time (single set, or phase overlay)
        if args.phase_data:
            groups = []
            for spec in args.phase_data:
                label, _, d = spec.partition(':')
                if not d:
                    print(f'  WARN: --phase_data entry "{spec}" is not label:dir; skipping')
                    continue
                try:
                    eps_p, _ = load_episodes(d)
                except Exception as e:
                    print(f'  WARN: could not load phase "{label}" from {d}: {e}')
                    continue
                st, L = compute_averaged_vitals(eps_p, clip_pct=args.avg_clip_pct)
                groups.append((label, st, L))
                print(f'  phase "{label}": {st[VITALS[0]]["n"] if st else 0} eps, clip_len={L}')
        else:
            st, L = compute_averaged_vitals(episodes, clip_pct=args.avg_clip_pct)
            groups = [('all episodes', st, L)]
            print(f'  averaged vitals: {st[VITALS[0]]["n"] if st else 0} eps, clip_len={L}')
        out4 = plot_averaged_vitals(groups, save_dir)
        results['averaged_vitals'] = {lbl: st for lbl, st, _ in groups}
        if out4:
            outputs.append(str(out4))
            print(f'Wrote {out4}')

        # 5. p(achievement | vital) — empirical conditional
        av_stats = compute_achievement_rate_vs_vital(
            episodes, horizon=args.achievement_horizon)
        out5 = plot_achievement_rate_vs_vital(
            av_stats, save_dir, args.achievement_horizon)
        results['achievement_rate_vs_vital'] = av_stats
        if out5:
            outputs.append(str(out5))
            print(f'Wrote {out5}')
        else:
            print('No achievement unlocks found; skipping p(achievement|vital)')

    res_file = save_dir / 'vital_dynamics_results.pkl'
    with open(res_file, 'wb') as f:
        pickle.dump(results, f)
    outputs.append(str(res_file))
    print(f'Saved results to {res_file}')

    log_run_info(save_dir, 'vital_dynamics', vars(args), outputs,
                 extra={'n_episodes': len(episodes), 'n_with_vitals': have_vitals})


if __name__ == '__main__':
    main()
