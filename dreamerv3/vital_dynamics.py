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
     Both panels are shaded by the day/night cycle (darker = night) via
     `_shade_daynight`, matching the GUI dashboard's vital timeline. The daylight
     trace comes from `get_daylight`: it prefers the per-step `daylight` logged by
     eval_trajectory (crafter's `log/daylight`), and otherwise recomputes it from
     `env_step` or the step index via crafter's formula, so pkls that predate
     daylight logging still render the cycle (the step-index fallback is exact for
     fresh eval resets, which always start at env-step 0).
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
import json
import pickle
from pathlib import Path
from collections import defaultdict
import plot_presentable as pp

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from run_info import log_run_info

# Output figure format (set from --fmt in main). SVG is vector (Illustrator-
# friendly) and ignores dpi; png/pdf also supported.
FIG_EXT = 'png'

# Fixed width (inches) for the per-episode strip, INDEPENDENT of episode length:
# long episodes condense horizontally into the same footprint instead of growing
# unboundedly wide (a 2000-step episode would otherwise be ~25 in). Keeps every
# vitals_episode_*.png the same size for poster assembly. matplotlib has no
# practical point limit, so 2k+ step episodes just read denser.
EP_STRIP_WIDTH = 5.0

VITALS = ['health', 'food', 'drink', 'energy']
VITAL_COLORS = {
    'health': '#d62728', 'food': '#ff7f0e',
    'drink': '#1f77b4', 'energy': '#2ca02c'}
VITAL_MAX = 9  # crafter vitals are ints in [0, 9]

# Curated behaviors for the subset p(behavior|vital) heatmap
# (achievement_rate_vs_vital_subset): the early/vital-driven behaviors. Includes
# 'sleep' (a raw action, not an achievement) to show the energy->rest link.
SUBSET_ACHIEVEMENTS = [
    'collect_drink', 'sleep', 'make_wood_pickaxe', 'make_wood_sword',
]

# Raw actions folded into the achievement rate/lift analysis as extra "behavior"
# rows (they are not achievements — we're showing how behavior shifts with state,
# e.g. does the agent sleep more when energy is low?). name -> crafter action idx.
BEHAVIOR_ACTIONS = {'sleep': 6}
# Cow-count overlay color (distinct from the vital + hunger-cause palette).
COW_COLOR = '#5c4033'
# Hostile-mob-count color (distinct from vitals/cow/hurt palette).
MOB_COLOR = '#9467bd'

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


# Crafter's lava death (objects.py `_move`) zeroes health the SAME env-step the
# agent steps onto a lava tile, but eval_trajectory's obs/action logging can place
# that move one row before the health=0 observation. Look back this many steps for
# the move onto lava rather than only checking the fatal step.
LAVA_MOVE_LOOKBACK = 2


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
        # Lava is the ONLY crafter mechanic that zeroes health in one step from any
        # value (objects.py `_move`); necessity degen is gradual (-1, gated above
        # by `lacking`) and mob hits are small/fixed (zombie 2 awake / 7 asleep,
        # arrow 2) and never require a move. So an instant drop to 0 at full
        # necessities is lava iff the agent moved onto the tile -- checked over a
        # short lookback window (the fatal move may be logged a row early), with a
        # magnitude catch-all (>=8 lost in one step is lava outright; no mob does
        # 8+, so this also covers missing/misaligned action logs).
        instant_to_zero = (h[t] == 0 and h[t - 1] >= 2)  # lost >=2 to 0 in one step
        hard_oneshot = (h[t] == 0 and h[t - 1] >= 8)
        if actions is not None:
            lo = max(0, t - LAVA_MOVE_LOOKBACK)
            recent_move = any(1 <= int(actions[s]) <= 4
                              for s in range(lo, min(t + 1, len(actions))))
        else:
            recent_move = True  # no action info -> can't rule out a move onto lava
        if not lacking and instant_to_zero and (hard_oneshot or recent_move):
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
    # Per-episode pkls carry no world metadata; read it from the lightweight
    # sidecar (written by eval_trajectory even when all_episodes.pkl is skipped)
    # so the world-map panel still has env_seed/area/fixed_layout/island.
    meta = {}
    sidecar = p / 'trajectory_meta.json'
    if sidecar.exists():
        try:
            with open(sidecar) as f:
                meta = json.load(f)
        except Exception:
            pass
    return eps, meta


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


def get_cows(ep):
    """Return {worldgen, world, inview: np.array(T)} per-step cow counts (from the
    crafter wrapper's log/num_cows_* counters), or None if not recorded (older
    pkls). worldgen is constant within an episode (initial supply); world is the
    live map total; inview is the subset the agent can actually see."""
    src = {'worldgen': 'num_cows_worldgen', 'world': 'num_cows_world',
           'inview': 'num_cows_inview'}
    out = {}
    for k, key in src.items():
        if key not in ep or len(ep[key]) == 0:
            return None
        out[k] = np.asarray(ep[key])
    return out


def get_mobs(ep):
    """Return {world, inview: np.array(T)} per-step hostile-mob counts (zombies +
    skeletons) from the crafter wrapper's log/num_mobs_* counters, or None if not
    recorded. Mob counting was added 2026-11-08, so trajectories eval'd before
    then have no mob trace (the plot simply omits that panel). world = live map
    total; inview = subset in the agent's view window (threat exposure)."""
    src = {'world': 'num_mobs_world', 'inview': 'num_mobs_inview'}
    out = {}
    for k, key in src.items():
        if key not in ep or len(ep[key]) == 0:
            return None
        out[k] = np.asarray(ep[key])
    return out


def get_daylight(ep):
    """Per-step daylight in [0,1] for the episode (1=noon, ~0=midnight), or None
    if it can't be derived.

    Prefers the recorded `daylight` trace (logged by eval_trajectory from the
    crafter wrapper's log/daylight); falls back to recomputing it from `env_step`
    (or, failing that, the step index) via crafter's formula so older pkls that
    predate daylight logging still render the cycle. The step-index fallback
    assumes the episode starts at env-step 0 (true for a fresh eval reset)."""
    dl = ep.get('daylight')
    if dl is not None and len(dl):
        return np.asarray(dl, dtype=float)
    T = int(ep.get('length') or len(ep.get('player_pos', [])) or 0)
    if T <= 0:
        return None
    es = ep.get('env_step')
    steps = np.asarray(es, dtype=float) if (es is not None and len(es)) \
        else np.arange(T, dtype=float)
    # crafter.env._update_time: progress=(step/300)%1+0.3; daylight=1-|cos(pi*p)|^3
    progress = (steps / 300.0) % 1 + 0.3
    return 1 - np.abs(np.cos(np.pi * progress)) ** 3


def _shade_daynight(ax, daylight, T):
    """Shade an axis's background by the day/night cycle: darker = night.

    Night is painted as a semi-transparent indigo whose opacity tracks
    (1 - daylight), so daytime stays clear and midnight is darkest. Sits behind
    all plotted data (negative zorder). Matches dashboard_backend._shade_daynight
    so the standalone SVGs and the GUI timeline read identically."""
    if daylight is None or len(daylight) == 0:
        return
    night = (1.0 - np.clip(np.asarray(daylight, dtype=float), 0, 1)).reshape(1, -1)
    rgba = np.zeros((1, night.shape[1], 4))
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = 0.06, 0.06, 0.22  # dark indigo
    rgba[..., 3] = 0.5 * night[0]  # opacity ~ darkness
    y0, y1 = ax.get_ylim()
    ax.imshow(rgba, extent=[0, max(1, T - 1), y0, y1], aspect='auto',
              origin='lower', zorder=-5, interpolation='nearest')
    ax.set_ylim(y0, y1)


def get_unlocks(ep):
    """From the per-step cumulative achievement dicts, return {achievement:
    [unlock step, ...]}. Every count increase (0->1 OR any later increment) is
    one event, so repeated behaviors (eat_cow, collect_drink, ...) contribute
    all their occurrences, not just the first unlock."""
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


def get_events(ep, extra_actions=None):
    """Behavior events as {name: [step, ...]} — achievement unlocks (get_unlocks)
    plus optional raw-action events. `extra_actions` is {name: action_idx}; every
    step that action is taken counts as one event (e.g. 'sleep'->6), so actions
    fold into the exact same horizon-conditional / lift framework as achievements.
    These are NOT achievements — they capture state-specific behavior shifts."""
    events = dict(get_unlocks(ep))
    if extra_actions:
        a = get_actions(ep)
        for name, idx in extra_actions.items():
            steps = np.where(a == idx)[0]
            if steps.size:
                events[name] = steps.tolist()
    return events


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


def _daynight_strip(ax, daylight, T):
    """Draw a thin day/night bar: warm daytime -> indigo night, interpolated by
    the per-step daylight value. No ticks; shares the step x-axis."""
    ax.set_ylim(0, 1)
    if daylight is None or len(daylight) == 0:
        ax.set_facecolor('0.95')
    else:
        dl = np.clip(np.asarray(daylight, dtype=float), 0, 1).reshape(1, -1, 1)
        day_col = np.array([1.0, 0.97, 0.80])     # warm noon
        night_col = np.array([0.10, 0.10, 0.30])  # indigo midnight
        strip = night_col * (1 - dl) + day_col * dl          # (1, T, 3)
        ax.imshow(strip, aspect='auto', extent=[0, max(1, T - 1), 0, 1],
                  origin='lower', interpolation='nearest', zorder=0)
    ax.set_yticks([])
    ax.set_ylabel('day/\nnight', rotation=0, ha='right', va='center', fontsize=9)
    ax.tick_params(labelbottom=False)


def _sleep_strip(ax, actions, steps, T):
    """Draw a thin lane with a tick wherever the agent sleeps (action==6)."""
    ax.set_ylim(0, 1)
    if len(actions) == T:
        sleep_steps = steps[actions == VITAL_ACTIONS['sleep']]
        if len(sleep_steps):
            ax.vlines(sleep_steps, 0, 1, color='#222222', lw=0.9, zorder=3)
    ax.set_yticks([])
    ax.set_ylabel('sleep', rotation=0, ha='right', va='center', fontsize=9)
    ax.tick_params(labelbottom=False)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)


def plot_episode(ep, save_path, ep_idx, meta=None):
    vitals = get_vitals(ep)
    if vitals is None:
        return False
    actions = get_actions(ep)
    T = len(next(iter(vitals.values())))
    steps = np.arange(T)
    death = cause_of_death(vitals, actions)
    daylight = get_daylight(ep)        # per-step day/night cycle (None if unknown)
    cows = get_cows(ep)
    mobs = get_mobs(ep)                 # None on pkls that predate mob counting

    # Trace panels, one series per panel, stacked top->bottom: the 4 vitals, then
    # cow level and mob level when recorded. (key, label, color, data, y-max).
    trace_panels = [('vital', v, VITAL_COLORS[v], np.asarray(vitals[v]), VITAL_MAX)
                    for v in VITALS]
    if cows is not None:
        trace_panels.append(('cow', 'cows\nin view', COW_COLOR, cows['inview'],
                             None))
    if mobs is not None:
        trace_panels.append(('mob', 'mobs\nin view', MOB_COLOR, mobs['inview'],
                             None))
    n_tr = len(trace_panels)

    # Thin vertical column: two thin bars (day/night, sleep) over the stacked
    # trace panels, all sharing the step axis. Narrow width + short panels so the
    # whole thing reads as a compact strip. Height grows with #trace panels.
    BAR_H, TR_H = 0.20, 0.58
    width = EP_STRIP_WIDTH               # fixed: long episodes condense, not widen
    height = 0.7 + 2 * BAR_H + n_tr * TR_H
    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(2 + n_tr, 1,
                          height_ratios=[BAR_H, BAR_H] + [TR_H] * n_tr,
                          hspace=0.32)
    ax_dn = fig.add_subplot(gs[0, 0])
    ax_sl = fig.add_subplot(gs[1, 0], sharex=ax_dn)
    axes = [fig.add_subplot(gs[2 + i, 0], sharex=ax_dn) for i in range(n_tr)]

    _daynight_strip(ax_dn, daylight, T)
    _sleep_strip(ax_sl, actions, steps, T)

    for ax, (kind, label, color, data, ymax) in zip(axes, trace_panels):
        ax.plot(steps, data, color=color, lw=1.6, zorder=3)
        ax.fill_between(steps, 0, data, color=color, alpha=0.10, zorder=1)
        if kind == 'vital':
            ax.set_ylim(-0.5, VITAL_MAX + 0.5)
            ax.set_yticks([0, VITAL_MAX])
        else:  # cow / mob counts
            top = max(1, int(np.max(data))) + 1
            ax.set_ylim(0, top)
            ax.set_yticks([0, top])
        ax.set_ylabel(label, rotation=0, ha='right', va='center', fontsize=9)
        ax.tick_params(axis='y', labelsize=8, pad=1)
        if ax is not axes[-1]:
            ax.tick_params(labelbottom=False)

    axes[-1].set_xlabel('step')
    ax_dn.set_xlim(0, max(1, T - 1))
    try:
        pp.nice_ticks(axes[-1], n=4, axis='x')
    except Exception:
        pass

    # Concise title above the top (day/night) bar; details that would widen the
    # thin column (cow/mob averages) live in the panels themselves.
    ax_dn.set_title(
        f'Episode {ep.get("episode", ep_idx)}  ·  len {T}  ·  {death}',
        fontsize=11, fontweight='bold', pad=6)
    out = Path(save_path) / f'vitals_episode_{ep_idx:03d}.{FIG_EXT}'
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
    out = Path(save_path) / f'action_rate_vs_vital.{FIG_EXT}'
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
    out = Path(save_path) / f'event_triggered_vitals.{FIG_EXT}'
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


def plot_averaged_vitals(groups, save_path, fname=None):
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
    out = Path(save_path) / (fname or f'averaged_vitals.{FIG_EXT}')
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# 5. Empirical p(achievement unlock within horizon | vital level)
# --------------------------------------------------------------------------- #
def compute_achievement_rate_vs_vital(episodes, horizon=15, extra_actions=None,
                                      phase=None, bright=0.8, dark=0.2):
    """For each vital and level L, P(behavior b occurs within the next `horizon`
    steps | vital == L), pooled over all steps/episodes. This is the OBSERVATIONAL
    conditional (correlational) — the policy net has no achievement head, so this
    comes from behavior, not a forward pass. Behaviors = achievement unlocks plus
    any `extra_actions` ({name: action_idx}, e.g. BEHAVIOR_ACTIONS for 'sleep'),
    folded in via get_events. Returns {vital: {'levels', 'ach', 'P'(n_ach,
    n_levels), 'counts'(n_levels)}}.

    `phase` (None/'day'/'night') restricts the CONDITIONING steps by time of day
    (peak-window daylight: day = daylight >= `bright`, night <= `dark`; twilight
    excluded), so the conditional becomes P(b soon | vital==L, time-of-day). The
    look-ahead window is unaffected (a behavior may land just after the phase).
    Episodes without recorded daylight are skipped when `phase` is set."""
    levels = np.arange(VITAL_MAX + 1)
    ach_names = sorted({a for ep in episodes for a in get_events(ep, extra_actions)})
    idx = {a: i for i, a in enumerate(ach_names)}
    acc = {v: {'hit': np.zeros((len(ach_names), len(levels))),
               'tot': np.zeros(len(levels))} for v in VITALS}
    for ep in episodes:
        vit = get_vitals(ep)
        if vit is None or not ach_names:
            continue
        T = len(vit['health'])
        pmask = None
        if phase is not None:
            dl = get_daylight(ep)
            if dl is None:
                continue
            dl = np.asarray(dl)[:T]
            pmask = (dl >= bright) if phase == 'day' else (dl <= dark)
        unlocks = get_events(ep, extra_actions)
        # future[a, t] = behavior a occurs at some t0 in [t, t+horizon]
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
                if pmask is not None:
                    mask = mask & pmask
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
            # Marginal base rate p(ach unlocks within horizon) = total hits over
            # all steps / total steps. Identical across vitals (every step is
            # counted once per vital), so the lift p(ach|v)/p(ach) centers at 1.
            total = tot.sum()
            base = (acc[v]['hit'].sum(axis=1) / total
                    if total > 0 else np.full(len(ach_names), np.nan))
        out[v] = {'levels': levels, 'ach': ach_names, 'P': P,
                  'counts': tot, 'base': base}
    return out


def plot_achievement_rate_vs_vital(stats, save_path, horizon, ach_subset=None,
                                   fname='achievement_rate_vs_vital',
                                   title='p(achievement | vital)'):
    """Heatmap per vital: rows = achievements, cols = vital level, color =
    P(unlock within horizon | vital). Shared color scale across vitals.

    `ach_subset`: optional list of achievement names to restrict (and order) the
    rows. Names absent from the data are silently dropped; the row order follows
    the subset. `fname`/`title` let callers emit a separate figure (e.g. a
    curated subset) alongside the full heatmap. The color scale is computed over
    the *displayed* rows so a small subset keeps a readable range."""
    all_achs = stats[VITALS[0]]['ach']
    if not all_achs:
        return None
    if ach_subset is not None:
        pos = {a: i for i, a in enumerate(all_achs)}
        rows = [pos[a] for a in ach_subset if a in pos]
        achs = [all_achs[i] for i in rows]
        if not achs:
            print(f'  None of the requested achievements present; skipping {fname}')
            return None
    else:
        rows = list(range(len(all_achs)))
        achs = all_achs
    # P restricted to the displayed rows, per vital
    P = {v: stats[v]['P'][rows, :] for v in VITALS}
    vmax = max([np.nanmax(P[v]) for v in VITALS
                if np.isfinite(P[v]).any()] + [1e-9])
    fig, axes = plt.subplots(1, len(VITALS), sharey=True,
                             figsize=(3.6 * len(VITALS), max(4.0, 0.32 * len(achs))))
    axes = np.atleast_1d(axes)
    im = None
    for ax, v in zip(axes, VITALS):
        s = stats[v]
        im = ax.imshow(P[v], aspect='auto', cmap='magma', vmin=0, vmax=vmax)
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
    cb = fig.colorbar(im, ax=list(axes), fraction=0.02, pad=0.02)
    cb.set_label(f'P(unlock ≤ {horizon} | vital)', fontsize=11)
    fig.suptitle(f'{title} — empirical conditional '
                 '(n below each column)', fontsize=12)
    out = Path(save_path) / f'{fname}.{FIG_EXT}'
    fig.savefig(out, dpi=130, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    return out


def plot_achievement_lift_vs_vital(stats, save_path, horizon, ach_subset=None,
                                   fname='achievement_lift_vs_vital',
                                   title='p(action | vital) / p(action)',
                                   n_min=20, log=True, vmin=0.5, vmax=2.0,
                                   vmax_cap=4.0):
    """Lift heatmap per vital: rows = achievements, cols = vital level, color =
    p(unlock within horizon | vital) / p(unlock within horizon). This divides
    out each achievement's base rate so the color reads as enrichment relative
    to that achievement's own average: 1 = independent, >1 = this vital level
    makes the behavior MORE likely than usual, <1 = less likely. Diverging
    colormap centered at 1.

    `log=True` (default) uses a log color scale clipped to [`vmin`, `vmax`]
    (default [0.5, 2.0]) — symmetric about 1 in log space, so lift=1 maps to the
    middle (white) of the diverging map and a halving (0.5) reads as blue exactly
    as strongly as a doubling (2.0) reads red. Values are clipped BEFORE the norm
    so a true 0 (behavior never seen at that level) renders deepest blue instead
    of being dropped by log(0). `log=False` restores the old linear
    `TwoSlopeNorm(0, 1, data_max)` scale with `vmax_cap` clipping the red end.

    Columns with fewer than `n_min` steps are masked (gray) — lift has high
    variance when a level bin is thin. `ach_subset` restricts/orders the rows
    exactly like plot_achievement_rate_vs_vital."""
    from matplotlib.colors import TwoSlopeNorm, LogNorm
    all_achs = stats[VITALS[0]]['ach']
    if not all_achs:
        return None
    if ach_subset is not None:
        pos = {a: i for i, a in enumerate(all_achs)}
        rows = [pos[a] for a in ach_subset if a in pos]
        achs = [all_achs[i] for i in rows]
        if not achs:
            print(f'  None of the requested achievements present; skipping {fname}')
            return None
    else:
        rows = list(range(len(all_achs)))
        achs = all_achs
    # lift = P / base, per vital, restricted to displayed rows + count-masked
    lift = {}
    for v in VITALS:
        s = stats[v]
        base = s['base'][rows]
        with np.errstate(invalid='ignore', divide='ignore'):
            L = s['P'][rows, :] / base[:, None]
        L[:, s['counts'] < n_min] = np.nan          # mask thin level bins
        lift[v] = L
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('0.85')                            # masked cells -> light gray
    cbar_ticks = None
    if log:
        if vmin <= 0:
            raise SystemExit(f'lift log scale needs vmin > 0, got {vmin}')
        # Clip BEFORE the norm so NaN (thin bins) stays NaN -> gray, while a true
        # 0 is brought up to vmin -> deepest blue (LogNorm would otherwise drop
        # log(0)). np.clip is NaN-preserving.
        lift = {v: np.clip(L, vmin, vmax) for v, L in lift.items()}
        norm = LogNorm(vmin=vmin, vmax=vmax)
        # Geometric (log-symmetric) ticks: vmin, √vmin, 1, √vmax, vmax.
        cbar_ticks = sorted({vmin, vmin ** 0.5, 1.0, vmax ** 0.5, vmax})
    else:
        finite = np.concatenate([lift[v][np.isfinite(lift[v])] for v in VITALS]
                                if any(np.isfinite(lift[v]).any() for v in VITALS)
                                else [np.array([1.0])])
        vhi = float(min(vmax_cap, max(1.2, np.nanmax(finite)))) if finite.size else 2.0
        norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=vhi)
    fig, axes = plt.subplots(1, len(VITALS), sharey=True,
                             figsize=(3.6 * len(VITALS), max(4.0, 0.32 * len(achs))))
    axes = np.atleast_1d(axes)
    im = None
    for ax, v in zip(axes, VITALS):
        s = stats[v]
        im = ax.imshow(lift[v], aspect='auto', cmap=cmap, norm=norm)
        ax.set_xticks(range(len(s['levels'])))
        ax.set_xticklabels(s['levels'], fontsize=7)
        ax.set_xlabel('vital level')
        ax.set_title(v, color=VITAL_COLORS[v])
        for j, lvl in enumerate(s['levels']):
            if s['counts'][j] > 0:
                ax.annotate(f'{int(s["counts"][j])}', xy=(j, len(achs) - 0.4),
                            fontsize=5, ha='center', va='top', color='0.4')
    axes[0].set_yticks(range(len(achs)))
    axes[0].set_yticklabels([a.replace('_', ' ') for a in achs], fontsize=6)
    cb = fig.colorbar(im, ax=list(axes), fraction=0.02, pad=0.02)
    # Short label at a fixed size: the full p(·|vital)/p(·) formula is already in
    # the suptitle, and a long vertical label overflows the thin colorbar (clips
    # under the larger presentable fonts).
    cb.set_label('lift  (1 = baseline)', fontsize=11)
    if cbar_ticks is not None:                       # log scale: explicit ticks
        cb.ax.minorticks_off()                        # drop LogNorm's minor labels
        cb.set_ticks(cbar_ticks)
        # Mark the end ticks as saturated (data is clipped to [vmin, vmax]), so
        # "<0.5" / ">2" signal that out-of-range cells pile onto the extremes.
        n = len(cbar_ticks)
        cb.set_ticklabels([
            f'<{t:.2g}' if i == 0 else f'>{t:.2g}' if i == n - 1 else f'{t:.2g}'
            for i, t in enumerate(cbar_ticks)])
    scale_txt = (f'log scale, clipped [{vmin:g}, {vmax:g}]' if log
                 else f'linear scale, cap {vmax_cap:g}')
    fig.suptitle(f'{title} — enrichment vs base rate '
                 f'(horizon {horizon}; cols <{n_min} steps masked; {scale_txt})',
                 fontsize=12)
    out = Path(save_path) / f'{fname}.{FIG_EXT}'
    fig.savefig(out, dpi=130, bbox_inches='tight', pad_inches=0.2)
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
    parser.add_argument('--lift_min_count', type=int, default=20,
                        help='min steps per vital level to show a lift cell '
                             '(thinner columns are masked gray)')
    parser.add_argument('--lift_vmin', type=float, default=0.5,
                        help='lift heatmap lower color bound (log scale; '
                             'default 0.5 = halving)')
    parser.add_argument('--lift_vmax', type=float, default=2.0,
                        help='lift heatmap upper color bound (log scale; '
                             'default 2.0 = doubling)')
    parser.add_argument('--lift_linear', action='store_true',
                        help='use the old linear diverging lift scale instead of '
                             'the (default) log scale')
    parser.add_argument('--avg_clip_pct', type=float, default=5,
                        help='averaged-vitals clip length: percentile of episode '
                             'lengths to clip to (default 5 = ignore the shortest '
                             '5%% so one early death does not collapse the window). '
                             '0 = strict min episode length.')
    parser.add_argument('--fmt', type=str, default='svg',
                        choices=['svg', 'pdf', 'png'],
                        help='output figure format (default svg = vector, '
                             'Illustrator-friendly; dpi ignored for svg/pdf)')
    parser.add_argument('--phase_data', type=str, nargs='*', default=None,
                        help='labelled trajectory sets for the averaged-vitals '
                             'overlay, e.g. early:./traj_ck1 mid:./traj_ck2 '
                             'late:./traj_ck3 (each label:dir). When given, the '
                             'averaged-vitals plot overlays one curve per phase '
                             'instead of the single --data set.')
    pp.add_presentable_arg(parser)
    args = parser.parse_args()
    pp.maybe_apply(args, small_figsize=False)  # Tier-1 style; no-op unless --presentable

    global FIG_EXT
    FIG_EXT = args.fmt

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
        outputs.append(str(save_dir / f'vitals_episode_*.{FIG_EXT}'))

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

        # 5. p(behavior | vital) — empirical conditional (achievements + sleep)
        av_stats = compute_achievement_rate_vs_vital(
            episodes, horizon=args.achievement_horizon,
            extra_actions=BEHAVIOR_ACTIONS)
        out5 = plot_achievement_rate_vs_vital(
            av_stats, save_dir, args.achievement_horizon)
        results['achievement_rate_vs_vital'] = av_stats
        if out5:
            outputs.append(str(out5))
            print(f'Wrote {out5}')
        else:
            print('No achievement unlocks found; skipping p(achievement|vital)')

        # 5b. Same conditional, restricted to a curated set of achievements
        out5b = plot_achievement_rate_vs_vital(
            av_stats, save_dir, args.achievement_horizon,
            ach_subset=SUBSET_ACHIEVEMENTS,
            fname='achievement_rate_vs_vital_subset',
            title='p(achievement | vital) [subset]')
        if out5b:
            outputs.append(str(out5b))
            print(f'Wrote {out5b}')

        # 5c. Lift view: p(ach|vital) / p(ach) — base-rate-normalized enrichment
        if out5:
            out5c = plot_achievement_lift_vs_vital(
                av_stats, save_dir, args.achievement_horizon,
                n_min=args.lift_min_count, log=not args.lift_linear,
                vmin=args.lift_vmin, vmax=args.lift_vmax)
            if out5c:
                outputs.append(str(out5c))
                print(f'Wrote {out5c}')
            out5d = plot_achievement_lift_vs_vital(
                av_stats, save_dir, args.achievement_horizon,
                ach_subset=SUBSET_ACHIEVEMENTS,
                fname='achievement_lift_vs_vital_subset',
                title='p(action | vital) / p(action) [subset]',
                n_min=args.lift_min_count, log=not args.lift_linear,
                vmin=args.lift_vmin, vmax=args.lift_vmax)
            if out5d:
                outputs.append(str(out5d))
                print(f'Wrote {out5d}')

            # 5e. Day/night gain views — same enrichment, split by time of day
            # (full + subset for each phase). Needs per-step daylight.
            if any(get_daylight(ep) is not None for ep in episodes):
                for phase in ('day', 'night'):
                    ph_stats = compute_achievement_rate_vs_vital(
                        episodes, horizon=args.achievement_horizon,
                        extra_actions=BEHAVIOR_ACTIONS, phase=phase)
                    o_full = plot_achievement_lift_vs_vital(
                        ph_stats, save_dir, args.achievement_horizon,
                        fname=f'achievement_lift_vs_vital_{phase}',
                        title=f'p(action | vital) / p(action) [{phase}]',
                        n_min=args.lift_min_count, log=not args.lift_linear,
                        vmin=args.lift_vmin, vmax=args.lift_vmax)
                    o_sub = plot_achievement_lift_vs_vital(
                        ph_stats, save_dir, args.achievement_horizon,
                        ach_subset=SUBSET_ACHIEVEMENTS,
                        fname=f'achievement_lift_vs_vital_subset_{phase}',
                        title=f'p(action | vital) / p(action) [{phase}, subset]',
                        n_min=args.lift_min_count, log=not args.lift_linear,
                        vmin=args.lift_vmin, vmax=args.lift_vmax)
                    for o in (o_full, o_sub):
                        if o:
                            outputs.append(str(o))
                            print(f'Wrote {o}')
            else:
                print('No daylight recorded; skipping day/night gain plots')

    res_file = save_dir / 'vital_dynamics_results.pkl'
    with open(res_file, 'wb') as f:
        pickle.dump(results, f)
    outputs.append(str(res_file))
    print(f'Saved results to {res_file}')

    log_run_info(save_dir, 'vital_dynamics', vars(args), outputs,
                 extra={'n_episodes': len(episodes), 'n_with_vitals': have_vitals})


if __name__ == '__main__':
    main()
