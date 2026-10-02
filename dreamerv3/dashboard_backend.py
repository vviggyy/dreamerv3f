"""
Compute backend for the interactive environment/behavior dashboard
(`dreamerv3/dashboard.py`).

This module is the half that touches JAX / crafter. It is deliberately kept
free of any Streamlit imports so it can also be driven head-less (e.g. from a
notebook or a batch job) and unit-tested.

Responsibilities
----------------
  * `load_agent(logdir, device)` -- rebuild the exact training config by
    inheriting the run's saved `config.yaml` (same inheritance main.py does for
    the INFERENCE_SCRIPTS), construct the agent via `main.make_agent`, and load
    `ckpt/latest`. The returned object is meant to be cached and kept resident
    on the GPU across many `generate()` calls (the agent is independent of the
    env knobs -- only the env is rebuilt per generation).

  * `generate(...)` -- build a crafter env with the dashboard's knob overrides,
    roll the loaded policy out for N episodes using the same `embodied.Driver`
    machinery as `eval_trajectory.py`, and stream **one pickle per episode** to
    a cache folder (image + pos + action + reward + vitals + per-step
    achievement unlocks -- NO activations, so it stays tiny). A lightweight
    `index.json` summarises every episode (length / reward / achievements /
    cause-of-death) for the episode picker.

RAM note: the viewer only ever loads the single selected episode from disk, so
peak RAM is ~one episode's frames (tens of MB), independent of N. The cache
folder is persistent -- it is the "separate folder with the data".
"""

import hashlib
import json
import pathlib
import pickle
import sys
import time
from collections import defaultdict
from functools import partial as bind

import numpy as np

# --- make the dreamerv3 package importable when run under `streamlit run` -----
REPO = pathlib.Path(__file__).resolve().parent.parent
_PKG = REPO / 'dreamerv3'
for _p in (str(REPO), str(_PKG)):
    # REPO enables `import dreamerv3.main`; _PKG enables the sibling imports some
    # modules do (e.g. vital_dynamics' `from run_info import ...`).
    if _p not in sys.path:
        sys.path.insert(0, _p)

import elements  # noqa: E402
import embodied  # noqa: E402
import ruamel.yaml as yaml  # noqa: E402

import dreamerv3.main as dmain  # noqa: E402  (make_agent / make_env / wrap_env)
from dreamerv3.vital_dynamics import cause_of_death  # noqa: E402

# Crafter's canonical action list (index -> name).
try:
    import crafter  # noqa: E402
    ACTION_NAMES = list(crafter.constants.actions)
except Exception:  # pragma: no cover - fallback if crafter import layout changes
    ACTION_NAMES = [
        'noop', 'move_left', 'move_right', 'move_up', 'move_down', 'do',
        'sleep', 'place_stone', 'place_table', 'place_furnace', 'place_plant',
        'make_wood_pickaxe', 'make_stone_pickaxe', 'make_iron_pickaxe',
        'make_wood_sword', 'make_stone_sword', 'make_iron_sword']

VITALS = ('health', 'food', 'drink', 'energy')

# The env.crafter knobs the dashboard exposes. Kept here so the UI and the
# cache-key hashing share one source of truth.
KNOB_DEFAULTS = {
    'disable_mobs': False,
    'random_spawn': False,
    'fixed_seed': False,
    'fixed_layout': False,
    'island_border': False,
    'island_fill': 0.72,
    'island_roughness': 0.16,
    'cow_spawn_prob': 0.01,
    'cow_span_dist': 5,
    'cow_target_scale': 1.0,
    'cow_worldgen_scale': 1.0,
    'egocentric_view': 0,
}

CACHE_ROOT = REPO / 'dashboard_cache'


# ============================================================================
# Config / agent construction
# ============================================================================

def _build_config(logdir, seed, knobs, device):
    """Rebuild the training config by inheriting the run's saved config.yaml,
    then apply the dashboard knob overrides on top of env.crafter.*."""
    logdir = pathlib.Path(logdir)
    cfg_folder = REPO / 'dreamerv3'
    configs = yaml.YAML(typ='safe').load(
        elements.Path(str(cfg_folder / 'configs.yaml')).read())
    config = elements.Config(configs['defaults'])

    saved_path = logdir / 'config.yaml'
    if not saved_path.exists():
        raise FileNotFoundError(
            f'No config.yaml in {logdir} -- cannot reconstruct the agent/env. '
            f'(This dashboard inherits the full resolved training config.)')
    saved = yaml.YAML(typ='safe').load(saved_path.read_text())
    # saved is the fully-resolved training config: inheriting it gives the exact
    # agent architecture, env settings, and size preset the net was trained with.
    config = config.update(saved)

    # Our overrides (dotted-flat keys, as main.py does for e.g. agent.record_*).
    overrides = {'logdir': str(logdir), 'seed': int(seed)}
    if device:
        overrides['jax.platform'] = device
    for k, v in (knobs or {}).items():
        overrides[f'env.crafter.{k}'] = v
    config = config.update(overrides)
    return config


def _resolve_checkpoint(logdir):
    logdir = pathlib.Path(logdir)
    latest = logdir / 'ckpt' / 'latest'
    if not latest.exists():
        raise FileNotFoundError(f'No ckpt/latest in {logdir}')
    name = latest.read_text().strip()
    ckpt = logdir / 'ckpt' / name
    if not (ckpt / 'agent.pkl').exists():
        raise FileNotFoundError(f'No agent.pkl in {ckpt}')
    return ckpt


class LoadedModel:
    """Bundle of the resident agent plus the reconstructed config, so later
    generations can rebuild the env off the same config without re-reading the
    saved yaml."""

    def __init__(self, agent, config, logdir, ckpt, device):
        self.agent = agent
        self.config = config
        self.logdir = str(logdir)
        self.ckpt = str(ckpt)
        self.device = device


def load_agent(logdir, device='cuda', seed=0):
    """Reconstruct + checkpoint-load the agent for `logdir`. Expensive; intended
    to be cached once and reused across generations (knobs don't affect it)."""
    config = _build_config(logdir, seed=seed, knobs={}, device=device)

    # Minimal portal setup (mirrors main.py). errfile disabled -- we are not a
    # distributed server here. Harmless if called once per process.
    try:
        import portal
        portal.setup(
            errfile=None,
            clientkw=dict(logging_color='cyan'),
            serverkw=dict(logging_color='cyan'),
            initfns=[],
            ipv6=config.ipv6,
        )
    except Exception as e:  # pragma: no cover
        print(f'[dashboard_backend] portal.setup skipped: {e}')

    agent = dmain.make_agent(config)
    ckpt = _resolve_checkpoint(logdir)
    with open(pathlib.Path(ckpt) / 'agent.pkl', 'rb') as f:
        data = pickle.load(f)
    # Skip optimizer state so a different wd/opt config doesn't shape-mismatch.
    agent.load(data, regex=r'^(?!opt/)')
    print(f'[dashboard_backend] loaded agent from {ckpt} (device={device})')
    return LoadedModel(agent, config, logdir, ckpt, device)


# ============================================================================
# Cache layout
# ============================================================================

def knobs_hash(knobs, seed, num_episodes):
    """Short stable hash of a generation request -> cache subfolder name."""
    payload = json.dumps(
        {'knobs': {k: knobs.get(k) for k in sorted(knobs)},
         'seed': int(seed), 'num_episodes': int(num_episodes)},
        sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


def run_cache_dir(logdir, knobs, seed, num_episodes, cache_root=None):
    root = pathlib.Path(cache_root) if cache_root else CACHE_ROOT
    name = pathlib.Path(logdir).name
    return root / name / knobs_hash(knobs, seed, num_episodes)


def episode_path(cache_dir, ep_num):
    return pathlib.Path(cache_dir) / f'episode_{ep_num:03d}.pkl'


def index_path(cache_dir):
    return pathlib.Path(cache_dir) / 'index.json'


def load_index(cache_dir):
    p = index_path(cache_dir)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_episode(cache_dir, ep_num):
    with open(episode_path(cache_dir, ep_num), 'rb') as f:
        return pickle.load(f)


# ============================================================================
# Rollout / generation
# ============================================================================

def _probe_world_meta(config):
    """Read env_seed / world_seed / area (for plot reconstruction + the index)
    without consuming an episode. Mirrors eval_trajectory's probe."""
    env = dmain.make_env(config, 0)
    env_seed = area = None
    try:
        env_seed = env._seed
    except (AttributeError, ValueError):
        pass
    try:
        area = tuple(int(x) for x in env._env._world._mat_map.shape)
    except (AttributeError, ValueError):
        pass
    env.close()
    world_seed = hash((env_seed, 1)) if env_seed is not None else None
    return env_seed, world_seed, area


def _episode_summary(ep, ep_num):
    """Lightweight summary dict for index.json / the episode picker."""
    vitals = {v: ep.get(f'vital_{v}') for v in VITALS}
    have_vitals = all(vitals[v] is not None and len(vitals[v]) for v in VITALS)
    if have_vitals:
        cod = cause_of_death(
            {v: np.asarray(vitals[v]) for v in VITALS},
            actions=np.asarray(ep.get('action', [])))
    else:
        cod = 'unknown'
    final = {k: int(v) for k, v in ep.get('final_achievements', {}).items() if v > 0}
    return {
        'episode': int(ep_num),
        'length': int(ep['length']),
        'reward': float(ep['total_reward']),
        'n_achievements': len(final),
        'achievements': final,
        'cause_of_death': cod,
    }


def generate(model, knobs, num_episodes=16, seed=0, cache_root=None,
             progress_cb=None, force=False):
    """Roll the loaded policy out for `num_episodes` under the given env knobs,
    streaming one pickle per episode into a cache folder.

    Args:
      model: a LoadedModel from load_agent().
      knobs: dict of env.crafter.* overrides (see KNOB_DEFAULTS).
      num_episodes: episodes to record.
      seed: env seed (changes the sampled world(s)).
      cache_root: base cache dir (defaults to repo/dashboard_cache).
      progress_cb: optional fn(done:int, total:int, msg:str) for a progress bar.
      force: regenerate even if a complete cached run already exists.

    Returns: (cache_dir:str, index:dict)
    """
    logdir = model.logdir
    cache_dir = run_cache_dir(logdir, knobs, seed, num_episodes, cache_root)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Reuse a complete cached run unless forced.
    existing = load_index(cache_dir)
    if existing and not force and len(existing.get('episodes', [])) >= num_episodes:
        if progress_cb:
            progress_cb(num_episodes, num_episodes, 'Loaded cached run')
        return str(cache_dir), existing

    # Rebuild config with these knobs + seed (agent stays the one already loaded).
    config = _build_config(logdir, seed=seed, knobs=knobs, device=model.device)
    agent = model.agent

    env_seed, world_seed, area = _probe_world_meta(config)
    island_meta = None
    if knobs.get('island_border'):
        island_meta = {
            'fill': float(knobs.get('island_fill', 0.72)),
            'roughness': float(knobs.get('island_roughness', 0.16)),
            'seed': int(knobs.get('island_seed', -1)),
        }

    # ---- per-episode recorder (lifted from eval_trajectory.logfn) ----
    episode_data = defaultdict(list)
    prev_ach = {}
    episode_count = [0]
    summaries = []

    def logfn(tran, worker):
        if worker != 0:
            return
        if tran['is_first']:
            episode_data.clear()
            prev_ach.clear()

        episode_data['player_pos'].append(np.asarray(tran['player_pos']).copy())
        episode_data['reward'].append(float(tran['reward']))
        episode_data['action'].append(int(tran.get('action', 0)))
        if 'image' in tran:
            episode_data['image'].append(np.asarray(tran['image']).astype(np.uint8))
        if 'log/player_facing_x' in tran:
            episode_data['player_facing'].append(np.array(
                [tran['log/player_facing_x'], tran['log/player_facing_y']],
                dtype=np.int32))
        for v in VITALS:
            key = f'log/{v}'
            if key in tran:
                episode_data[f'vital_{v}'].append(int(tran[key]))

        # per-step newly-unlocked achievements (diff vs previous step)
        current = {}
        unlocked_now = []
        for key in tran:
            if key.startswith('log/achievement_'):
                name = key[len('log/achievement_'):]
                count = int(tran[key])
                current[name] = count
                if count > prev_ach.get(name, 0):
                    unlocked_now.append(name)
        prev_ach.update(current)
        episode_data['unlocks'].append(unlocked_now)

        if tran['is_last']:
            episode_count[0] += 1
            ep_num = episode_count[0]
            ep = {}
            for k, v in episode_data.items():
                if k == 'unlocks':
                    ep[k] = list(v)  # ragged per-step list
                else:
                    ep[k] = np.array(v)
            ep['episode'] = ep_num
            ep['length'] = len(episode_data['player_pos'])
            ep['total_reward'] = float(sum(episode_data['reward']))
            ep['final_achievements'] = dict(current)
            ep['action_names'] = ACTION_NAMES

            if ep['length'] > 0:
                with open(episode_path(cache_dir, ep_num), 'wb') as f:
                    pickle.dump(ep, f)
                summaries.append(_episode_summary(ep, ep_num))
            if progress_cb:
                progress_cb(episode_count[0], num_episodes,
                            f'episode {ep_num}: len={ep["length"]} '
                            f'reward={ep["total_reward"]:.1f}')

    fns = [bind(dmain.make_env, config, 0)]
    driver = embodied.Driver(fns, parallel=False)
    driver.on_step(logfn)
    policy = lambda *a: agent.policy(*a, mode='eval')
    driver.reset(agent.init_policy)

    t0 = time.time()
    if progress_cb:
        progress_cb(0, num_episodes, 'starting rollouts...')
    while episode_count[0] < num_episodes:
        driver(policy, steps=100)
    try:
        driver.close()
    except Exception:
        pass

    index = {
        'logdir': str(logdir),
        'ckpt': model.ckpt,
        'knobs': dict(knobs),
        'seed': int(seed),
        'num_episodes': int(num_episodes),
        'env_seed': env_seed,
        'world_seed': world_seed,
        'area': list(area) if area else None,
        'island': island_meta,
        'fixed_layout': bool(knobs.get('fixed_layout', False)),
        'fixed_seed': bool(knobs.get('fixed_seed', False)),
        'action_names': ACTION_NAMES,
        'elapsed_sec': round(time.time() - t0, 1),
        'episodes': summaries,
    }
    index_path(cache_dir).write_text(json.dumps(index, indent=2, default=str))
    return str(cache_dir), index


# ============================================================================
# Viewer-side rendering helpers (pure: no Streamlit, so they are unit-testable)
# ============================================================================

VITAL_COLORS = {'health': '#d62728', 'food': '#ff7f0e',
                'drink': '#1f77b4', 'energy': '#bcbd22'}


def vitals_at(ep, step):
    """{vital: value} at a step (clamped), for whichever vitals were recorded."""
    out = {}
    for v in VITALS:
        arr = ep.get(f'vital_{v}')
        if arr is not None and len(arr):
            out[v] = int(arr[int(np.clip(step, 0, len(arr) - 1))])
    return out


def draw_trajectory(ep, step, area=None):
    """Trajectory in world-tile space, colored by time, current step marked.
    Returns a matplotlib Figure (caller is responsible for closing it)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pos = np.asarray(ep['player_pos'], dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    t = np.arange(len(pos))
    ax.plot(pos[:, 0], pos[:, 1], '-', color='0.75', lw=1.0, zorder=1)
    ax.scatter(pos[:, 0], pos[:, 1], c=t, cmap='viridis', s=10, zorder=2)
    ax.scatter([pos[0, 0]], [pos[0, 1]], c='lime', s=120, marker='*',
               edgecolors='k', zorder=4, label='start')
    s = int(np.clip(step, 0, len(pos) - 1))
    ax.scatter([pos[s, 0]], [pos[s, 1]], c='red', s=140, marker='o',
               edgecolors='k', zorder=5, label=f'step {s}')
    if area:
        ax.set_xlim(-0.5, area[0] - 0.5)
        ax.set_ylim(area[1] - 0.5, -0.5)  # invert y to match world orientation
    else:
        ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xlabel('x (tiles)')
    ax.set_ylabel('y (tiles)')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig


def list_runs(logdir, cache_root=None):
    """Enumerate previously-generated cached runs for a logdir (for a 'reload
    past run' picker). Returns list of (cache_dir, index) newest-first."""
    root = pathlib.Path(cache_root) if cache_root else CACHE_ROOT
    base = root / pathlib.Path(logdir).name
    out = []
    if base.exists():
        for d in base.iterdir():
            idx = load_index(d)
            if idx:
                out.append((str(d), idx))
    out.sort(key=lambda x: x[1].get('elapsed_sec', 0), reverse=True)
    return out


def available_logdirs(logdir_root=None):
    """List logdirs under ./logdir that have both a config.yaml and ckpt/latest."""
    root = pathlib.Path(logdir_root) if logdir_root else (REPO / 'logdir')
    out = []
    if root.exists():
        for d in sorted(root.iterdir()):
            if (d / 'config.yaml').exists() and (d / 'ckpt' / 'latest').exists():
                out.append(str(d))
    return out
