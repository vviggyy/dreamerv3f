"""
Interoceptive state probe: fix the visual scene, vary the status vitals shown in
the image (health/food/drink/energy), and measure how the policy shifts.

Crafter draws the bottom status bar from `env._env._player.inventory` (a dict of
ints 0-9) and copies it into the observation (crafter.py:_render_egocentric ->
`result[-inv_rows:] = raw_image[-inv_rows:]`). So we can manipulate the *state*
and re-render, rather than editing pixels — the terrain is untouched and the
icons are the exact ones the model trained on (guaranteed on-distribution).

Procedure (fresh-carry, single-frame stimulus probe):
  1. Drive the env with the real policy; every `warmup_steps` steps, capture the
     current scene as a probe frame.
  2. At each frame, for every combination in the factorial grid over `vitals`
     (default food x drink x energy, each over `values`), overwrite the player's
     inventory, re-render the observation, and run `agent.policy` with a FRESH
     carry (is_first=True, so the recurrent state resets — the decision depends
     on this single frame only).
  3. Record the action distribution (softmax of policy_logits over the 17
     crafter actions) and the value estimate for each (frame, vital-combo).

Readouts favor the *distribution shift + value* over the argmax action: crafter
eating/drinking is the single "do" action while facing a cow/water, so a lone
frame won't cleanly flip to "eat". The smooth signals — P(do), P(sleep), value —
as functions of the vitals are what to read. NOTE: `value` is the value head's
raw pred() (normalized output space), so read it for its *shape* vs vitals, not
absolute return.

Requires the `agent.probe_policy` flag (set by main.py for this script), which
makes agent.policy() emit `policy_logits` and `value`.

Usage:
  python dreamerv3/main.py \
    --configs crafter_small size25m --logdir ./logdir/my_run \
    --script state_probe \
    --run.from_checkpoint ./logdir/my_run/ckpt/TIMESTAMP_DIR \
    --state_probe.save_path ./logdir/my_run/state_probe \
    --state_probe.num_frames 8 --state_probe.warmup_steps 20 \
    --state_probe.vitals food,drink,energy --state_probe.levels 0,3,6,9 \
    --seed 45 --jax.platform cpu

Replot only (no agent/env):
  --state_probe.from_pkl ./logdir/my_run/state_probe/state_probe_results.pkl
"""

import itertools
import pickle
from functools import partial as bind
from pathlib import Path

import elements
import matplotlib.pyplot as plt
import numpy as np


ALL_VITALS = ['health', 'food', 'drink', 'energy']


def _softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _unwrap_crafter(env):
    """Descend the wrapper stack to the Crafter wrapper (has _render_egocentric).

    embodied wrappers delegate via __getattr__, so most attributes are reachable
    on the outer env directly, but we want the object that actually owns the
    crafter internals so mutations are unambiguous."""
    e = env
    for _ in range(16):
        if hasattr(e, '__dict__') and '_render_egocentric' in dir(type(e)):
            return e
        inner = getattr(e, 'env', None)
        if inner is None or inner is e:
            break
        e = inner
    return e  # best effort (attribute delegation still works via __getattr__)


def _action_names(n):
    try:
        from crafter import constants
        names = list(constants.actions)
        if len(names) == n:
            return names
    except Exception:
        pass
    return [f'a{i}' for i in range(n)]


def state_probe(make_agent, make_env, make_logger, args):
    cfg = args.state_probe
    logdir = elements.Path(args.logdir)
    save_dir = Path(cfg.save_path or str(logdir / 'state_probe'))

    # Fast path: replot from a saved results pkl (no agent/env).
    if cfg.from_pkl:
        save_dir = Path(cfg.save_path or str(Path(cfg.from_pkl).parent))
        replot_from_pkl(cfg.from_pkl, save_dir)
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    vitals = [v.strip() for v in cfg.vitals.split(',') if v.strip()]
    assert all(v in ALL_VITALS for v in vitals), \
        f"vitals must be subset of {ALL_VITALS}, got {vitals}"
    values = [int(v.strip()) for v in cfg.levels.split(',') if v.strip() != '']
    combos = list(itertools.product(values, repeat=len(vitals)))
    print(f"Sweeping {vitals} over {values} -> {len(combos)} combos/frame, "
          f"{cfg.num_frames} frames")

    # 1. Agent + checkpoint (params only, skip opt/ — see dream_seed_ablation).
    print("Creating agent...")
    agent = make_agent()
    logger = make_logger()
    from_ckpt = args.from_checkpoint
    if cfg.load_checkpoint and not from_ckpt:
        latest = logdir / 'ckpt' / 'latest'
        if latest.exists():
            from_ckpt = str(logdir / 'ckpt' / latest.read_text().strip())
            print(f"Auto-detected checkpoint: {from_ckpt}")
    assert from_ckpt, "No checkpoint (set --run.from_checkpoint or ckpt/latest)"
    with open(Path(from_ckpt) / 'agent.pkl', 'rb') as f:
        agent.load(pickle.load(f), regex=r'^(?!opt/)')
    print(f"Loaded checkpoint (params only): {from_ckpt}")

    # 2. Single env (fixed world). random_spawn etc. come from config.env.
    env = make_env(0, fixed_seed=True)
    crafter = _unwrap_crafter(env)
    try:
        env_seed = env._seed
    except (AttributeError, ValueError):
        env_seed = None
    try:
        area = tuple(env._env._world._mat_map.shape)
    except (AttributeError, ValueError):
        area = None
    ego = getattr(env, '_egocentric_view', None)
    obs_keys = list(agent.obs_space.keys())

    def mkact(a, reset):
        return {'action': np.asarray(a, np.int32),
                'reset': np.asarray(reset, bool)}

    def to_agent_obs(obs, image=None, is_first=True):
        o = {}
        for k in obs_keys:
            if k == 'image' and image is not None:
                v = image
            elif k == 'is_first':
                v = np.asarray(is_first, bool)
            elif k in ('is_last', 'is_terminal'):
                v = np.asarray(False, bool)
            elif k == 'reward':
                v = np.asarray(0.0, np.float32)
            else:
                v = np.asarray(obs[k])
            o[k] = np.asarray(v)[None]  # add batch dim
        return o

    def render_vitals(vital_dict):
        """Re-render the current scene with the given vitals overwritten
        (restoring the real inventory afterward). Returns a uint8 image."""
        player = crafter._env._player
        saved = {k: player.inventory.get(k) for k in vital_dict}
        for k, val in vital_dict.items():
            player.inventory[k] = int(val)
        raw = crafter._env._obs()                      # standard crafter render
        img = crafter._render_egocentric(raw) if ego else raw
        for k, val in saved.items():                   # restore
            player.inventory[k] = val
        return np.asarray(img, np.uint8)

    # 3. Drive the real policy; probe every warmup_steps steps.
    print("Driving policy and probing...")
    carry_real = agent.init_policy(1)
    obs = env.step(mkact(0, True))
    n_actions = None
    frames = []      # per-frame: {image, player_pos, base_vitals, probs, values}
    step, nf = 0, 0
    max_steps = cfg.num_frames * max(cfg.warmup_steps, 1) * 5
    while nf < cfg.num_frames and step < max_steps:
        if step % max(cfg.warmup_steps, 1) == 0:
            player = crafter._env._player
            base_vitals = {k: int(player.inventory.get(k, 0)) for k in ALL_VITALS}
            probs_list, value_list = [], []
            for combo in combos:
                vd = dict(zip(vitals, combo))
                aobs = to_agent_obs(obs, image=render_vitals(vd), is_first=True)
                carry = agent.init_policy(1)
                carry, acts, outs = agent.policy(carry, aobs, mode='eval')
                logits = np.asarray(outs['policy_logits'])[0]
                n_actions = logits.shape[-1]
                probs_list.append(_softmax(logits))
                value_list.append(float(np.asarray(outs['value'])[0]))
            frames.append({
                'image': np.asarray(obs['image'], np.uint8),
                'player_pos': np.asarray(obs['player_pos'], np.float32),
                'base_vitals': base_vitals,
                'probs': np.stack(probs_list),      # (n_combos, n_actions)
                'values': np.asarray(value_list),   # (n_combos,)
            })
            nf += 1
            print(f"  probed frame {nf}/{cfg.num_frames} at "
                  f"pos {np.asarray(obs['player_pos']).tolist()} "
                  f"vitals {base_vitals}")
        # advance the real policy one step
        aobs = to_agent_obs(obs, is_first=(step == 0))
        carry_real, acts, _ = agent.policy(carry_real, aobs, mode='eval')
        a = int(np.asarray(acts['action'])[0])
        obs = env.step(mkact(a, bool(np.asarray(obs['is_last']))))
        step += 1
    env.close()

    action_names = _action_names(n_actions)
    results = {
        'vitals': vitals, 'levels': values, 'combos': np.asarray(combos),
        'action_names': action_names, 'n_actions': n_actions,
        'probs': np.stack([f['probs'] for f in frames]),   # (F, C, A)
        'values': np.stack([f['values'] for f in frames]),  # (F, C)
        'images': np.stack([f['image'] for f in frames]),
        'player_pos': np.stack([f['player_pos'] for f in frames]),
        'base_vitals': [f['base_vitals'] for f in frames],
        'metadata': {'env_seed': env_seed, 'area': area, 'task': 'crafter'},
    }
    with open(save_dir / 'state_probe_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {save_dir / 'state_probe_results.pkl'}")

    _print_summary(results)
    print("\nGenerating plots...")
    make_plots(results, save_dir)

    try:
        from .run_info import log_run_info
        log_run_info(
            save_dir, 'state_probe',
            args={'vitals': cfg.vitals, 'levels': cfg.levels,
                  'num_frames': cfg.num_frames, 'warmup_steps': cfg.warmup_steps,
                  'from_checkpoint': from_ckpt},
            outputs=['state_probe_results.pkl', 'value_vs_vitals.png',
                     'action_probs_vs_vitals.png', 'value_heatmaps.png',
                     'probe_frames.png'],
            extra={'n_frames': int(len(frames)), 'n_combos': int(len(combos)),
                   'n_actions': int(n_actions)})
    except Exception as e:
        print(f"  (run_info logging skipped: {e})")

    logger.close()
    print("Done.")


def _grids(results):
    """Reshape flat (F, C, ...) records into per-vital factorial grids.

    Returns value_grid (F, v1, v2, ...) and probs_grid (F, v1, v2, ..., A)."""
    vitals, values = results['vitals'], results['levels']
    F = results['values'].shape[0]
    gshape = tuple(len(values) for _ in vitals)
    value_grid = results['values'].reshape((F, *gshape))
    probs_grid = results['probs'].reshape((F, *gshape, results['n_actions']))
    return value_grid, probs_grid


def _marginal(grid, vital_axis, n_vitals):
    """Mean over the frame axis (0) and every vital axis except `vital_axis`.

    grid axes: 0=frame, 1..n_vitals=vitals, (optional trailing action axis kept).
    Returns (mean, std) each shaped (len(values), [n_actions])."""
    keep = 1 + vital_axis
    reduce_axes = tuple(a for a in range(grid.ndim)
                        if a != keep and a < 1 + n_vitals) + (0,)
    reduce_axes = tuple(sorted(set(reduce_axes)))
    return grid.mean(axis=reduce_axes), grid.std(axis=reduce_axes)


def _print_summary(results):
    vitals, values = results['vitals'], results['levels']
    value_grid, _ = _grids(results)
    print("\n[summary] mean value vs each vital (marginal over frames + others):")
    for i, vit in enumerate(vitals):
        mean, _ = _marginal(value_grid, i, len(vitals))
        curve = "  ".join(f"{v}:{m:+.2f}" for v, m in zip(values, mean))
        print(f"  {vit:<7} {curve}   (Δ={mean.max() - mean.min():.2f})")


def make_plots(results, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    _plot_value_vs_vitals(results, save_dir)
    _plot_action_probs_vs_vitals(results, save_dir)
    _plot_value_heatmaps(results, save_dir)
    _plot_probe_frames(results, save_dir)


def _plot_value_vs_vitals(results, save_dir):
    vitals, values = results['vitals'], results['levels']
    value_grid, _ = _grids(results)
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for i, vit in enumerate(vitals):
        mean, std = _marginal(value_grid, i, len(vitals))
        ax.plot(values, mean, '-o', label=vit, markersize=4)
        ax.fill_between(values, mean - std, mean + std, alpha=0.15)
    ax.set_xlabel('Vital value shown in status bar (0-9)')
    ax.set_ylabel('Value estimate (head pred, normalized space)')
    ax.set_title('Value vs interoceptive vital\n(marginal over frames + other vitals)')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = save_dir / 'value_vs_vitals.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out.name}")


def _plot_action_probs_vs_vitals(results, save_dir, top_k=6):
    vitals, values = results['vitals'], results['levels']
    names = results['action_names']
    _, probs_grid = _grids(results)
    n = len(vitals)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.6), squeeze=False)
    for i, vit in enumerate(vitals):
        ax = axes[0, i]
        mean, _ = _marginal(probs_grid, i, n)        # (len(values), A)
        rng = mean.max(axis=0) - mean.min(axis=0)     # variation per action
        top = np.argsort(rng)[::-1][:top_k]
        for a in top:
            ax.plot(values, mean[:, a], '-o', markersize=3,
                    label=f'{names[a]} (Δ{rng[a]:.2f})')
        ax.set_title(f'P(action) vs {vit}')
        ax.set_xlabel(f'{vit} value (0-9)')
        if i == 0:
            ax.set_ylabel('P(action)  (marginal)')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle('Action-probability shift by interoceptive vital '
                 f'(top {top_k} most-varying actions per vital)', fontsize=12)
    fig.tight_layout()
    out = save_dir / 'action_probs_vs_vitals.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out.name}")


def _plot_value_heatmaps(results, save_dir):
    vitals, values = results['vitals'], results['levels']
    value_grid, _ = _grids(results)          # (F, v1, v2, ...)
    n = len(vitals)
    if n < 2:
        return
    pairs = list(itertools.combinations(range(n), 2))
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.8 * len(pairs), 4.4),
                             squeeze=False)
    for pi, (i, j) in enumerate(pairs):
        ax = axes[0, pi]
        # mean over frame axis + all vital axes except i, j
        reduce_axes = tuple([0] + [1 + a for a in range(n) if a not in (i, j)])
        hm = value_grid.mean(axis=reduce_axes)        # (len(values), len(values))
        if i > j:
            hm = hm.T
        im = ax.imshow(hm, origin='lower', aspect='auto', cmap='viridis')
        ax.set_xticks(range(len(values)), values)
        ax.set_yticks(range(len(values)), values)
        ax.set_xlabel(vitals[j]); ax.set_ylabel(vitals[i])
        ax.set_title(f'value: {vitals[i]} x {vitals[j]}')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle('Value over vital pairs (marginal over frames + other vitals)',
                 fontsize=12)
    fig.tight_layout()
    out = save_dir / 'value_heatmaps.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out.name}")


def _plot_probe_frames(results, save_dir):
    imgs = results['images']
    pos = results['player_pos']
    base = results['base_vitals']
    F = imgs.shape[0]
    ncol = min(F, 8)
    nrow = int(np.ceil(F / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 2.6 * nrow),
                             squeeze=False)
    for k in range(nrow * ncol):
        ax = axes[k // ncol, k % ncol]
        if k < F:
            ax.imshow(imgs[k])
            bv = base[k]
            ax.set_title(f"#{k}  pos {pos[k].astype(int).tolist()}\n"
                         f"f{bv['food']} d{bv['drink']} e{bv['energy']}",
                         fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('Probe frames (real vitals shown; the sweep overwrites these)',
                 fontsize=11)
    fig.tight_layout()
    out = save_dir / 'probe_frames.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out.name}")


def replot_from_pkl(pkl_path, save_dir):
    pkl_path = Path(pkl_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Replotting from {pkl_path}")
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)
    _print_summary(results)
    print("\nGenerating plots...")
    make_plots(results, save_dir)
    print(f"Done. Plots written to {save_dir}")
