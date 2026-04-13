"""
Replay saved trajectory observations through a (possibly untrained) agent
to record activations. Used as a control: does spatial decoding work with
random weights, or only with learned representations?

Usage:
  # Untrained control (random weights):
  python dreamerv3/main.py \
    --configs crafter_small size25m \
    --logdir ./logdir/my_run \
    --script replay_activations \
    --replay_activations.source_data ./logdir/my_run/trajectories \
    --replay_activations.save_path ./logdir/my_run/untrained_activations \
    --replay_activations.load_checkpoint False \
    --jax.platform cpu

  # Trained replay (same obs, trained weights):
  python dreamerv3/main.py \
    --configs crafter_small size25m \
    --logdir ./logdir/my_run \
    --script replay_activations \
    --run.from_checkpoint ./logdir/my_run/ckpt/TIMESTAMP \
    --replay_activations.source_data ./logdir/my_run/trajectories \
    --replay_activations.save_path ./logdir/my_run/trained_replay \
    --jax.platform cpu
"""

import os
import pickle
import pathlib
import signal
import sys
from collections import defaultdict

import elements
import numpy as np


def _load_episodes(source_path):
    """Load episodes from all_episodes.pkl or a directory of episode_NNN.pkl."""
    source = pathlib.Path(source_path)
    metadata = {}

    if source.is_file() and source.name == 'all_episodes.pkl':
        with open(source, 'rb') as f:
            data = pickle.load(f)
        episodes = data['episodes']
        for k in ('env_seed', 'world_seed', 'fixed_seed', 'task', 'area'):
            if k in data:
                metadata[k] = data[k]
        return episodes, metadata

    # Try all_episodes.pkl inside directory
    all_pkl = source / 'all_episodes.pkl'
    if all_pkl.exists():
        with open(all_pkl, 'rb') as f:
            data = pickle.load(f)
        episodes = data['episodes']
        for k in ('env_seed', 'world_seed', 'fixed_seed', 'task', 'area'):
            if k in data:
                metadata[k] = data[k]
        return episodes, metadata

    # Fall back to individual episode_NNN.pkl files
    ep_files = sorted(source.glob('episode_*.pkl'))
    assert ep_files, f"No episode files found in {source}"
    episodes = []
    for ep_file in ep_files:
        with open(ep_file, 'rb') as f:
            episodes.append(pickle.load(f))
    return episodes, metadata


def replay_activations(make_agent, make_logger, args):
    ra_config = args.replay_activations
    source_data = ra_config.source_data
    assert source_data, "Must set --replay_activations.source_data"

    save_path = elements.Path(
        ra_config.save_path or str(elements.Path(args.logdir) / 'replay_activations'))
    save_path.mkdir()

    load_checkpoint = ra_config.load_checkpoint
    max_episodes = ra_config.max_episodes  # 0 = all

    # 1. Load source episodes
    print(f'Loading source episodes from {source_data}')
    episodes, src_metadata = _load_episodes(source_data)
    if max_episodes > 0:
        episodes = episodes[:max_episodes]
    print(f'Loaded {len(episodes)} episodes')

    # 2. Create agent (random weights initially)
    print('Creating agent...')
    agent = make_agent()

    # 3. Optionally load checkpoint
    if load_checkpoint:
        logdir = elements.Path(args.logdir)
        from_checkpoint = args.from_checkpoint
        if not from_checkpoint:
            latest_file = logdir / 'ckpt' / 'latest'
            assert latest_file.exists(), (
                f"No --run.from_checkpoint given and no ckpt/latest found in "
                f"{logdir}. Set --replay_activations.load_checkpoint False "
                f"for untrained control.")
            latest_name = latest_file.read_text().strip()
            from_checkpoint = str(logdir / 'ckpt' / latest_name)
            print(f'Auto-detected checkpoint: {from_checkpoint}')

        ckpt_file = pathlib.Path(from_checkpoint) / 'agent.pkl'
        with open(ckpt_file, 'rb') as f:
            ckpt_data = pickle.load(f)
        agent.load(ckpt_data, regex=r'^(?!opt/)')
        print(f'Loaded checkpoint from {from_checkpoint}')
    else:
        print('Using UNTRAINED (random) weights')

    # 4. Run episodes
    completed_episodes = []

    def _save_results(tag=''):
        if not completed_episodes:
            print(f"WARNING: No completed episodes to save{tag}.")
            return
        all_file = save_path / 'all_episodes.pkl'
        save_data = {
            'episodes': completed_episodes,
            'source_data': str(source_data),
            'load_checkpoint': load_checkpoint,
        }
        save_data.update(src_metadata)
        with open(str(all_file), 'wb') as f:
            pickle.dump(save_data, f)
        print(f"\nSaved {len(completed_episodes)} episodes to {all_file}{tag}")

    def _sigterm_handler(signum, _frame):
        print(f"\nReceived signal {signum}, saving partial results...")
        _save_results(tag=' (partial, interrupted by signal)')
        sys.exit(1)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    # Determine obs keys the agent expects (non-log/ keys from obs_space)
    obs_keys = sorted(agent.obs_space.keys())

    carry = agent.init_policy(1)

    try:
        for ep_idx, ep in enumerate(episodes):
            T = ep['length']
            print(f"\n--- Episode {ep_idx + 1}/{len(episodes)} "
                  f"(length={T}) ---")

            episode_data = defaultdict(list)

            for t in range(T):
                # Build obs dict with batch dim
                obs = {}
                obs['image'] = ep['image'][t][np.newaxis]  # (1, H, W, 3)
                obs['reward'] = np.array([ep['reward'][t]], dtype=np.float32)
                obs['is_first'] = np.array([t == 0])
                obs['is_last'] = np.array([t == T - 1])
                obs['is_terminal'] = np.array([t == T - 1])
                obs['player_pos'] = ep['player_pos'][t][np.newaxis].astype(
                    np.float32)  # (1, 2)

                # Call agent policy
                carry, _acts, outs = agent.policy(carry, obs, mode='eval')

                # Override prevact with saved action so the next step uses the
                # original trajectory's action, not the agent's sampled action.
                saved_action = int(ep['action'][t])
                # After _split(), carry is a 4-tuple of pytrees with list leaves.
                # Element [3] is the prevact: {'action': [np.array(...)]}
                enc_c, dyn_c, dec_c, _ = carry
                carry = (enc_c, dyn_c, dec_c,
                         {'action': [np.array([saved_action], dtype=np.int32)]})

                # Collect trajectory data
                episode_data['image'].append(ep['image'][t])
                episode_data['player_pos'].append(ep['player_pos'][t])
                episode_data['action'].append(saved_action)
                episode_data['reward'].append(float(ep['reward'][t]))
                if 'player_facing' in ep:
                    episode_data['player_facing'].append(
                        ep['player_facing'][t])
                if 'achievements' in ep and t < len(ep['achievements']):
                    episode_data['achievements'].append(
                        ep['achievements'][t])

                # Collect activations
                for key, val in outs.items():
                    if key.startswith('activations/'):
                        layer_name = key[len('activations/'):]
                        store_key = f'act/{layer_name}'
                        # val is list-of-arrays after _split; take element 0
                        v = val[0] if isinstance(val, list) else val
                        v = np.asarray(v)
                        # Remove batch dim if present
                        if v.ndim > 0 and v.shape[0] == 1:
                            v = v[0]
                        episode_data[store_key].append(v)

                # Legacy deter/stoch
                if 'activations/dyn/deter' in outs:
                    v = outs['activations/dyn/deter']
                    v = v[0] if isinstance(v, list) else v
                    v = np.asarray(v)
                    if v.ndim > 0 and v.shape[0] == 1:
                        v = v[0]
                    episode_data['deter'].append(v)
                if 'activations/dyn/stoch' in outs:
                    v = outs['activations/dyn/stoch']
                    v = v[0] if isinstance(v, list) else v
                    v = np.asarray(v)
                    if v.ndim > 0 and v.shape[0] == 1:
                        v = v[0]
                    episode_data['stoch'].append(v)

            # Finalize episode
            ep_result = {}
            for k, v in episode_data.items():
                if k == 'achievements':
                    ep_result[k] = v
                else:
                    ep_result[k] = np.array(v)

            ep_result['episode'] = ep_idx + 1
            ep_result['length'] = T
            ep_result['total_reward'] = sum(episode_data['reward'])
            ep_result['layer_names'] = sorted(
                k[len('act/'):] for k in ep_result if k.startswith('act/'))
            if 'achievements' in ep and ep['achievements']:
                final = ep['achievements'][-1] if isinstance(
                    ep['achievements'][-1], dict) else {}
                ep_result['final_achievements'] = final
            elif 'final_achievements' in ep:
                ep_result['final_achievements'] = ep['final_achievements']
            else:
                ep_result['final_achievements'] = {}

            completed_episodes.append(ep_result)

            # Save per-episode pkl
            ep_file = save_path / f'episode_{ep_idx + 1:03d}.pkl'
            with open(str(ep_file), 'wb') as f:
                pickle.dump(ep_result, f)

            print(f"  Episode {ep_idx + 1}: length={T}, "
                  f"reward={ep_result['total_reward']:.1f}, "
                  f"layers={ep_result['layer_names']}")

    except Exception as e:
        print(f"\nERROR during replay: {e}")
        _save_results(tag=' (partial, crashed)')
        raise

    _save_results()

    # Summary
    print("\n=== Replay Activations Summary ===")
    print(f"Episodes: {len(completed_episodes)}")
    print(f"Checkpoint loaded: {load_checkpoint}")
    if completed_episodes:
        lengths = [ep['length'] for ep in completed_episodes]
        print(f"Avg length: {np.mean(lengths):.1f}")
        ep0 = completed_episodes[0]
        if ep0.get('layer_names'):
            print(f"Layers recorded: {ep0['layer_names']}")
            for ln in ep0['layer_names']:
                arr = ep0[f'act/{ln}']
                print(f"  act/{ln}: {arr.shape[1:]}")
