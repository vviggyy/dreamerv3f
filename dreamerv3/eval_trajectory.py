"""
Trajectory recording evaluation script for interpretability analysis.
Records agent positions and world model activations during evaluation.

Usage:
  python dreamerv3/main.py \
    --configs crafter \
    --logdir ./logdir/crafter_run1 \
    --run.script eval_trajectory \
    --run.from_checkpoint ./logdir/crafter_run1/checkpoint.ckpt \
    --env.crafter.seed 42 \
    --eval_trajectory.num_episodes 5 \
    --eval_trajectory.save_path ./trajectories
"""

from collections import defaultdict
from functools import partial as bind
import pickle

import elements
import embodied
import numpy as np


def eval_trajectory(make_agent, make_env, make_logger, args):
  assert args.from_checkpoint, "Must provide --run.from_checkpoint"

  agent = make_agent()
  logger = make_logger()

  logdir = elements.Path(args.logdir)
  logdir.mkdir()
  print('Logdir', logdir)

  # Trajectory storage
  save_path = elements.Path(getattr(args, 'eval_trajectory', {}).get(
      'save_path', str(logdir / 'trajectories')))
  save_path.mkdir()
  num_episodes = getattr(args, 'eval_trajectory', {}).get('num_episodes', 5)

  print(f'Will record {num_episodes} episodes to {save_path}')

  # Storage for current episode
  episode_data = defaultdict(list)
  completed_episodes = []
  episode_count = [0]  # Use list for mutability in closure

  def logfn(tran, worker):
    if worker != 0:
      return

    # Start new episode
    if tran['is_first']:
      episode_data.clear()

    # Record trajectory data
    episode_data['player_pos'].append(tran['player_pos'].copy())
    episode_data['reward'].append(float(tran['reward']))
    episode_data['action'].append(tran.get('action', 0))
    episode_data['image'].append(tran['image'].copy())

    # Record world model activations if available
    if 'dyn/deter' in tran:
      episode_data['deter'].append(tran['dyn/deter'].copy())
    if 'dyn/stoch' in tran:
      episode_data['stoch'].append(tran['dyn/stoch'].copy())

    # End of episode
    if tran['is_last']:
      episode_count[0] += 1
      ep_num = episode_count[0]

      # Convert lists to arrays
      ep_result = {k: np.array(v) for k, v in episode_data.items()}
      ep_result['episode'] = ep_num
      ep_result['length'] = len(episode_data['player_pos'])
      ep_result['total_reward'] = sum(episode_data['reward'])

      completed_episodes.append(ep_result)
      print(f"Episode {ep_num}: length={ep_result['length']}, "
            f"reward={ep_result['total_reward']:.1f}")

      # Save intermediate results
      if ep_num <= num_episodes:
        ep_file = save_path / f'episode_{ep_num:03d}.pkl'
        with open(str(ep_file), 'wb') as f:
          pickle.dump(ep_result, f)
        print(f"  Saved to {ep_file}")

  # Create environment
  env = make_env(0)
  env_seed = getattr(env, '_seed', None)  # Try to get Crafter seed
  fns = [bind(make_env, i) for i in range(1)]  # Single env for trajectory recording
  driver = embodied.Driver(fns, parallel=False)
  driver.on_step(logfn)

  # Load checkpoint
  cp = elements.Checkpoint()
  cp.agent = agent
  cp.load(args.from_checkpoint, keys=['agent'])

  print('Start trajectory recording')
  policy = lambda *args: agent.policy(*args, mode='eval')
  driver.reset(agent.init_policy)

  # Run until we have enough episodes
  while episode_count[0] < num_episodes:
    driver(policy, steps=100)

  # Save all episodes together with metadata
  all_file = save_path / 'all_episodes.pkl'
  save_data = {
      'episodes': completed_episodes,
      'env_seed': env_seed,
      'task': 'crafter',  # For now assume crafter
  }
  with open(str(all_file), 'wb') as f:
    pickle.dump(save_data, f)
  print(f"\nSaved all {len(completed_episodes)} episodes to {all_file}")
  if env_seed:
    print(f"Environment seed: {env_seed}")

  # Print summary
  print("\n=== Trajectory Recording Summary ===")
  print(f"Episodes recorded: {len(completed_episodes)}")
  lengths = [ep['length'] for ep in completed_episodes]
  rewards = [ep['total_reward'] for ep in completed_episodes]
  print(f"Avg length: {np.mean(lengths):.1f} +/- {np.std(lengths):.1f}")
  print(f"Avg reward: {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")

  if completed_episodes and 'deter' in completed_episodes[0]:
    print(f"Deter shape per step: {completed_episodes[0]['deter'].shape[1:]}")
    print(f"Stoch shape per step: {completed_episodes[0]['stoch'].shape[1:]}")
  else:
    print("Note: World model activations not recorded (enable replay_context)")

  logger.close()
