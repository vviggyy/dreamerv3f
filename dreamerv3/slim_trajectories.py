#!/usr/bin/env python
"""Slim per-episode trajectory pkls by dropping unused activation layers.

eval_trajectory writes one episode_NNN.pkl per episode carrying EVERY recorded
layer's activations (act/enc/*, act/pol/*, act/val/*, act/dyn/*) plus images and
scalars. Downstream analysis of RSSM representations only ever reads:
    deter, stoch                   (top-level; decode_position standard mode)
    act/dyn/deter, act/dyn/stoch   (tuning_curve / manifold_analysis)
    player_pos, player_facing, image, + scalar series
The big encoder CNN maps (act/enc/cnn0 is 49152-dim ~= 78 MB/ep) and the tiny
pol/val MLP acts are loaded-then-discarded by every one of those steps, so a
monolithic pickle forces a full ~145 MB/ep deserialize to use ~20 MB of it. A
pickle cannot be partially read, so the only way to cut load time/RAM is to
rewrite the files without the dead layers.

This drops the layers whose act/ key starts with any --drop-prefixes (default
act/enc/, act/pol/, act/val/), making every downstream load ~7x smaller/faster
and ~9x lighter on RAM (e.g. 33 GB -> ~4 GB for 100 episodes). IRREVERSIBLE for
the dropped layers: only layer-wise decoding (pipeline step 4) ever consumes
enc/*, so in the pipeline this runs as step 4b, AFTER step 4. In the pipeline it
is gated by the SLIM_ACTIVATIONS toggle (default true).

Dry-run by default; pass --apply to rewrite. Each file is written to a .tmp
sibling then os.replace'd over the original (atomic; a crash leaves the original
intact). Idempotent — re-running finds no matching keys to drop.

Usage:
  python dreamerv3/slim_trajectories.py --data DIR            # dry run
  python dreamerv3/slim_trajectories.py --data DIR --apply    # rewrite in place
  python dreamerv3/slim_trajectories.py --data DIR --apply \
      --drop-prefixes act/enc/,act/pol/,act/val/
"""
import argparse
import os
import pickle
from pathlib import Path

DEFAULT_DROP = ('act/enc/', 'act/pol/', 'act/val/')
# Keys that MUST survive for downstream decode/tuning/manifold — a file missing
# any of these is left untouched (it is not an activation episode, or is already
# slimmed past usefulness) rather than silently stripped.
REQUIRED = ('deter', 'stoch', 'act/dyn/deter', 'act/dyn/stoch', 'player_pos')


def human(n):
    for u in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024 or u == 'TB':
            return f'{n:.1f}{u}'
        n /= 1024


def key_nbytes(v):
    return int(getattr(v, 'nbytes', 0))


def main():
    ap = argparse.ArgumentParser(
        description='Drop unused activation layers from episode pkls.')
    ap.add_argument('--data', required=True,
                    help='trajectory dir containing episode_*.pkl')
    ap.add_argument('--apply', action='store_true',
                    help='actually rewrite the files (default: dry-run)')
    ap.add_argument('--drop-prefixes', default=','.join(DEFAULT_DROP),
                    help='comma-separated key prefixes to drop '
                         f'(default: {",".join(DEFAULT_DROP)})')
    ap.add_argument('--max-files', type=int, default=0,
                    help='limit number of episodes processed (0 = all; for testing)')
    args = ap.parse_args()

    data = Path(args.data)
    drops = tuple(p for p in args.drop_prefixes.split(',') if p)
    ep_files = sorted(data.glob('episode_*.pkl'))
    if args.max_files:
        ep_files = ep_files[:args.max_files]
    if not ep_files:
        print(f'No episode_*.pkl in {data} — nothing to slim.')
        return

    print(f'{"APPLY" if args.apply else "DRY RUN"} over {len(ep_files)} '
          f'episodes in {data}')
    print(f'Dropping keys with prefixes: {drops}\n')

    tot_before = tot_after = tot_dropped = 0
    n_skipped = n_rewritten = 0
    for i, f in enumerate(ep_files):
        on_disk = f.stat().st_size
        tot_before += on_disk
        with open(f, 'rb') as fh:
            ep = pickle.load(fh)

        missing = [k for k in REQUIRED if k not in ep]
        if missing:
            print(f'  SKIP {f.name}: missing required keys {missing}')
            n_skipped += 1
            tot_after += on_disk
            continue

        drop_keys = [k for k in ep if any(k.startswith(p) for p in drops)]
        dropped_bytes = sum(key_nbytes(ep[k]) for k in drop_keys)
        tot_dropped += dropped_bytes

        if args.apply and drop_keys:
            for k in drop_keys:
                del ep[k]
            tmp = f.with_suffix('.pkl.tmp')
            with open(tmp, 'wb') as fh:
                pickle.dump(ep, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, f)
            n_rewritten += 1
        tot_after += f.stat().st_size

        if i < 3 or i == len(ep_files) - 1:
            print(f'  {f.name}: on_disk={human(on_disk)}  '
                  f'dropping {len(drop_keys)} keys (~{human(dropped_bytes)} in-mem)'
                  + (f'  -> {human(f.stat().st_size)}' if args.apply else ''))

    print(f'\n{"=" * 60}')
    print(f'episodes:                     {len(ep_files)}  '
          f'(skipped {n_skipped}, rewritten {n_rewritten})')
    print(f'in-memory bytes dropped (est): {human(tot_dropped)}')
    print(f'on-disk before:               {human(tot_before)}')
    if args.apply:
        print(f'on-disk after:                {human(tot_after)}   '
              f'(freed {human(tot_before - tot_after)})')
    else:
        print('on-disk after:                (run with --apply to rewrite)')

    # Provenance (best-effort; mirrors the other pipeline scripts).
    try:
        from run_info import log_run_info
        log_run_info(data, 'slim_trajectories', vars(args),
                     extra={'episodes': len(ep_files),
                            'skipped': n_skipped,
                            'rewritten': n_rewritten,
                            'drop_prefixes': list(drops),
                            'bytes_before': tot_before,
                            'bytes_after': tot_after,
                            'applied': bool(args.apply)})
    except Exception as e:  # never fail the prune over logging
        print(f'  (run_info logging skipped: {e})')


if __name__ == '__main__':
    main()
