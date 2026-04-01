"""
Lightweight run provenance logger.

Each pipeline stage calls `log_run_info(save_dir, stage, args, outputs)`
which appends a JSON entry to `<save_dir>/run_info.json`.  The file is a
JSON array so it's both human-readable and machine-parseable.

Usage:
    from run_info import log_run_info
    log_run_info(
        save_dir=Path('./results'),
        stage='decode_position',
        args=vars(parsed_args),          # or any dict
        outputs=['occupancy_vs_error_deter.png', 'decode_results.pkl'],
        extra={'n_episodes': 50, 'grid': '32x32'},
    )
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _git_sha():
    """Return short git SHA of current HEAD, or 'unknown'."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL, cwd=Path(__file__).parent
        ).decode().strip()
    except Exception:
        return 'unknown'


def log_run_info(save_dir, stage, args=None, outputs=None, extra=None):
    """Append a run entry to save_dir/run_info.json.

    Parameters
    ----------
    save_dir : str or Path
        Directory where run_info.json lives (created if needed).
    stage : str
        Pipeline stage name (e.g. 'decode_position', 'plot_trajectories').
    args : dict, optional
        Arguments/settings used for this run.  Non-serializable values
        are converted to strings.
    outputs : list of str, optional
        List of output filenames produced by this run.
    extra : dict, optional
        Any additional metadata (metrics, data source paths, etc.).
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    run_file = save_dir / 'run_info.json'

    # Build entry
    entry = {
        'stage': stage,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'git_sha': _git_sha(),
        'command': ' '.join(sys.argv),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID', None),
    }
    if args is not None:
        entry['args'] = _make_serializable(args)
    if outputs is not None:
        entry['outputs'] = outputs
    if extra is not None:
        entry['extra'] = _make_serializable(extra)

    # Load existing entries (or start fresh)
    entries = []
    if run_file.exists():
        try:
            with open(run_file) as f:
                entries = json.load(f)
            if not isinstance(entries, list):
                entries = [entries]
        except (json.JSONDecodeError, ValueError):
            entries = []

    entries.append(entry)

    with open(run_file, 'w') as f:
        json.dump(entries, f, indent=2)

    print(f"  Run info logged to {run_file}")


def _make_serializable(obj):
    """Recursively convert a dict/list so it's JSON-safe."""
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (int, float, bool, str, type(None))):
        return obj
    return str(obj)
