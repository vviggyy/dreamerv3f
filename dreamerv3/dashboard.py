"""
Interactive environment / behavior dashboard (Streamlit).

Load a trained net, dial in crafter env knobs (mob toggles, cow density,
random_spawn, fixed_layout, island_border, egocentric view, ...), roll the
policy out for N episodes, then scrub through each trajectory step-by-step --
seeing the agent's actual frame, the action it took, its vitals, the
achievement it just unlocked, and its path on the world.

Launch (inside an OOD remote-desktop session on a GPU partition so no
port-forward is needed -- the in-desktop browser hits localhost):

    PYTHONPATH=. streamlit run dreamerv3/dashboard.py

Then open http://localhost:8501 in the desktop's browser. See run_dashboard.sh.

Design notes
------------
  * The agent is loaded once via @st.cache_resource and stays resident on the
    GPU; only the env is rebuilt per "Generate" (knobs don't touch the agent).
  * Generation streams one pickle per episode to ./dashboard_cache/; the viewer
    lazily loads only the selected episode, so RAM stays ~one episode regardless
    of how many were generated. That cache folder persists -- reload past runs
    from the sidebar.
"""

import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# dashboard_backend sets up sys.path for the dreamerv3 package / embodied.
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from dreamerv3 import dashboard_backend as B


# ============================================================================
# helpers
# ============================================================================

def detect_default_device():
    """cuda if an NVIDIA GPU is visible, else cpu."""
    try:
        out = subprocess.run(['nvidia-smi', '-L'], capture_output=True,
                             text=True, timeout=5)
        if out.returncode == 0 and 'GPU 0' in out.stdout:
            return 'cuda'
    except Exception:
        pass
    return 'cpu'


@st.cache_resource(show_spinner=False)
def get_model(logdir, device):
    """Resident agent for (logdir, device). Cached across reruns."""
    return B.load_agent(logdir, device=device, seed=0)


@st.cache_data(show_spinner=False, max_entries=6)
def get_episode(cache_dir, ep_num):
    """Lazily load one episode pickle (bounded LRU keeps RAM in check)."""
    return B.load_episode(cache_dir, ep_num)


# Pure rendering helpers live in the backend (no Streamlit -> unit-testable).
draw_trajectory = B.draw_trajectory
draw_vitals_timeline = B.draw_vitals_timeline
draw_event_strip = B.draw_event_strip
vitals_at = B.vitals_at


# ============================================================================
# page
# ============================================================================

st.set_page_config(page_title='DreamerV3 behavior dashboard', layout='wide')
st.title('Environment → Behavior dashboard')
st.caption('Load a net · set env knobs · generate trajectories · scrub frames')

ss = st.session_state
ss.setdefault('run', None)  # dict: {cache_dir, index}

# ---------------------------------------------------------------- sidebar ----
with st.sidebar:
    st.header('1 · Model')
    logdirs = B.available_logdirs()
    if not logdirs:
        st.error('No logdirs with config.yaml + ckpt/latest found under ./logdir')
        st.stop()
    labels = [__import__('pathlib').Path(d).name for d in logdirs]
    sel = st.selectbox('Checkpoint (logdir)', range(len(logdirs)),
                       format_func=lambda i: labels[i])
    logdir = logdirs[sel]
    device = st.radio('Device', ['cuda', 'cpu'],
                      index=0 if detect_default_device() == 'cuda' else 1,
                      horizontal=True,
                      help='cuda needs a GPU on this node. cpu works anywhere '
                           'but generation is much slower.')

    with st.spinner('Loading agent…'):
        try:
            model = get_model(logdir, device)
            st.success(f'Loaded · ckpt {__import__("pathlib").Path(model.ckpt).name}')
        except Exception as e:
            st.error(f'Load failed: {e}')
            st.stop()

    st.header('2 · Env knobs')
    k = {}
    c1, c2 = st.columns(2)
    with c1:
        k['disable_mobs'] = st.toggle('disable_mobs', value=False,
                                      help='Scrub hostile mobs (cows are kept).')
        k['random_spawn'] = st.toggle('random_spawn', value=True)
        k['fixed_seed'] = st.toggle('fixed_seed', value=False,
                                    help='Same world every episode.')
    with c2:
        k['fixed_layout'] = st.toggle('fixed_layout', value=True,
                                      help='Frozen terrain shell, resampled '
                                           'resources/mobs per episode.')
        k['island_border'] = st.toggle('island_border', value=False)

    st.markdown('**Cow density**')
    k['cow_spawn_prob'] = st.slider('cow_spawn_prob', 0.0, 0.5, 0.01, 0.01,
                                    help='Per-tick spawn probability (stock 0.01).')
    k['cow_span_dist'] = st.slider('cow_span_dist', 1, 10, 5, 1,
                                   help='Min spawn distance from player (stock 5).')
    k['cow_target_scale'] = st.slider('cow_target_scale', 0.5, 5.0, 1.0, 0.5,
                                      help='Population cap multiplier (stock 1).')
    k['cow_worldgen_scale'] = st.slider('cow_worldgen_scale', 0.5, 5.0, 1.0, 0.5,
                                        help='Initial worldgen density (stock 1).')

    st.markdown('**View**')
    ego = st.select_slider('egocentric_view', options=[0, 5, 7, 9, 11], value=9,
                           help='0 = full world render; odd N = N×N egocentric.')
    k['egocentric_view'] = int(ego)

    if k['island_border']:
        k['island_fill'] = st.slider('island_fill', 0.3, 0.95, 0.72, 0.01)
        k['island_roughness'] = st.slider('island_roughness', 0.0, 0.5, 0.16, 0.01)

    st.header('3 · Generate')
    num_episodes = st.number_input('num_episodes', 1, 500, 16, 1)
    seed = st.number_input('seed', 0, 10_000, 45, 1)
    force = st.checkbox('force regen (ignore cache)', value=False)
    gen = st.button('▶ Generate trajectories', type='primary',
                    use_container_width=True)

    past = B.list_runs(logdir)
    if past:
        st.markdown('— or reload a past run —')
        pidx = st.selectbox(
            'Cached runs', range(len(past)),
            format_func=lambda i: (
                f"{len(past[i][1].get('episodes', []))} eps · "
                f"seed {past[i][1].get('seed')} · "
                f"cows {past[i][1].get('knobs', {}).get('cow_spawn_prob')}"),
            index=None, placeholder='pick a cached run')
        if st.button('↻ Load cached run', use_container_width=True) and pidx is not None:
            cd, idx = past[pidx]
            ss.run = {'cache_dir': cd, 'index': idx}
            ss.pop('ep_num', None)

# ---------------------------------------------------------------- generate ---
if gen:
    prog = st.progress(0.0, text='starting…')
    status = st.empty()

    def cb(done, total, msg):
        prog.progress(min(done / max(total, 1), 1.0),
                      text=f'{done}/{total} · {msg}')

    try:
        cache_dir, index = B.generate(
            model, k, num_episodes=int(num_episodes), seed=int(seed),
            progress_cb=cb, force=force)
        ss.run = {'cache_dir': cache_dir, 'index': index}
        ss.pop('ep_num', None)
        prog.progress(1.0, text='done')
        status.success(f'Generated {len(index["episodes"])} episodes '
                       f'({index.get("elapsed_sec", "?")}s) → {cache_dir}')
    except Exception as e:
        prog.empty()
        status.error(f'Generation failed: {e}')
        st.exception(e)

# ---------------------------------------------------------------- viewer -----
run = ss.get('run')
if not run:
    st.info('Pick a checkpoint, set knobs, and hit **Generate** — or load a '
            'cached run from the sidebar.')
    st.stop()

index = run['index']
cache_dir = run['cache_dir']
eps = index.get('episodes', [])
area = index.get('area')
if not eps:
    st.warning('This run has no episodes recorded.')
    st.stop()

# ---- run-level summary ----
lengths = [e['length'] for e in eps]
rewards = [e['reward'] for e in eps]
cods = [e['cause_of_death'] for e in eps]
with st.expander('Run summary', expanded=True):
    m = st.columns(5)
    m[0].metric('episodes', len(eps))
    m[1].metric('mean length', f'{np.mean(lengths):.0f}')
    m[2].metric('mean reward', f'{np.mean(rewards):.1f}')
    m[3].metric('max length', f'{np.max(lengths)}')
    survived = sum('survived' in c for c in cods)
    m[4].metric('survived', f'{survived}/{len(eps)}')
    from collections import Counter
    cod_counts = Counter(cods)
    st.caption('cause of death: ' + ' · '.join(
        f'{c}×{n}' for c, n in cod_counts.most_common()))
    st.caption('knobs: ' + ', '.join(f'{kk}={vv}' for kk, vv in index['knobs'].items()))

# ---- episode picker ----
def ep_label(e):
    ach = f"{e['n_achievements']} ach" if e['n_achievements'] else 'no ach'
    return (f"ep {e['episode']:03d} · len {e['length']} · "
            f"rew {e['reward']:.1f} · {ach} · {e['cause_of_death']}")

ep_nums = [e['episode'] for e in eps]
chosen = st.selectbox('Episode', range(len(eps)),
                      format_func=lambda i: ep_label(eps[i]),
                      key='ep_select')
ep_num = ep_nums[chosen]
ep = get_episode(cache_dir, ep_num)
T = int(ep['length'])

# ---- step scrubber ----
events = B.episode_events(ep)
event_steps = sorted({e['step'] for e in events})

# Buttons run before the slider widget is instantiated, so they can nudge
# ss['step'] (which seeds the slider's value) this rerun.
sc = st.columns([1, 1, 1, 6, 1, 1, 1])
cur = ss.get('step', 0)
if sc[0].button('⏮', help='first'):
    ss['step'] = 0
if sc[1].button('◀', help='prev step'):
    ss['step'] = max(0, cur - 1)
if sc[2].button('◀⚑', help='prev event'):
    prev_ev = [e for e in event_steps if e < cur]
    if prev_ev:
        ss['step'] = prev_ev[-1]
if sc[4].button('⚑▶', help='next event'):
    next_ev = [e for e in event_steps if e > cur]
    if next_ev:
        ss['step'] = next_ev[0]
if sc[5].button('▶', help='next step'):
    ss['step'] = min(T - 1, cur + 1)
if sc[6].button('⏭', help='last'):
    ss['step'] = T - 1
step = sc[3].slider('step', 0, max(T - 1, 0), min(ss.get('step', 0), T - 1),
                    key='step')

# ---- event strip (markers aligned to the step axis, cursor at current step) ----
esfig = draw_event_strip(ep, step, events=events)
st.pyplot(esfig, use_container_width=True)
plt.close(esfig)

# ---- panels ----
left, right = st.columns([1, 1])

with left:
    st.subheader('Agent view')
    imgs = ep.get('image')
    if imgs is not None and len(imgs):
        frame = np.asarray(imgs[int(np.clip(step, 0, len(imgs) - 1))])
        st.image(frame, width=360,
                 caption=f'step {step}/{T - 1} · what the agent sees')
    else:
        st.caption('No frames recorded (egocentric_view produced no image?).')

    # readout
    action_names = ep.get('action_names', B.ACTION_NAMES)
    a = int(ep['action'][int(np.clip(step, 0, T - 1))])
    r = float(ep['reward'][int(np.clip(step, 0, T - 1))])
    rc = st.columns(2)
    rc[0].metric('action', action_names[a] if a < len(action_names) else str(a))
    rc[1].metric('step reward', f'{r:+.2f}')

    st.markdown('**Vitals**')
    vit = vitals_at(ep, step)
    for v in B.VITALS:
        if v in vit:
            st.progress(vit[v] / 9.0, text=f'{v}: {vit[v]}/9')

    unlocks = ep.get('unlocks')
    if unlocks is not None and step < len(unlocks) and unlocks[step]:
        st.success('🏆 unlocked this step: ' + ', '.join(unlocks[step]))
    # cumulative achievements up to this step
    if unlocks is not None:
        cum = sorted({a for s in unlocks[:step + 1] for a in s})
        if cum:
            st.caption('unlocked so far: ' + ', '.join(cum))

with right:
    st.subheader('Trajectory')
    fig = draw_trajectory(ep, step, area)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption(f"cause of death: **{eps[chosen]['cause_of_death']}** · "
               f"world_seed {index.get('world_seed')}")

# ---- vitals timeline (full width, step cursor tracks the scrubber) ----
st.subheader('Vitals over time')
vfig = draw_vitals_timeline(ep, step)
if vfig is not None:
    st.pyplot(vfig, use_container_width=True)
    plt.close(vfig)
    st.caption('Top: achievement unlocks + do/sleep actions. Bottom: vital '
               'traces (0-9) with cause-of-hurt markers on health. Red line = '
               'current step.')
else:
    st.caption('No vitals recorded for this episode.')
