# =====================================================================
# pipeline/defaults.sh — DEFAULT settings for the 13-step pipeline
# ---------------------------------------------------------------------
# These are the POSTER-config defaults (the masked_k5_invert_reg_wrld45 /
# default_masked_25m_2xtrain net: boxy world 45, size25m, relu, masked k5).
#
# USAGE (from a thin run_0All_*.sh wrapper):
#     source "$(dirname "$0")/pipeline/defaults.sh"   # set every knob to default
#     SIZE=size25m; DISABLE_MOBS=True; ...            # override only what differs
#     source "$(dirname "$0")/pipeline/lib.sh"        # derive paths + run pipeline
#
# Every knob is defined here (incl. cow/mob/vital at STOCK values), so lib.sh
# can build the full arg set unconditionally — a wrapper only ever *changes*
# values, never adds plumbing. This is the YAML-style "defaults + overrides"
# behavior: short wrappers, one source of truth for the pipeline body.
# =====================================================================

# --- Environment ---
ENV=crafter_small               # crafter or crafter_small (32x32)
SIZE=size25m                    # size25m-WM + smallnet encoder + big value head
RANDOM_SPAWN=True
FIXED_LAYOUT=True               # identical terrain shell; resources/mobs resample
TRAIN_FIXED_SEED=False
EVAL_FIXED_SEED=False
TRAIN_SEED=45
EVAL_SEED=45
ISLAND_BORDER=False             # True -> island world; False -> normal boxy world
ISLAND_SEED=4
ISLAND_FILL=0.72
ISLAND_ROUGHNESS=0.16
EGOCENTRIC_VIEW=9               # 0=allo, odd int (e.g. 7/9)=ego
UPRIGHT_SPRITES=True
DISABLE_MOBS=False              # training
EVAL_DISABLE_MOBS=True          # eval/trajectory
IMAGE_SIZE=64
CUSTOM_WORLD=""                 # non-empty -> --env.crafter.custom_world <path>

# --- Analysis gate (step 3 eval mode + steps 4/5/6/7/9) ---
# RUN_ANALYSIS=true  -> full-activation eval (vitals_only=False) + run the heavy
#   decode/tuning/dream/manifold steps. The default for analysis nets.
# RUN_ANALYSIS=false -> vitals_only eval (pos/vitals/actions/achievements, no
#   activations/images) + SKIP steps 4,5,6,7,9. For long-episode behavior-only
#   runs (more cows / variable mobs) where full activations would OOM.
# NOTE: deliberately NOT set here. lib.sh defaults it to true via `:=` so that a
# wrapper's `RUN_ANALYSIS=${RUN_ANALYSIS:-false}` (and submit-time env override)
# both work — hard-setting it here would clobber those. Behavior-only wrappers
# set their own default; the standard analysis nets get true from lib.sh.

# --- Cow density knobs (STOCK = no-op) ---
COW_SPAWN_PROB=0.01
COW_SPAN_DIST=5
COW_TARGET_SCALE=1.0
COW_WORLDGEN_SCALE=1.0

# --- Hostile-mob density knobs (STOCK = no-op) ---
MOB_SCALE_DIST=off              # off | uniform | exponential
MOB_TARGET_SCALE=1.0
MOB_SPAWN_PROB_SCALE=1.0
MOB_WORLDGEN_SCALE=1.0
MOB_SCALE_MIN=0.0
MOB_SCALE_MAX=2.0
MOB_SCALE_MEAN=0.5

# --- Reward shaping (STOCK = off) ---
VITAL_REWARD=False

# --- Training length (env steps) ---
ENV_STEPS=""                    # empty -> preset default (crafter_small = 1.1M)
HALF_ENV_STEPS=False
CRAFTER_SMALL_DEFAULT_STEPS=1100000

# --- Model architecture ---
INCLUDE_POSITION=False
GRU_ACT=relu
RSSM_ACT=relu
ENC_ACT=relu
DEC_ACT=relu
REWHEAD_ACT=relu
CONHEAD_ACT=relu
POLICY_ACT=relu
VALUE_ACT=relu

# --- Regularization ---
L1_WM=1e0
L1_POLICY=1e-2
L1_VALUE=1e-2
WEIGHT_DECAY=1e0

# --- Normalization / GRU gate ---
NORM=rms                        # RSSM norm impl: rms | layer
ENC_NORM=rms
GATE_NORM=False                 # dyn.rssm.gate_norm
CONDITION=regular_masked        # label only (pinned to default net)
TRAIN_NOISE_STD=0.0
TRAIN_NOISE_WHERE=candidate

# --- Masking / rollout / base logdir ---
ROLLOUT_K=0                     # 0=disabled
ROLLOUT_DYN_SCALE=1.0
MASK_VISUAL_K=5
MASK_VISUAL_INVERT=True
MASK_VISUAL_PROB=False
MASK_RECON_LOSS=True
BASE_LOGDIR=./logdir/default_masked_25m_2xtrain

# --- Eval trajectory ---
EVAL_EPISODES=100

# --- Layer decoding (step 4) ---
LAYER_DECODE_NJOBS=8
LAYER_DECODE_MAX_EPISODES=100

# --- Standard decoder (step 5, for dream decode) ---
DECODE_REPR=deter
DECODE_NJOBS=4
DECODE_DEVICE=cuda

# --- Dream decode (step 6) ---
DREAM_EPISODES=10
DREAM_BATCHES=5

# --- Dream seed ablation (step 10) ---
DSA_ENABLE=true
DSA_NUM_EPISODES=10
DSA_NUM_BATCHES=10
DSA_N_SEEDS=16
DSA_REPORT_BATCH_SIZE=16
DSA_HORIZON=16
DSA_WARMUP=0
DSA_WARMUP_SLACK=16
DSA_NOISE_MODE=shuffle
DSA_CENTRAL=mean
DSA_SHADING=none
DSA_N_EXAMPLE_DREAMS=5
DSA_MAKE_GIF=True
DSA_SAVE_ACTIVATIONS=True       # dump dream_deter_{A,B,C,D}.pkl (steps 11/12)
DSA_INCLUDE_E=False
# Delete the (32-64 GB) dream_deter_{A,B,C,D}.pkl dumps after steps 11+12 have
# consumed them. They are only inputs to the manifold steps; nothing reads them
# afterward. Set false to keep them (e.g. to re-run manifolds without re-rolling).
DSA_CLEANUP_DUMPS=true

# --- Per-condition manifold (step 11) ---
PERCOND_MANIFOLD=true
PERCOND_LAYERS="dyn/deter"
PERCOND_MAX_WAKE=4000
PERCOND_MAX_DREAM=200000
PERCOND_N_NEIGHBORS=150

# --- Condition-overlay manifold (step 12) ---
OVERLAY_MANIFOLD=true
OVERLAY_LAYERS="dyn/deter"
OVERLAY_MAX_WAKE=4000
OVERLAY_MAX_DREAM=0
OVERLAY_SHADING=band

# --- Tuning curve (step 7) ---
TUNING_NJOBS=2
TUNING_MAX_EPISODES=100
TUNING_LAYERS=""
TUNING_TEST_DATA=""
TUNING_MAX_NEURONS=0
TUNING_NO_HD=false
SI_THRESH=0.5
EV_THRESH=0.5
EV_UNTHRESH=0.15
HD_THRESH=0.5
SMOOTH_SIGMA=0
PEAK_SMOOTH_SIGMA=1.5
PEAK_THRESHOLD_FRAC=0.75
PEAK_PROMINENCE_FRAC=0.1
EV_FILTER=0.4
MIN_BBOX=5
DIST_CUTOFF=7
AREA="32 32"

# --- Clustering / analyze_tuning (step 8, disabled by default — OOM) ---
RUN_STEP8_CLUSTERING=false
N_COMPONENTS=50
PERPLEXITY=30
UMAP_NEIGHBORS=15
MIN_CLUSTER_SIZE=20
MIN_SAMPLES=5
ISOMAP_NEIGHBORS=15
NORMALIZE=true
ANALYZE_EV_THRESH=""

# --- Manifold (step 9) ---
MAX_WAKE_SAMPLES=4000
MAX_DREAM_SAMPLES=0
MANIFOLD_N_NEIGHBORS=150
MANIFOLD_LAYERS=""
MANIFOLD_MIN_BBOX=0
NO_ISOMAP=false
MANIFOLD_EV_THRESH=0.5

# --- Vital dynamics (step 3b) ---
VITAL_DYNAMICS_ENABLE=true
VITAL_MAX_EPISODE_PLOTS=10
VITAL_EVENT_WINDOW=15
