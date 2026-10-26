# =====================================================================
# pipeline/lib.sh — the 13-step pipeline body (sourced by run_0All_*.sh)
# ---------------------------------------------------------------------
# Expects every knob from pipeline/defaults.sh to be set (sourced first), plus
# any wrapper overrides and a final LOGDIR. Sourcing this file derives paths,
# builds the full arg set, writes hyperparams.txt, then runs all 13 steps with
# resume/stamp checkpointing. Fix the pipeline ONCE here, not in 12 scripts.
#
# Submit-time overrides (env vars; pass via sbatch --export):
#   FORCE_FROM=N      redo steps >= N        FORCE_FROM=all  redo every step
#   FORCE_STEP="3 7"  redo only these steps  FORCE_TRAIN=true retrain even if ckpt exists
# =====================================================================

set -e
export PATH="/gpfs/radev/home/vv266/.conda/envs/base311/bin:$PATH"

# --- Dry-run mode: PIPELINE_DRY_RUN=1 echoes every python command instead of
# running it (no GPU, no training). Useful for wiring/arg verification. ---
if [ -n "${PIPELINE_DRY_RUN:-}" ]; then
    python() { echo "+ python $*"; }
    echo "  [dry-run] PIPELINE_DRY_RUN set — python commands will be echoed, not executed"
fi

# --- GPU utilization logging (background; killed on exit) ---
if [ -z "${PIPELINE_DRY_RUN:-}" ] && command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 10 \
        > "gpu_usage_${SLURM_JOB_ID:-local}.log" &
    _GPU_LOG_PID=$!
    trap "kill $_GPU_LOG_PID 2>/dev/null" EXIT
fi

# =====================================================================
# DERIVED PATHS
# =====================================================================
: "${LOGDIR:?LOGDIR must be set by the wrapper before sourcing lib.sh}"
CONFIGS="$ENV $SIZE"
TRAJ_DIR="${LOGDIR}/trajectories"
DECODER_DIR="${LOGDIR}/decoder_results"
DECODER_PKL="${DECODER_DIR}/classifier_${DECODE_REPR}.pkl"
DREAM_DIR="${LOGDIR}/dream_results"
DREAM_PKL="${DREAM_DIR}/dream_results.pkl"
TUNING_SAVE="${LOGDIR}/tuning_results"
TUNING_PKL="${TUNING_SAVE}/tuning_results.pkl"
CLUSTER_SAVE="${TUNING_SAVE}/cluster_plots"
VITAL_DYNAMICS_SAVE="${LOGDIR}/vital_dynamics"
DSA_DIR="${LOGDIR}/dream_seed_ablation_${DSA_HORIZON}"
DSA_REPORT_LENGTH=$(( DSA_HORIZON + DSA_WARMUP_SLACK ))
DSA_BATCH_LENGTH=$DSA_REPORT_LENGTH
OVERLAY_SAVE="${DSA_DIR}/manifold_conditions"

# Training-length override (STEP 1 only). ENV_STEPS wins; else HALF toggle.
TRAIN_STEP_ARGS=""
if [ -n "$ENV_STEPS" ]; then
    TRAIN_STEP_ARGS=" --run.steps $ENV_STEPS"
    echo "Training length: run.steps=$ENV_STEPS (explicit override)"
elif [ "$HALF_ENV_STEPS" = "True" ]; then
    TRAIN_STEP_ARGS=" --run.steps $(( CRAFTER_SMALL_DEFAULT_STEPS / 2 ))"
    echo "Training length: run.steps=$(( CRAFTER_SMALL_DEFAULT_STEPS / 2 )) (HALF_ENV_STEPS)"
else
    echo "Training length: run.steps from preset (crafter_small default = $CRAFTER_SMALL_DEFAULT_STEPS)"
fi

# =====================================================================
# PIPELINE CHECKPOINTING (stamp files / done-markers)
# =====================================================================
STAMP_DIR="${LOGDIR}/.pipeline_status"
# Dry-run must not touch the real run dir: redirect stamps to a throwaway temp.
[ -n "${PIPELINE_DRY_RUN:-}" ] && STAMP_DIR="$(mktemp -d)"
mkdir -p "$STAMP_DIR"
_GIT_SHA=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --short HEAD 2>/dev/null || echo nogit)

step_done() { [ -f "${STAMP_DIR}/step$1.done" ]; }
mark_done() {
    printf 'step=%s\ntime=%s\ngit=%s\njob=%s\n' \
        "$1" "$(date -Is)" "$_GIT_SHA" "${SLURM_JOB_ID:-none}" > "${STAMP_DIR}/step$1.done"
    echo "[$(date -Is)] step $1 DONE (git $_GIT_SHA job ${SLURM_JOB_ID:-none})" >> "${STAMP_DIR}/history.log"
    echo "  [status] step $1 complete -> ${STAMP_DIR}/step$1.done"
}
skip_note() {
    echo "  [status] STEP $1 SKIPPED (marker present; FORCE_FROM=$1 to redo)"
    echo "[$(date -Is)] step $1 SKIP" >> "${STAMP_DIR}/history.log"
}

# Apply FORCE overrides up front by removing the relevant markers.
if [ "${FORCE_FROM:-}" = "all" ]; then
    rm -f "${STAMP_DIR}"/step*.done
    echo "  [status] FORCE_FROM=all -> cleared all step markers"
elif [ -n "${FORCE_FROM:-}" ]; then
    for _n in $(seq "${FORCE_FROM}" 13); do rm -f "${STAMP_DIR}/step${_n}.done"; done
    echo "  [status] FORCE_FROM=${FORCE_FROM} -> cleared markers for steps >= ${FORCE_FROM}"
fi
if [ -n "${FORCE_STEP:-}" ]; then
    for _n in ${FORCE_STEP}; do rm -f "${STAMP_DIR}/step${_n}.done"; done
    echo "  [status] FORCE_STEP='${FORCE_STEP}' -> cleared those step markers"
fi

# =====================================================================
# BUILD TRAIN/EVAL ARGS (full set, always — knobs at stock values are no-ops)
# =====================================================================
MODEL_ARGS=""
MODEL_ARGS+=" --agent.include_position $INCLUDE_POSITION"
MODEL_ARGS+=" --agent.dyn.rssm.gru_act $GRU_ACT"
MODEL_ARGS+=" --agent.dyn.rssm.norm $NORM"
MODEL_ARGS+=" --agent.dyn.rssm.gate_norm $GATE_NORM"
MODEL_ARGS+=" --agent.dyn.rssm.train_noise_std $TRAIN_NOISE_STD"
MODEL_ARGS+=" --agent.dyn.rssm.train_noise_where $TRAIN_NOISE_WHERE"
MODEL_ARGS+=" --agent.dyn.rssm.act $RSSM_ACT"
MODEL_ARGS+=" --agent.enc.simple.act $ENC_ACT"
MODEL_ARGS+=" --agent.enc.simple.norm $ENC_NORM"
MODEL_ARGS+=" --agent.dec.simple.act $DEC_ACT"
MODEL_ARGS+=" --agent.rewhead.act $REWHEAD_ACT"
MODEL_ARGS+=" --agent.conhead.act $CONHEAD_ACT"
MODEL_ARGS+=" --agent.policy.act $POLICY_ACT"
MODEL_ARGS+=" --agent.value.act $VALUE_ACT"
MODEL_ARGS+=" --agent.opt.wd $WEIGHT_DECAY"
MODEL_ARGS+=" --agent.l1_wm $L1_WM"
MODEL_ARGS+=" --agent.l1_policy $L1_POLICY"
MODEL_ARGS+=" --agent.l1_value $L1_VALUE"
MODEL_ARGS+=" --agent.mask_visual_k $MASK_VISUAL_K"
MODEL_ARGS+=" --agent.mask_visual_invert $MASK_VISUAL_INVERT"
MODEL_ARGS+=" --agent.mask_visual_prob $MASK_VISUAL_PROB"
MODEL_ARGS+=" --agent.mask_recon_loss $MASK_RECON_LOSS"
if [ "${ROLLOUT_K:-0}" -gt 0 ]; then
    MODEL_ARGS+=" --agent.rollout_k $ROLLOUT_K"
    MODEL_ARGS+=" --agent.rollout_dyn_scale $ROLLOUT_DYN_SCALE"
    MODEL_ARGS+=" --agent.loss_scales.rollout_dyn $ROLLOUT_DYN_SCALE"
fi

# Shared env args (train + eval differ only in fixed_seed + disable_mobs).
# _build_env_args <fixed_seed> <disable_mobs> -> echoes the arg string.
_build_env_args() {
    local fs="$1" dm="$2"
    local a=""
    a+=" --env.crafter.random_spawn $RANDOM_SPAWN"
    a+=" --env.crafter.fixed_seed $fs"
    a+=" --env.crafter.egocentric_view $EGOCENTRIC_VIEW"
    a+=" --env.crafter.size $IMAGE_SIZE $IMAGE_SIZE"
    a+=" --env.crafter.upright_sprites $UPRIGHT_SPRITES"
    a+=" --env.crafter.disable_mobs $dm"
    a+=" --env.crafter.fixed_layout $FIXED_LAYOUT"
    a+=" --env.crafter.island_border $ISLAND_BORDER"
    a+=" --env.crafter.island_seed $ISLAND_SEED"
    a+=" --env.crafter.island_fill $ISLAND_FILL"
    a+=" --env.crafter.island_roughness $ISLAND_ROUGHNESS"
    a+=" --env.crafter.cow_spawn_prob $COW_SPAWN_PROB"
    a+=" --env.crafter.cow_span_dist $COW_SPAN_DIST"
    a+=" --env.crafter.cow_target_scale $COW_TARGET_SCALE"
    a+=" --env.crafter.cow_worldgen_scale $COW_WORLDGEN_SCALE"
    a+=" --env.crafter.mob_scale_dist $MOB_SCALE_DIST"
    a+=" --env.crafter.mob_target_scale $MOB_TARGET_SCALE"
    a+=" --env.crafter.mob_spawn_prob_scale $MOB_SPAWN_PROB_SCALE"
    a+=" --env.crafter.mob_worldgen_scale $MOB_WORLDGEN_SCALE"
    a+=" --env.crafter.mob_scale_min $MOB_SCALE_MIN"
    a+=" --env.crafter.mob_scale_max $MOB_SCALE_MAX"
    a+=" --env.crafter.mob_scale_mean $MOB_SCALE_MEAN"
    a+=" --env.crafter.vital_reward $VITAL_REWARD"
    [ -n "${CUSTOM_WORLD:-}" ] && a+=" --env.crafter.custom_world $CUSTOM_WORLD"
    echo "$a"
}
TRAIN_ENV_ARGS="$(_build_env_args "$TRAIN_FIXED_SEED" "$DISABLE_MOBS")"
EVAL_ENV_ARGS="$(_build_env_args "$EVAL_FIXED_SEED" "$EVAL_DISABLE_MOBS")"

# RUN_ANALYSIS=false -> vitals_only eval (no activations/images) + skip heavy
# analysis steps (4,5,6,7,9). Default true keeps full-activation behavior.
# `:=` so a wrapper default / submit-time env override survives (defaults.sh
# deliberately does NOT set it — see note there).
: "${RUN_ANALYSIS:=true}"
if [ "$RUN_ANALYSIS" = "true" ]; then EVAL_VITALS_ONLY=False; else EVAL_VITALS_ONLY=True; fi

# =====================================================================
# SAVE HYPERPARAMS
# =====================================================================
if [ -n "${PIPELINE_DRY_RUN:-}" ]; then
  echo "[dry-run] skipping LOGDIR mkdir + hyperparams.txt write ($LOGDIR)"
else
mkdir -p "$LOGDIR"
DATE_SUBMITTED=$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | grep -oP 'SubmitTime=\K\S+' || date '+%Y-%m-%dT%H:%M:%S')
cat > "${LOGDIR}/hyperparams.txt" <<EOF
# Hyperparams — $(date)
date_submitted=$DATE_SUBMITTED
date_started=$(date '+%Y-%m-%dT%H:%M:%S')
# --- Environment ---
env=$ENV
size=$SIZE
random_spawn=$RANDOM_SPAWN
train_fixed_seed=$TRAIN_FIXED_SEED
eval_fixed_seed=$EVAL_FIXED_SEED
fixed_layout=$FIXED_LAYOUT
island_border=$ISLAND_BORDER
island_seed=$ISLAND_SEED
island_fill=$ISLAND_FILL
island_roughness=$ISLAND_ROUGHNESS
train_seed=$TRAIN_SEED
eval_seed=$EVAL_SEED
egocentric_view=$EGOCENTRIC_VIEW
image_size=$IMAGE_SIZE
upright_sprites=$UPRIGHT_SPRITES
disable_mobs=$DISABLE_MOBS
eval_disable_mobs=$EVAL_DISABLE_MOBS
env_steps=$ENV_STEPS
half_env_steps=$HALF_ENV_STEPS
train_step_args=$TRAIN_STEP_ARGS
# --- Cow / mob density ---
cow_spawn_prob=$COW_SPAWN_PROB
cow_span_dist=$COW_SPAN_DIST
cow_target_scale=$COW_TARGET_SCALE
cow_worldgen_scale=$COW_WORLDGEN_SCALE
mob_scale_dist=$MOB_SCALE_DIST
mob_target_scale=$MOB_TARGET_SCALE
mob_spawn_prob_scale=$MOB_SPAWN_PROB_SCALE
mob_worldgen_scale=$MOB_WORLDGEN_SCALE
mob_scale_min=$MOB_SCALE_MIN
mob_scale_max=$MOB_SCALE_MAX
mob_scale_mean=$MOB_SCALE_MEAN
vital_reward=$VITAL_REWARD
# --- Model architecture ---
include_position=$INCLUDE_POSITION
gru_act=$GRU_ACT
rssm_norm=$NORM
enc_norm=$ENC_NORM
gru_gate_norm=$GATE_NORM
rssm_act=$RSSM_ACT
enc_act=$ENC_ACT
dec_act=$DEC_ACT
rewhead_act=$REWHEAD_ACT
conhead_act=$CONHEAD_ACT
policy_act=$POLICY_ACT
value_act=$VALUE_ACT
# --- Regularization ---
weight_decay=$WEIGHT_DECAY
l1_wm=$L1_WM
l1_policy=$L1_POLICY
l1_value=$L1_VALUE
train_noise_std=$TRAIN_NOISE_STD
train_noise_where=$TRAIN_NOISE_WHERE
# --- Visual masking / rollout ---
mask_visual_k=$MASK_VISUAL_K
mask_visual_invert=$MASK_VISUAL_INVERT
mask_visual_prob=$MASK_VISUAL_PROB
mask_recon_loss=$MASK_RECON_LOSS
condition=$CONDITION
rollout_k=$ROLLOUT_K
rollout_dyn_scale=$ROLLOUT_DYN_SCALE
# --- Eval / decode / dream ---
eval_episodes=$EVAL_EPISODES
layer_decode_njobs=$LAYER_DECODE_NJOBS
layer_decode_max_episodes=$LAYER_DECODE_MAX_EPISODES
decode_repr=$DECODE_REPR
decode_njobs=$DECODE_NJOBS
decode_device=$DECODE_DEVICE
dream_episodes=$DREAM_EPISODES
dream_batches=$DREAM_BATCHES
# --- Dream seed ablation ---
dsa_enable=$DSA_ENABLE
dsa_num_episodes=$DSA_NUM_EPISODES
dsa_num_batches=$DSA_NUM_BATCHES
dsa_n_seeds=$DSA_N_SEEDS
dsa_report_batch_size=$DSA_REPORT_BATCH_SIZE
dsa_horizon=$DSA_HORIZON
dsa_warmup=$DSA_WARMUP
dsa_warmup_slack=$DSA_WARMUP_SLACK
dsa_noise_mode=$DSA_NOISE_MODE
dsa_save_activations=$DSA_SAVE_ACTIVATIONS
dsa_cleanup_dumps=$DSA_CLEANUP_DUMPS
# --- Tuning ---
tuning_njobs=$TUNING_NJOBS
tuning_max_episodes=$TUNING_MAX_EPISODES
tuning_layers=$TUNING_LAYERS
si_thresh=$SI_THRESH
ev_thresh=$EV_THRESH
min_bbox=$MIN_BBOX
area=$AREA
# --- Manifold ---
max_wake_samples=$MAX_WAKE_SAMPLES
manifold_n_neighbors=$MANIFOLD_N_NEIGHBORS
manifold_ev_thresh=$MANIFOLD_EV_THRESH
# --- Vital dynamics ---
vital_dynamics_enable=$VITAL_DYNAMICS_ENABLE
# --- Meta ---
logdir=$LOGDIR
slurm_job_id=${SLURM_JOB_ID:-none}
EOF
echo "Saved hyperparams to ${LOGDIR}/hyperparams.txt"
fi

# =====================================================================
# STEP 1: TRAIN
# =====================================================================
echo ""; echo "============================================================"
echo "  STEP 1/13: TRAINING"; echo "============================================================"
if step_done 1; then
    skip_note 1
elif [ -f "${LOGDIR}/ckpt/latest" ] && [ "${FORCE_TRAIN:-}" != "true" ]; then
    echo "  [status] STEP 1 AUTO-SKIP: checkpoint exists (${LOGDIR}/ckpt/latest). FORCE_TRAIN=true to retrain."
    mark_done 1
else
    python dreamerv3/main.py \
        --configs $CONFIGS --logdir "$LOGDIR" --seed $TRAIN_SEED \
        $MODEL_ARGS $TRAIN_ENV_ARGS $TRAIN_STEP_ARGS
    mark_done 1
fi

# Resolve checkpoint (ALWAYS — later steps need $CKPT even if STEP 1 skipped)
LATEST_FILE="${LOGDIR}/ckpt/latest"
if [ -f "$LATEST_FILE" ]; then
    CKPT="${LOGDIR}/ckpt/$(cat "$LATEST_FILE")"
    echo "Resolved checkpoint: $CKPT"
elif [ -n "${PIPELINE_DRY_RUN:-}" ]; then
    CKPT="${LOGDIR}/ckpt/DRYRUN_TIMESTAMP"
    echo "[dry-run] no checkpoint; using placeholder $CKPT"
else
    echo "ERROR: No checkpoint found at $LATEST_FILE"; exit 1
fi

# =====================================================================
# STEP 2: PLOT TRAINING PROGRESS
# =====================================================================
echo ""; echo "============================================================"
echo "  STEP 2/13: TRAINING PLOTS"; echo "============================================================"
if step_done 2; then skip_note 2; else
    MPLBACKEND=Agg python dreamerv3/plot_training.py \
        --logdir "$LOGDIR" --save "${LOGDIR}/plots/" --smooth 50
    mark_done 2
fi

# =====================================================================
# STEP 3: EVAL TRAJECTORIES + PLOTS
# =====================================================================
echo ""; echo "============================================================"
echo "  STEP 3/13: EVAL TRAJECTORIES"; echo "============================================================"
if step_done 3; then skip_note 3; else
    python dreamerv3/main.py \
        --configs $CONFIGS --logdir "$LOGDIR" --script eval_trajectory \
        --run.from_checkpoint "$CKPT" \
        --eval_trajectory.num_episodes $EVAL_EPISODES \
        --eval_trajectory.save_path "$TRAJ_DIR" \
        --eval_trajectory.vitals_only $EVAL_VITALS_ONLY \
        --seed $EVAL_SEED $MODEL_ARGS $EVAL_ENV_ARGS --jax.platform cpu
    # Full trajectory/world/activation plots need recorded images+activations, so
    # run them only when RUN_ANALYSIS=true. plot_eval (vitals eval summary) works
    # on any trajectory dir, so it always runs.
    if [ "$EVAL_VITALS_ONLY" != "True" ]; then
        MPLBACKEND=Agg python dreamerv3/plot_trajectories.py \
            --data "$TRAJ_DIR" --plot all --save "${LOGDIR}/plots/"
    fi
    MPLBACKEND=Agg PYTHONPATH=. python dreamerv3/plot_eval.py \
        --data "$TRAJ_DIR" --save "${LOGDIR}/plots/"
    mark_done 3
fi

# =====================================================================
# STEP 3b: VITAL DYNAMICS (observational; only needs $TRAJ_DIR)
# =====================================================================
echo ""; echo "============================================================"
echo "  STEP 3b/13: VITAL DYNAMICS"; echo "============================================================"
if step_done 3b; then
    skip_note 3b
elif [ "$VITAL_DYNAMICS_ENABLE" != "true" ]; then
    echo "STEP 3b: SKIPPED (VITAL_DYNAMICS_ENABLE != true). Not marked done."
else
    PYTHONPATH=. MPLBACKEND=Agg python dreamerv3/vital_dynamics.py \
        --data "$TRAJ_DIR" --save "$VITAL_DYNAMICS_SAVE" \
        --max_episode_plots $VITAL_MAX_EPISODE_PLOTS --event_window $VITAL_EVENT_WINDOW
    mark_done 3b
fi

# =====================================================================
# STEP 3c: OCCUPANCY MAP (positions-only). Runs ALWAYS — needs only player_pos,
# so it works on vitals_only trajectories (RUN_ANALYSIS=false) too. Emits
# occupancy_fullset_*.svg (visit-count hotspots + pRNN coverage) without a
# decoder/activations/GPU. In full-analysis mode step 5 also makes the
# decode-based occupancy_vs_error; this is the lightweight always-on companion.
# =====================================================================
echo ""; echo "============================================================"
echo "  STEP 3c/13: OCCUPANCY MAP (positions-only)"; echo "============================================================"
if step_done 3c; then skip_note 3c; else
    # Occupancy over ALL eval trajectories (positions-only, cheap) — not a
    # test-split subset. max_episodes 0 = every recorded episode.
    OCC_ARGS="--data $TRAJ_DIR --save ${LOGDIR}/occupancy --mode occupancy"
    OCC_ARGS+=" --max_episodes 0"
    [ "$FIXED_LAYOUT" = "True" ]  && OCC_ARGS+=" --fixed_layout"
    [ "$ISLAND_BORDER" = "True" ] && OCC_ARGS+=" --island --island_seed $ISLAND_SEED --island_fill $ISLAND_FILL --island_roughness $ISLAND_ROUGHNESS"
    MPLBACKEND=Agg python dreamerv3/decode_position.py $OCC_ARGS
    mark_done 3c
fi

# =====================================================================
# STEP 4: LAYER DECODING
# =====================================================================
echo ""; echo "============================================================"
echo "  STEP 4/13: LAYER DECODING"; echo "============================================================"
if step_done 4; then skip_note 4
elif [ "$RUN_ANALYSIS" != "true" ]; then echo "  [status] STEP 4 SKIPPED (RUN_ANALYSIS != true)"; else
    MPLBACKEND=Agg python dreamerv3/decode_position.py \
        --data "$TRAJ_DIR" --save "${LOGDIR}/layer_decoder_results" \
        --mode layers --device cuda --n_jobs $LAYER_DECODE_NJOBS \
        --max_episodes $LAYER_DECODE_MAX_EPISODES
    mark_done 4
fi

# =====================================================================
# STEP 5: STANDARD DECODER (save_model for dream decode)
# =====================================================================
echo ""; echo "============================================================"
echo "  STEP 5/13: STANDARD DECODER (--save_model)"; echo "============================================================"
if step_done 5; then skip_note 5
elif [ "$RUN_ANALYSIS" != "true" ]; then echo "  [status] STEP 5 SKIPPED (RUN_ANALYSIS != true)"; else
    MPLBACKEND=Agg python dreamerv3/decode_position.py \
        --data "$TRAJ_DIR" --save "$DECODER_DIR" --repr "$DECODE_REPR" \
        --save_model --n_jobs $DECODE_NJOBS --device $DECODE_DEVICE
    mark_done 5
fi

# =====================================================================
# STEP 6: DREAM DECODE
# =====================================================================
echo ""; echo "============================================================"
echo "  STEP 6/13: DREAM DECODE"; echo "============================================================"
if step_done 6; then skip_note 6
elif [ "$RUN_ANALYSIS" != "true" ]; then echo "  [status] STEP 6 SKIPPED (RUN_ANALYSIS != true)"; else
    python dreamerv3/main.py \
        --configs $CONFIGS --logdir "$LOGDIR" --script dream_decode \
        --run.from_checkpoint "$CKPT" \
        --dream_decode.decoder_model "$DECODER_PKL" \
        --dream_decode.save_path "$DREAM_DIR" \
        --dream_decode.num_episodes $DREAM_EPISODES \
        --dream_decode.num_batches $DREAM_BATCHES \
        --seed $EVAL_SEED $MODEL_ARGS $EVAL_ENV_ARGS
    mark_done 6
fi

# =====================================================================
# STEP 7: TUNING CURVE ANALYSIS
# =====================================================================
echo ""; echo "============================================================"
echo "  STEP 7/13: TUNING CURVES"; echo "============================================================"
if step_done 7; then skip_note 7
elif [ "$RUN_ANALYSIS" != "true" ]; then echo "  [status] STEP 7 SKIPPED (RUN_ANALYSIS != true)"; else
    TUNING_ARGS="--data $TRAJ_DIR --save $TUNING_SAVE --n_jobs $TUNING_NJOBS"
    TUNING_ARGS+=" --max_episodes $TUNING_MAX_EPISODES"
    TUNING_ARGS+=" --SI_thresh $SI_THRESH --EV_thresh $EV_THRESH"
    TUNING_ARGS+=" --EV_unthresh $EV_UNTHRESH --HD_thresh $HD_THRESH"
    TUNING_ARGS+=" --max_neurons $TUNING_MAX_NEURONS --smooth_sigma $SMOOTH_SIGMA"
    TUNING_ARGS+=" --peak_smooth_sigma $PEAK_SMOOTH_SIGMA"
    TUNING_ARGS+=" --peak_threshold_frac $PEAK_THRESHOLD_FRAC"
    TUNING_ARGS+=" --peak_prominence_frac $PEAK_PROMINENCE_FRAC"
    TUNING_ARGS+=" --ev_filter $EV_FILTER --min_bbox $MIN_BBOX --dist_cutoff $DIST_CUTOFF"
    [ -n "$TUNING_LAYERS" ]    && TUNING_ARGS+=" --layers $TUNING_LAYERS"
    [ -n "$TUNING_TEST_DATA" ] && TUNING_ARGS+=" --test_data $TUNING_TEST_DATA"
    [ "$TUNING_NO_HD" = "true" ] && TUNING_ARGS+=" --no_hd"
    [ -n "$AREA" ]             && TUNING_ARGS+=" --area $AREA"
    MPLBACKEND=Agg python dreamerv3/tuning_curve.py $TUNING_ARGS

    PKL_ARGS="--from_pkl $TUNING_PKL --plot_autocorr"
    [ -n "$TUNING_LAYERS" ] && PKL_ARGS+=" --layers $TUNING_LAYERS"
    MPLBACKEND=Agg python dreamerv3/tuning_curve.py $PKL_ARGS
    mark_done 7
fi

# =====================================================================
# STEP 8: TUNING CLUSTERING & DISTRIBUTIONS  (disabled by default — OOM)
# =====================================================================
echo ""; echo "============================================================"
echo "  STEP 8/13: TUNING CLUSTERING & DISTRIBUTIONS"; echo "============================================================"
if step_done 8; then
    skip_note 8
elif [ "$RUN_STEP8_CLUSTERING" != "true" ]; then
    echo "STEP 8: SKIPPED (RUN_STEP8_CLUSTERING != true — OOM-prone autocorr). Not marked done."
else
    COMMON_ANALYZE_ARGS=""
    [ -n "$TUNING_LAYERS" ]      && COMMON_ANALYZE_ARGS+=" --layers $TUNING_LAYERS"
    [ "$NORMALIZE" = "false" ]   && COMMON_ANALYZE_ARGS+=" --no_normalize"
    [ -n "$ANALYZE_EV_THRESH" ]  && COMMON_ANALYZE_ARGS+=" --ev_thresh $ANALYZE_EV_THRESH"
    echo "--- Distributions ---"
    MPLBACKEND=Agg python dreamerv3/analyze_tuning.py \
        --from_pkl "$TUNING_PKL" --save "$CLUSTER_SAVE" --mode distributions $COMMON_ANALYZE_ARGS
    echo "--- Isomap (metrics mode) ---"
    MPLBACKEND=Agg python dreamerv3/analyze_tuning.py \
        --from_pkl "$TUNING_PKL" --save "$CLUSTER_SAVE" --mode metrics \
        --isomap_neighbors $ISOMAP_NEIGHBORS $COMMON_ANALYZE_ARGS
    echo "--- Autocorr clustering ---"
    MPLBACKEND=Agg python dreamerv3/analyze_tuning.py \
        --from_pkl "$TUNING_PKL" --save "$CLUSTER_SAVE" --mode autocorr \
        --n_components $N_COMPONENTS --perplexity $PERPLEXITY \
        --umap_neighbors $UMAP_NEIGHBORS --min_cluster_size $MIN_CLUSTER_SIZE \
        --min_samples $MIN_SAMPLES $COMMON_ANALYZE_ARGS
    mark_done 8
fi

# =====================================================================
# STEP 9: MANIFOLD ANALYSIS
# =====================================================================
echo ""; echo "============================================================"
echo "  STEP 9/13: MANIFOLD ANALYSIS"; echo "============================================================"
if step_done 9; then skip_note 9
elif [ "$RUN_ANALYSIS" != "true" ]; then echo "  [status] STEP 9 SKIPPED (RUN_ANALYSIS != true)"; else
    MANIFOLD_ARGS="--data $TRAJ_DIR --dream_data $DREAM_PKL"
    MANIFOLD_ARGS+=" --save ${LOGDIR}/manifold_results"
    MANIFOLD_ARGS+=" --max_wake_samples $MAX_WAKE_SAMPLES --max_dream_samples $MAX_DREAM_SAMPLES"
    MANIFOLD_ARGS+=" --n_neighbors $MANIFOLD_N_NEIGHBORS --min_bbox $MANIFOLD_MIN_BBOX"
    [ -n "$MANIFOLD_LAYERS" ] && MANIFOLD_ARGS+=" --layers $MANIFOLD_LAYERS"
    [ "$NO_ISOMAP" = "true" ] && MANIFOLD_ARGS+=" --no_isomap"
    if [ -n "$MANIFOLD_EV_THRESH" ]; then
        MANIFOLD_ARGS+=" --ev_thresh $MANIFOLD_EV_THRESH --tuning_pkl $TUNING_PKL"
    fi
    MPLBACKEND=Agg python dreamerv3/manifold_analysis.py $MANIFOLD_ARGS
    mark_done 9
fi

# =====================================================================
# STEP 10: DREAM SEED ABLATION (A/B/C/D)
# =====================================================================
echo ""; echo "============================================================"
echo "  STEP 10/13: DREAM SEED ABLATION"; echo "============================================================"
if step_done 10; then
    skip_note 10
elif [ "$DSA_ENABLE" != "true" ]; then
    echo "DSA_ENABLE != true — skipping dream seed ablation. Not marked done."
else
    python dreamerv3/main.py \
        --configs $CONFIGS --logdir "$LOGDIR" --script dream_seed_ablation \
        --run.from_checkpoint "$CKPT" \
        --dream_seed_ablation.decoder_model "$DECODER_PKL" \
        --dream_seed_ablation.save_path "$DSA_DIR" \
        --dream_seed_ablation.num_episodes $DSA_NUM_EPISODES \
        --dream_seed_ablation.num_batches $DSA_NUM_BATCHES \
        --dream_seed_ablation.n_seeds $DSA_N_SEEDS \
        --agent.report_batch_size $DSA_REPORT_BATCH_SIZE \
        --dream_seed_ablation.warmup $DSA_WARMUP \
        --dream_seed_ablation.horizon $DSA_HORIZON \
        --dream_seed_ablation.noise_mode $DSA_NOISE_MODE \
        --dream_seed_ablation.central $DSA_CENTRAL \
        --dream_seed_ablation.shading $DSA_SHADING \
        --dream_seed_ablation.n_example_dreams $DSA_N_EXAMPLE_DREAMS \
        --dream_seed_ablation.make_gif $DSA_MAKE_GIF \
        --dream_seed_ablation.save_activations $DSA_SAVE_ACTIVATIONS \
        --dream_seed_ablation.include_e $DSA_INCLUDE_E \
        --report_length $DSA_REPORT_LENGTH --batch_length $DSA_BATCH_LENGTH \
        --seed $EVAL_SEED $MODEL_ARGS $EVAL_ENV_ARGS
    mark_done 10
fi

# =====================================================================
# STEP 11: PER-CONDITION MANIFOLD (A/B/C/D)
# =====================================================================
echo ""; echo "============================================================"
echo "  STEP 11/13: PER-CONDITION MANIFOLD (A/B/C/D)"; echo "============================================================"
if step_done 11; then
    skip_note 11
elif [ "$PERCOND_MANIFOLD" != "true" ]; then
    echo "STEP 11: SKIPPED (PERCOND_MANIFOLD != true). Not marked done."
elif [ "$DSA_SAVE_ACTIVATIONS" != "True" ]; then
    echo "STEP 11: SKIPPED (DSA_SAVE_ACTIVATIONS != True -> no dumps to analyze). Not marked done."
else
    PERCOND_CONDS="A B C D"
    [ "$DSA_INCLUDE_E" = "true" ] && PERCOND_CONDS="$PERCOND_CONDS E"
    for C in $PERCOND_CONDS; do
        DUMP="${DSA_DIR}/dream_deter_${C}.pkl"
        SAVE="${DSA_DIR}/manifold_${C}"
        echo "--- condition $C ---"
        if [ ! -f "$DUMP" ]; then
            echo "  WARN: dump missing ($DUMP) — skipping condition $C"; continue
        fi
        set +e
        MPLBACKEND=Agg python dreamerv3/manifold_analysis.py \
            --data "$TRAJ_DIR" --dream_data "$DUMP" --save "$SAVE" \
            --layers $PERCOND_LAYERS --max_wake_samples $PERCOND_MAX_WAKE \
            --max_dream_samples $PERCOND_MAX_DREAM --n_neighbors $PERCOND_N_NEIGHBORS
        rc=$?; set -e
        [ $rc -ne 0 ] && echo "  WARN: condition $C manifold returned $rc (continuing)"
    done
    mark_done 11
fi

# =====================================================================
# STEP 12: CONDITION-OVERLAY MANIFOLD (SW-over-time A/B/C/D vs shared wake)
# =====================================================================
echo ""; echo "============================================================"
echo "  STEP 12/13: CONDITION-OVERLAY MANIFOLD (SW-over-time)"; echo "============================================================"
if step_done 12; then
    skip_note 12
elif [ "$OVERLAY_MANIFOLD" != "true" ]; then
    echo "STEP 12: SKIPPED (OVERLAY_MANIFOLD != true). Not marked done."
elif [ "$DSA_SAVE_ACTIVATIONS" != "True" ]; then
    echo "STEP 12: SKIPPED (DSA_SAVE_ACTIVATIONS != True -> no dumps to overlay). Not marked done."
else
    MPLBACKEND=Agg python dreamerv3/manifold_analysis.py \
        --data "$TRAJ_DIR" --seed_ablation_dir "$DSA_DIR" --save "$OVERLAY_SAVE" \
        --layers $OVERLAY_LAYERS --max_wake_samples $OVERLAY_MAX_WAKE \
        --max_dream_samples $OVERLAY_MAX_DREAM --shading $OVERLAY_SHADING --min_bbox 0
    mark_done 12
fi

# =====================================================================
# STEP 12b: CLEAN UP DSA DUMPS (the 32-64 GB dream_deter_{A,B,C,D}.pkl)
# They are inputs to steps 11+12 only; nothing reads them afterward. Deleted
# iff BOTH consuming steps are complete (or disabled), so a resume that still
# needs them keeps them. Gated by DSA_CLEANUP_DUMPS.
# =====================================================================
if [ "$DSA_CLEANUP_DUMPS" = "true" ] && [ "$DSA_SAVE_ACTIVATIONS" = "True" ]; then
    _s11ok=false; _s12ok=false
    { step_done 11 || [ "$PERCOND_MANIFOLD" != "true" ]; } && _s11ok=true
    { step_done 12 || [ "$OVERLAY_MANIFOLD" != "true" ]; } && _s12ok=true
    if [ "$_s11ok" = true ] && [ "$_s12ok" = true ]; then
        _freed=0
        for C in A B C D E; do
            DUMP="${DSA_DIR}/dream_deter_${C}.pkl"
            if [ -f "$DUMP" ]; then
                _sz=$(du -m "$DUMP" 2>/dev/null | cut -f1); _freed=$(( _freed + ${_sz:-0} ))
                rm -f "$DUMP"
                echo "  [cleanup] removed $DUMP (${_sz:-?} MB)"
            fi
        done
        echo "  [cleanup] freed ~${_freed} MB of DSA deter dumps (DSA_CLEANUP_DUMPS=false to keep)"
    else
        echo "  [cleanup] DSA dumps kept (steps 11/12 not both complete yet)"
    fi
fi

# =====================================================================
echo ""; echo "============================================================"
if [ "$RUN_ANALYSIS" = "true" ]; then echo "  ALL 13 STEPS COMPLETE"
else echo "  PIPELINE COMPLETE (steps 1,2,3,3b; RUN_ANALYSIS=$RUN_ANALYSIS -> 4-9 skipped)"; fi
echo "  Results: $LOGDIR"
echo "  Overlay: $OVERLAY_SAVE/dyn_deter_sw_over_time_by_condition.png"
echo "  Vitals:  $VITAL_DYNAMICS_SAVE/"
echo "============================================================"
