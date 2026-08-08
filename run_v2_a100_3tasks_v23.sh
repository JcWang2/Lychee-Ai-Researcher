#!/usr/bin/env bash
# V2.3 template-compiled closed loop: run N MLE-Bench tasks concurrently (default 3).
# Run on A100 as xzr. Requires the v2.3 tar.gz already uploaded to incoming
# (or point V2_PKG_DIR/V2_PKG_TAR at a newer package, e.g. v2.3.1).
#
# Usage:
#   bash run_v2_a100_3tasks_v23.sh [PRESET] [MAX_GRANTS]
#   bash run_v2_a100_3tasks_v23.sh 24h-mle
#   bash run_v2_a100_3tasks_v23.sh 24h-sota
#   bash run_v2_a100_3tasks_v23.sh smoke
#     GPU_ID      default: auto-select the freest GPU (nvidia-smi)
#     MAX_GRANTS  default: 3 per task
#     "24h-mle"   v2.3 generic preset: MAX_GRANTS=128 / MAX_TOTAL_TRIALS=256 /
#                 TOTAL_WALL_CLOCK=86400s (highest authority), ROUND_TIMEOUT
#                 =3600s, stagnation_limit=48, PRETRAINED_POLICY=cache.
#                 Per-grant children are chosen by HERA's research intent
#                 (feasibility 1-2, cheap_probe 2-4, exploitation 2-3,
#                 expensive_structural 1-2, confirmation 2-3, final 1).
#     "smoke"     v2.3 smoke preset: MAX_GRANTS=4 / MAX_TOTAL_TRIALS=12 /
#                 TOTAL_WALL_CLOCK=3600s / ROUND_TIMEOUT=600s (before 24h-mle).
#     "24h-sota"  legacy SOTA preset (kept): 72 max rounds, 2400s per-trial
#                 timeout, trial_budget=3, stagnation_limit=24, 86400s total
#
#   V2_GPU_MAP="4,5,6"  per-task GPU sharding: task i runs on GPU list[i % n].
#                       When set, the single-GPU auto/requested logic is
#                       bypassed. Unset (default) keeps the old behavior.
#   V2_CPU=1            CPU-only mode: no GPU selection, no --gpus docker
#                       flag; implementer prompt is switched to CPU-only
#                       contract. Pairs well with "64"/"128" round presets.
#   "64" / "128"        full-day presets capped at 64 / 128 rounds
#                       (aliases 24h-64 / 24h-128), 3600s round timeout,
#                       stagnation 24 / 48, total wall clock 86400s.
#
# Tasks: aerial-cactus-identification, aptos2019-blindness-detection,
#        dog-breed-identification (data layout already OK on A100).
set -euo pipefail

# v2.3: HERA decides (method/axis/intent), ProgramCompiler renders deterministic
# scripts. The knobs below bound the legacy implementer fallback only:
# codegen uses 600s per-chunk read + 1200s overall cap (env-overridable)
# so a slow-but-alive generation completes instead of ReadTimeout-
# falling back every child. Attempts stay 2; tokens 4000.
export LLM_CODE_TIMEOUT="${LLM_CODE_TIMEOUT:-600}"
export LLM_CODE_TOTAL="${LLM_CODE_TOTAL:-1200}"
export LLM_CODE_ATTEMPTS="${LLM_CODE_ATTEMPTS:-2}"
export LLM_CODE_MAX_TOKENS="${LLM_CODE_MAX_TOKENS:-4000}"
export MAX_SYNTHESIS_ACTIONS="${MAX_SYNTHESIS_ACTIONS:-2}"
INCOMING=/mnt/data/stage42_delivery/incoming
# Package to unpack/install. Override for newer builds:
#   V2_PKG_DIR=ai_scientist_execution_layer_v2_20260806_v231 \
#   V2_PKG_TAR=ai_scientist_execution_layer_v2_20260806_v231.tar.gz \
#   bash run_v2_a100_3tasks_v23.sh smoke
PKG_DIR="$INCOMING/${V2_PKG_DIR:-ai_scientist_execution_layer_v2_20260806_v23}"
PKG_TAR="$INCOMING/${V2_PKG_TAR:-ai_scientist_execution_layer_v2_20260806_v23.tar.gz}"
STAMP="$(date +%Y%m%dT%H%M%SZ)"
# Writable root for per-task state/code/submission. Override with STATE_ROOT=...
RUN_ROOT="${STATE_ROOT:-/home/xzr/v2_state}"
DEPLOY_ROOT=/mnt/data/stage42_deployments/20260803T000000Z_legacy_l1_v2
TARGET="$DEPLOY_ROOT/MLE-bench/agents/aisci"
PYTHON_BIN=/mnt/data/pact_hera_r231_control_env/py310/bin/python

TASKS=(
  "aerial-cactus-identification|Aerial cactus: binary image classification (has_cactus 0/1)"
  "aptos2019-blindness-detection|APTOS 2019: diabetic retinopathy severity 0-4 from retina images"
  "dog-breed-identification|Dog breed: multi-class image classification (120 breeds)"
)

# Generalization: run ANY MLE-Bench competition (or a curated subset) by
# overriding the task list. Format: comma-separated "id|prompt" entries.
#   TASK_LIST='aerial-cactus-identification|cactus,aptos2019-blindness-detection|retina' bash run_v2_a100_3tasks.sh 24h
# Metrics/directions for all 82 MLE-Bench competitions come from
# metrics_registry.py (exact/proxy/inferred), so no per-competition code
# changes are needed.
# v2.3.7 launcher fix: prompts may contain commas ("(regression, RMSE)") -
# splitting on ',' alone shreds the list into junk entries. A new
# "id|prompt" entry starts only at a field that contains '|'; comma-only
# fields are re-joined into the previous entry's prompt. Task ids never
# contain '|' or ',' (MLE-Bench kebab-case).
if [ -n "${TASK_LIST:-}" ]; then
    IFS=',' read -r -a _TASK_FIELDS <<< "$TASK_LIST"
    TASKS=()
    _cur=""
    for _f in "${_TASK_FIELDS[@]}"; do
        if [[ "$_f" == *"|"* ]]; then
            [ -n "$_cur" ] && TASKS+=("$_cur")
            _cur="$_f"
        else
            _cur="${_cur:+$_cur,}$_f"
        fi
    done
    [ -n "$_cur" ] && TASKS+=("$_cur")
    unset _TASK_FIELDS _cur _f
fi

GPU_ID="${1:-auto}"
MAX_ROUNDS="${2:-3}"

if [ "${1:-}" = "24h-mle" ] || [ "${1:-}" = "24h" ]; then
    GPU_ID=auto
    MAX_GRANTS="${V2_MLE_MAX_GRANTS:-128}"
    MAX_ROUNDS="${V2_MLE_MAX_ROUNDS:-$MAX_GRANTS}"
    MAX_TOTAL_TRIALS="${V2_MLE_MAX_TOTAL_TRIALS:-256}"
    TOTAL_WALL_CLOCK="${V2_MLE_TOTAL_WALL_CLOCK:-86400}"
    PACT_TOTAL_WALL_CLOCK_SECONDS="$TOTAL_WALL_CLOCK"
    ROUND_TIMEOUT="${V2_MLE_TIMEOUT:-3600}"
    V2_STAGNATION_LIMIT="${V2_MLE_STAGNATION:-48}"
    PRETRAINED_POLICY="${V2_MLE_PRETRAINED_POLICY:-cache}"
    echo "MODE=24H-MLE(v2.3): grants=$MAX_GRANTS trials=$MAX_TOTAL_TRIALS total_wall_clock=${TOTAL_WALL_CLOCK}s round_timeout=$ROUND_TIMEOUT stagnation=$V2_STAGNATION_LIMIT pretrained=$PRETRAINED_POLICY"
fi

if [ "${1:-}" = "24h-sota" ]; then
    GPU_ID=auto
    MAX_GRANTS="${V2_SOTA_GRANTS:-128}"
    MAX_ROUNDS="${V2_SOTA_MAX_ROUNDS:-$MAX_GRANTS}"
    MAX_TOTAL_TRIALS="${V2_SOTA_TRIALS:-256}"
    TOTAL_WALL_CLOCK="${V2_SOTA_TOTAL:-86400}"
    PACT_TOTAL_WALL_CLOCK_SECONDS="$TOTAL_WALL_CLOCK"
    ROUND_TIMEOUT="${V2_SOTA_TIMEOUT:-2400}"
    V2_STAGNATION_LIMIT="${V2_SOTA_STAGNATION:-24}"
    TRIAL_BUDGET="${V2_SOTA_TRIAL_BUDGET:-3}"
    PRETRAINED_POLICY="${V2_SOTA_PRETRAINED_POLICY:-cache}"
    echo "MODE=24H-SOTA(v2.3): grants=$MAX_GRANTS trials=$MAX_TOTAL_TRIALS round_timeout=$ROUND_TIMEOUT trial_budget=$TRIAL_BUDGET total_wall_clock=${TOTAL_WALL_CLOCK}s stagnation=$V2_STAGNATION_LIMIT pretrained=$PRETRAINED_POLICY"
fi

if [ "${1:-}" = "smoke" ] || [ "${1:-}" = "smoke-v23" ]; then
    GPU_ID=auto
    MAX_GRANTS="${V2_SMOKE_MAX_GRANTS:-4}"
    MAX_ROUNDS="${V2_SMOKE_MAX_ROUNDS:-$MAX_GRANTS}"
    MAX_TOTAL_TRIALS="${V2_SMOKE_MAX_TOTAL_TRIALS:-12}"
    TOTAL_WALL_CLOCK="${V2_SMOKE_TOTAL_WALL_CLOCK:-3600}"
    PACT_TOTAL_WALL_CLOCK_SECONDS="$TOTAL_WALL_CLOCK"
    ROUND_TIMEOUT="${V2_SMOKE_TIMEOUT:-600}"
    V2_STAGNATION_LIMIT="${V2_SMOKE_STAGNATION:-4}"
    PRETRAINED_POLICY="${V2_SMOKE_PRETRAINED_POLICY:-cache}"
    echo "MODE=SMOKE(v2.3): grants=$MAX_GRANTS trials=$MAX_TOTAL_TRIALS total_wall_clock=${TOTAL_WALL_CLOCK}s round_timeout=$ROUND_TIMEOUT stagnation=$V2_STAGNATION_LIMIT pretrained=$PRETRAINED_POLICY"
fi
if [ "${1:-}" = "64" ] || [ "${1:-}" = "24h-64" ]; then
    V2_CPU="${V2_CPU:-1}"
    MAX_ROUNDS="${V2_64_ROUNDS:-64}"
    ROUND_TIMEOUT="${V2_64_TIMEOUT:-3600}"
    PACT_TOTAL_WALL_CLOCK_SECONDS="${V2_64_TOTAL:-86400}"
    V2_STAGNATION_LIMIT="${V2_64_STAGNATION:-24}"
    echo "MODE=64: rounds=$MAX_ROUNDS round_timeout=$ROUND_TIMEOUT total_wall_clock=${PACT_TOTAL_WALL_CLOCK_SECONDS}s stagnation=$V2_STAGNATION_LIMIT cpu=$V2_CPU"
fi

if [ "${1:-}" = "128" ] || [ "${1:-}" = "24h-128" ]; then
    V2_CPU="${V2_CPU:-1}"
    MAX_ROUNDS="${V2_128_ROUNDS:-128}"
    ROUND_TIMEOUT="${V2_128_TIMEOUT:-3600}"
    PACT_TOTAL_WALL_CLOCK_SECONDS="${V2_128_TOTAL:-86400}"
    V2_STAGNATION_LIMIT="${V2_128_STAGNATION:-48}"
    echo "MODE=128: rounds=$MAX_ROUNDS round_timeout=$ROUND_TIMEOUT total_wall_clock=${PACT_TOTAL_WALL_CLOCK_SECONDS}s stagnation=$V2_STAGNATION_LIMIT cpu=$V2_CPU"
fi
# V2.3: independent resident HostSupervisorService daemon per task.
# Set V2_HOST_DAEMON=0 to run the host inline inside the director instead.
V2_HOST_DAEMON="${V2_HOST_DAEMON:-1}"

if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(command -v python3 || command -v python)"
fi
[ -n "$PYTHON_BIN" ] || { echo "FAIL: no python found"; exit 1; }

# LLM env: source the shared env file (API key/base/model), if present.
# Override the path with V2_LLM_ENV=... ; no key -> deterministic fallback.
LLM_ENV_FILE="${V2_LLM_ENV:-/mnt/data/stage42_delivery/latest_ai_scientist_v6.env}"
if [ -f "$LLM_ENV_FILE" ]; then
    echo "LLM_ENV_FILE=$LLM_ENV_FILE"
    set +u
    # shellcheck disable=SC1090
    source "$LLM_ENV_FILE"
    set -u
else
    echo "LLM_ENV_FILE=$LLM_ENV_FILE (missing -> LLM disabled)"
fi
if [ -n "${OPENAI_API_KEY:-}" ]; then
    echo "LLM_STATUS=READY model=${LLM_MODEL:-qwen3.8-max} base=${OPENAI_BASE_URL:-default}"
else
    echo "LLM_STATUS=UNSET (PACT will use deterministic fallback)"
fi
echo "EXEC_MODE_IMAGE=${V2_EXEC_IMAGE:-host-subprocess}"
echo "V2_PREFLIGHT=${V2_PREFLIGHT:-strict} V2_DETERMINISTIC_BASELINE=${V2_DETERMINISTIC_BASELINE:-1} V2_HOST_DAEMON=$V2_HOST_DAEMON V2_TORCH_CACHE=${V2_TORCH_CACHE:-none} V2_HF_CACHE=${V2_HF_CACHE:-none}"

echo "===== [1/7] unpack + verify package ====="
[ -f "$PKG_TAR" ] || { echo "FAIL: $PKG_TAR missing (upload first)"; exit 1; }
rm -rf "$PKG_DIR"
mkdir -p "$INCOMING"
tar xzf "$PKG_TAR" -C "$INCOMING"
cd "$PKG_DIR"
sha256sum -c MANIFEST.sha256

echo "===== [2/7] install into deploy tree ====="
mkdir -p "$TARGET"
bash ./install_v2_execution_layer.sh --target "$TARGET" --run-tests

if [ -f "$TARGET/v2_closed_loop.py" ]; then
    RUN_SCRIPT="$TARGET/v2_closed_loop.py"
else
    RUN_SCRIPT="$PKG_DIR/payload/agents/aisci/v2_closed_loop.py"
fi
[ -f "$RUN_SCRIPT" ] || { echo "FAIL: v2_closed_loop.py not found after install"; exit 1; }
echo "RUN_SCRIPT=$RUN_SCRIPT"

echo "===== [3/7] GPU selection ====="
GPU_MAP=()
if [ "${V2_CPU:-0}" = "1" ]; then
    echo "CPU_MODE=1 (no GPU selection; tasks run CPU-only)"
else
if [ -n "${V2_GPU_MAP:-}" ]; then
    # Per-task GPU map: V2_GPU_MAP="4,5,6" -> task i uses GPU_MAP[i % n].
    _gpus=()
    IFS=',' read -r -a _gpus <<< "$V2_GPU_MAP"
    for _g in "${_gpus[@]}"; do
        _g="$(echo "$_g" | tr -d ' ')"
        if echo "$_g" | grep -Eq '^[0-9]+$'; then
            GPU_MAP+=("$_g")
        else
            echo "GPU_MAP_WARN: ignoring invalid entry '$_g'"
        fi
    done
    if [ "${#GPU_MAP[@]}" -eq 0 ]; then
        echo "GPU_MAP_WARN: V2_GPU_MAP='$V2_GPU_MAP' has no valid GPUs; falling back to auto"
        V2_GPU_MAP=""
    fi
fi
if [ "${#GPU_MAP[@]}" -gt 0 ]; then
    echo "GPU_MAP=${V2_GPU_MAP} -> task i runs on GPU ${GPU_MAP[0]}... (cycles if more tasks than GPUs)"
    if command -v nvidia-smi >/dev/null 2>&1; then
        for _g in "${GPU_MAP[@]}"; do
            _line=$(nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
                --format=csv,noheader,nounits | sed -n "$((_g+1))p")
            if [ -n "$_line" ]; then
                echo "  GPU $_g: $_line"
            else
                echo "  GPU $_g: NOT VISIBLE (nvidia-smi reports fewer GPUs)"
            fi
        done
    fi
elif [ "$GPU_ID" = "auto" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_ID=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
            --format=csv,noheader,nounits | \
            awk -F', ' '{ if ($2/$3 < best || best == "") { best=$2/$3; gpu=$1 } } END { print gpu }')
        echo "AUTO_GPU=$GPU_ID (freest by memory usage)"
    else
        GPU_ID=0
        echo "AUTO_GPU=0 (no nvidia-smi; defaulting to 0)"
    fi
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits | sed -n "$((GPU_ID+1))p" || echo "(gpu $GPU_ID not visible)"
else
    echo "REQUESTED_GPU=$GPU_ID"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits | sed -n "$((GPU_ID+1))p" || echo "(gpu $GPU_ID not visible)"
fi
fi

# v2.5.5: layout preflight. TASK_DATA_OK now means resolve_dataset_layout
# succeeds - MLE-Bench prepare quirks (localized prefixes, zipped tables)
# are caught BEFORE launch instead of a silent closed-loop crash. Generic;
# no competition names. Disable with V2_DATA_PREFLIGHT=0.
preflight_layout() {
    local task="$1" dir="$2" out rc
    [ "${V2_DATA_PREFLIGHT:-1}" = "1" ] || return 0
    out="$("$PYTHON_BIN" -c "
import sys
sys.path.insert(0, sys.argv[1])
from data_layout import resolve_dataset_layout
try:
    d = resolve_dataset_layout(sys.argv[2])
except Exception as e:
    print('PREFLIGHT_RESOLVE_FAIL %r' % (e,))
    sys.exit(1)
print('layout=%s train=%s test=%s labels=%s' % (d.layout_name, d.train_path.name, d.test_path.name, d.test_has_labels))
" "$TARGET" "$dir" 2>&1)"
    rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "  preflight: $out"
        return 0
    fi
    echo "  preflight FAIL: $out" >&2
    return 1
}
echo "===== [4/7] task data readiness ====="
# Candidate data roots. Override with TASK_DATA_ROOT=<root> if your layout differs.
CANDIDATE_ROOTS=(
    "${TASK_DATA_ROOT:-}"
    "/mnt/data/mle-bench/data"
    "/mnt/data/mlebench_repair/openai_mle_bench/mlebench/competitions"
    "/mnt/data/pact_stage421_deploy_snapshot_20260715T133351Z/mlebench/competitions"
    "/mnt/data/pact_stage421_deploy_payload_snapshot_20260715T133619Z/mlebench/competitions"
    "/home/data"
    "/mnt/data/mle_lite"
    "/mnt/data/mle-lite"
    "/mnt/data/stage42_delivery/mle_lite"
    "/mnt/data/stage42_delivery/MLE-bench/mlebench/competitions"
)
resolve_data_dir() {
    local task="$1"
    local root
    for root in "${CANDIDATE_ROOTS[@]}"; do
        [ -n "$root" ] || continue
        if [ -d "$root/$task" ]; then
            echo "$root/$task"
            return 0
        fi
    done
    return 1
}

DATA_DIRS=()
for entry in "${TASKS[@]}"; do
    task="${entry%%|*}"
    dir="$(resolve_data_dir "$task" || true)"
    if [ -n "$dir" ]; then
        if preflight_layout "$task" "$dir"; then
            DATA_DIRS+=("$task|$dir")
            echo "TASK_DATA_OK: $task -> $dir"
            ls "$dir" | head -5
        else
            DATA_DIRS+=("$task|")
            echo "TASK_DATA_FAIL: $task (layout not resolvable; skipping)"
        fi
    else
        DATA_DIRS+=("$task|")
        echo "TASK_DATA_WARN: $task not found under any candidate root (set TASK_DATA_ROOT)"
    fi
done

if [ "${V2_CPU:-0}" = "1" ]; then
    echo "===== [5/7] launch 3 tasks concurrently on CPU ====="
elif [ "${#GPU_MAP[@]}" -gt 0 ]; then
    echo "===== [5/7] launch 3 tasks concurrently on GPUs ${GPU_MAP[*]} ====="
else
    echo "===== [5/7] launch 3 tasks concurrently on GPU $GPU_ID ====="
fi
mkdir -p "$RUN_ROOT"
echo "RUN_ROOT=$RUN_ROOT (override with STATE_ROOT=...)"
PIDS=()
LOGS=()
i=0
for entry in "${TASKS[@]}"; do
    i=$((i+1))
    task="${entry%%|*}"
    prompt="${entry#*|}"
    STATE_DIR="$RUN_ROOT/run_v2_${task}_${STAMP}"
    LOG="$INCOMING/run_v2_${task}_${STAMP}.log"
    mkdir -p "$STATE_DIR"

    # use the resolved data dir from step 4
    DATA_DIR="${DATA_DIRS[$((i-1))]#*|}"
    if [ -z "$DATA_DIR" ] || [ ! -d "$DATA_DIR" ]; then
        echo ">>> [$i] $task: DATA_DIR_MISSING (skipping)"
        continue
    fi
    mkdir -p "$RUN_ROOT/code/$task" "$RUN_ROOT/submission/$task"

    if [ "${V2_CPU:-0}" = "1" ]; then
        TASK_GPU=""
        GPU_NOTE="(CPU)"
    elif [ "${#GPU_MAP[@]}" -gt 0 ]; then
        TASK_GPU="${GPU_MAP[$(( (i-1) % ${#GPU_MAP[@]} ))]}"
        GPU_NOTE="(map)"
    else
        TASK_GPU="$GPU_ID"
        GPU_NOTE=""
    fi
    if [ "${V2_CPU:-0}" = "1" ]; then
        echo ">>> [$i] $task on CPU (rounds=$MAX_ROUNDS) data=$DATA_DIR"
    else
        echo ">>> [$i] $task on GPU $TASK_GPU $GPU_NOTE (rounds=$MAX_ROUNDS) data=$DATA_DIR"
    fi
    CUDA_VISIBLE_DEVICES="$TASK_GPU" \
    PACT_STAGE4_SELF_EVOLUTION=0 \
    MAX_ROUNDS="${MAX_ROUNDS:-${MAX_GRANTS:-128}}" \
    MAX_GRANTS="${MAX_GRANTS:-128}" \
    MAX_TOTAL_TRIALS="${MAX_TOTAL_TRIALS:-256}" \
    ROUND_TIMEOUT="${ROUND_TIMEOUT:-3600}" \
    TOTAL_WALL_CLOCK="${TOTAL_WALL_CLOCK:-${PACT_TOTAL_WALL_CLOCK_SECONDS:-86400}}" \
    PACT_TOTAL_WALL_CLOCK_SECONDS="${PACT_TOTAL_WALL_CLOCK_SECONDS:-${TOTAL_WALL_CLOCK:-86400}}" \
    PRETRAINED_POLICY="${PRETRAINED_POLICY:-cache}" \
    STATE_DIR="$STATE_DIR" \
    OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
    OPENAI_BASE_URL="${OPENAI_BASE_URL:-}" \
    LLM_MODEL="${LLM_MODEL:-qwen3.8-max}" \
    V2_STAGNATION_LIMIT="${V2_STAGNATION_LIMIT:-6}" \
    V2_PREFLIGHT="${V2_PREFLIGHT:-strict}" \
    V2_DETERMINISTIC_BASELINE="${V2_DETERMINISTIC_BASELINE:-1}" \
    V2_HOST_DAEMON="$V2_HOST_DAEMON" \
    V2_TORCH_CACHE="${V2_TORCH_CACHE:-}" \
    V2_HF_CACHE="${V2_HF_CACHE:-}" \
    V2_CPU_ONLY="${V2_CPU:-0}" \
    V2_S1_MAX_GRANTS="${V2_S1_MAX_GRANTS:-8}" \
    MAX_SYNTHESIS_ACTIONS="${MAX_SYNTHESIS_ACTIONS:-2}" \
    "$PYTHON_BIN" "$RUN_SCRIPT" \
      --competition "$task" \
      --task-prompt "$prompt" \
      --data-dir "$DATA_DIR" \
      --work-dir "$RUN_ROOT/code/$task" \
      --submission-dir "$RUN_ROOT/submission/$task" \
      --state-dir "$STATE_DIR" \
      --trial-budget "${TRIAL_BUDGET:-3}" \
      > "$LOG" 2>&1 &
    PIDS+=("$!")
    LOGS+=("$LOG")
done

echo "===== [6/7] concurrent run started ====="
if [ "${#PIDS[@]}" -eq 0 ]; then
    echo "NO_TASKS_LAUNCHED (data dirs missing?)"
    exit 1
fi
for idx in "${!PIDS[@]}"; do
    echo "  PID ${PIDS[$idx]} -> ${LOGS[$idx]}"
done
echo "Waiting for all tasks (this blocks until they finish)..."

FAILED=0
for idx in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$idx]}"; then
        echo "TASK_FAILED: ${LOGS[$idx]}"
        echo "--- tail ${LOGS[$idx]} ---"
        tail -80 "${LOGS[$idx]}" || true
        FAILED=$((FAILED+1))
    fi
done

echo "===== [7/7] DONE ====="
if [ "${V2_CPU:-0}" = "1" ]; then
    echo "MODE=CPU (no GPU)"
elif [ "${#GPU_MAP[@]}" -gt 0 ]; then
    echo "GPU_MAP=${V2_GPU_MAP} (task i -> GPU ${GPU_MAP[0]}...)"
else
    echo "GPU_ID=$GPU_ID"
fi
echo "FAILED=$FAILED"
echo "LOGS in $INCOMING/run_v2_*_${STAMP}.log"
echo "Monitor: bash /mnt/data/stage42_delivery/incoming/monitor_v2_v23.sh $STAMP"
exit $FAILED
