#!/usr/bin/env bash
# monitor_v2_v255_live.sh - v2.5.5 live monitor (declarative method architecture) (24h-mle / smoke / lite / any TASK_LIST)
# Continuously refreshing monitor: prints a compact snapshot only when state
# changes, and tails the live task logs ([llm] calls + daemon details).
# Usage:
#   bash monitor_v2_v255_live.sh                 # auto-pick newest run
#   bash monitor_v2_v255_live.sh <STAMP>         # pin a specific run
#   FOLLOW_TASK=hubmap-kidney-segmentation bash monitor_v2_v255_live.sh <STAMP>
#   REFRESH_SECS=3 bash monitor_v2_v255_live.sh  # faster refresh
#   STATE_ROOT=/mnt/data/v2_state_lite bash monitor_v2_v255_live.sh   # lite trio
set +e
INCOMING=/mnt/data/stage42_delivery/incoming
if [ -n "${TASK_LIST:-}" ]; then
    IFS=',' read -r -a TASKS <<< "$TASK_LIST"
else
    TASKS=(aerial-cactus-identification aptos2019-blindness-detection dog-breed-identification)
fi
REFRESH_SECS="${REFRESH_SECS:-6}"

root_now() {
  # Auto-detect the state root (v2_state / v2_state_lite) that holds the newest run.
  local first="${TASKS[0]}"
  local cand best=""
  for cand in "${STATE_ROOT:-}" /mnt/data/v2_state_lite /mnt/data/v2_state /home/xzr/v2_state; do
    [ -n "$cand" ] || continue
    local hit
    hit="$(ls -dt "$cand"/run_v2_${first}_* 2>/dev/null | head -1)"
    if [ -n "$hit" ] && { [ -z "$best" ] || [ "$hit" \> "$best" ]; }; then
      best="$hit"
    fi
  done
  [ -n "$best" ] && dirname "$best"
}

stamp_now() {
  local r; r="$(root_now)"
  [ -n "$r" ] && ls -dt "$r"/run_v2_${TASKS[0]}_* 2>/dev/null \
    | head -1 | grep -oE '[0-9]{8}T[0-9]{6}Z'
}

outer_log() {
  ls -t "$INCOMING"/run_v2_*_outer.log 2>/dev/null | head -1
}

budget_line() {
  python3 - "$1" <<'PY' 2>/dev/null
import json, sys, time
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print("no budget_state")
    raise SystemExit
left = ""
dl = d.get("wall_deadline_epoch") or 0
if dl:
    l = int(dl - time.time())
    left = ("%dh%02dm" % (l // 3600, (l % 3600) // 60)) if l >= 0 else "ended"
print("grants=%s/%s trials=%s/%s wall_left=%s" % (
    d.get("grants_committed", "?"), d.get("max_grants", "?"),
    d.get("trials_reserved", "?"), d.get("max_total_trials", "?"), left))
PY
}

stage_line() {
  python3 - "$1" <<'PY' 2>/dev/null
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print("%s grants_seen=%s" % (d.get("current_stage", "?"), d.get("grants_seen", "?")))
except Exception:
    print("no stage_state")
PY
}

signature() {
  local s="$STAMP"
  local t
  for t in "${TASKS[@]}"; do
    local f="$INCOMING/run_v2_${t}_${STAMP}.log"
    s="$s|$(grep -c 'receipt: verdict' "$f" 2>/dev/null)"
    s="$s|$(grep -c 'grant plan:' "$f" 2>/dev/null)"
    s="$s|$(grep -c '\[llm\] FAIL' "$f" 2>/dev/null)"
    s="$s|$(grep -c 'NEW BEST' "$f" 2>/dev/null)"
  done
  echo "$s"
}

gpu_rows() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null
}

show_header() {
  local OLOG; OLOG="$(outer_log)"
  local RUN_GPU=""
  if [ -n "$OLOG" ]; then
    RUN_GPU="$(grep -oE 'AUTO_GPU=[0-9]+' "$OLOG" | tail -1 | cut -d= -f2)"
    [ -z "$RUN_GPU" ] && RUN_GPU="$(grep -oE 'GPU_MAP=[0-9,]+' "$OLOG" | tail -1 | cut -d= -f2 | cut -d, -f1)"
  fi
  printf '\n===== V2.5.2 LIVE %s STAMP=%s (root=%s) =====\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$STAMP" "${ACTIVE_ROOT:-?}"
  echo "----- OUTER: $OLOG -----"
  grep -E "MODE=|LLM_STATUS|\[5/7\]|\[6/7\]|Waiting|V2_OFFLINE_TESTS|V2_INSTALL_VERIFY" "$OLOG" 2>/dev/null | tail -8
  echo "----- OUTER TAIL -----"
  tail -n 6 "$OLOG" 2>/dev/null
  local t
  for t in "${TASKS[@]}"; do
    local f="$INCOMING/run_v2_${t}_${STAMP}.log"
    local sd="${ACTIVE_ROOT:-/mnt/data/v2_state_lite}/run_v2_${t}_${STAMP}"
    echo "### $t"
    [ -f "$f" ] || { echo "  (no log yet)"; continue; }
    echo "  stage=$(stage_line "$sd/stage_state.json") | budget=$(budget_line "$sd/budget_state.json")"
    echo "  receipts=$(grep -c 'receipt: verdict' "$f") ok=$(grep -c 'verdict=success' "$f") fail=$(grep -c 'verdict=failure' "$f") stag=$(grep -c 'verdict=stagnant' "$f") | llm_fail=$(grep -c '\[llm\] FAIL' "$f") reject=$(grep -c 'CODE_QUALITY_REJECT' "$f")"
    echo "  best> $(grep 'NEW BEST' "$f" | tail -1 | sed 's/^\[[0-9:]*\] //')"
    echo "  restored> $(grep 'restored scientific state' "$f" | tail -1 | sed 's/^\[[0-9:]*\] //')"
    echo "  grant> $(grep -E 'grant plan:|grant:' "$f" | tail -1 | sed 's/^\[[0-9:]*\] //')"
    echo "  rcpt> $(grep 'receipt: verdict' "$f" | tail -1 | sed 's/^\[[0-9:]*\] //')"
    echo "  llm>  $(grep -E '\[llm\] (START|OK|FAIL)' "$f" | tail -1 | sed 's/^\[[0-9:]*\] //')"
    echo "  layout> $(grep -hE 'profile: task_type=|manifest: layout|image cache:|synthesize:|sanitize:' "$f" | tail -2 | sed 's/^\[[0-9:]*\] //' | tr '\n' ' | ')"
    local dl; dl="$(ls -t "$sd"/host_daemon_*.log 2>/dev/null | head -1)"
    if [ -n "$dl" ]; then
      echo "  daemon> $(basename "$dl")"
      tail -n 2 "$dl" 2>/dev/null | sed 's/^/    /'
    else
      echo "  daemon> (none yet)"
    fi
  done
  echo "----- GPU (run GPU=${RUN_GPU:-auto}) -----"
  local row
  while IFS= read -r row; do
    if [ -n "$RUN_GPU" ] && [[ "$row" == "$RUN_GPU,"* ]]; then
      echo "  * $row"
    else
      echo "    $row"
    fi
  done < <(gpu_rows)
  echo "----- PROCESSES -----"
  ps aux | grep -E "v2_closed_loop|v2_host_daemon|run_v2_a100_3tasks" | grep -v grep \
    | awk '{printf "  pid=%s cpu=%s mem=%s start=%s %s\n", $2, $3, $4, $9, substr($0, index($0,$11), 110)}'
  echo "----- HINT -----"
  echo "  Full log: tail -n 200 $INCOMING/run_v2_<task>_${STAMP}.log   (daemon: ls -t ${ACTIVE_ROOT:-$STATE_ROOT}/run_v2_<task>_${STAMP}/host_daemon_*.log)"
  echo "  Follow one task: FOLLOW_TASK=<task> bash monitor_v2_v255_live.sh $STAMP"
}

ACTIVE_ROOT="$(root_now)"
PIN="${1:-}"
STAMP="${PIN:-$(stamp_now)}"
LAST=""
while :; do
  local_root="$(root_now)"
  [ -n "$local_root" ] && ACTIVE_ROOT="$local_root"
  if [ -z "$PIN" ]; then
    STAMP="$(stamp_now)"   # track the newest run unless pinned
  fi
  if [ -z "$STAMP" ]; then
    echo "[$(date -u '+%H:%M:%S')] waiting for state dir under v2_state / v2_state_lite ..."
    tail -n 20 "$(outer_log)" 2>/dev/null
    sleep 3
    continue
  fi
  S="$(signature)"
  if [ "$S" != "$LAST" ]; then
    show_header
    LAST="$S"
  fi
  files=()
  if [ -n "$FOLLOW_TASK" ]; then
    files+=("$INCOMING/run_v2_${FOLLOW_TASK}_${STAMP}.log")
  else
    for t in "${TASKS[@]}"; do files+=("$INCOMING/run_v2_${t}_${STAMP}.log"); done
  fi
  timeout "${REFRESH_SECS}s" tail -n 0 -F "${files[@]}" 2>/dev/null
done