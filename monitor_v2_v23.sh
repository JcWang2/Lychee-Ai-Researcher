#!/usr/bin/env bash
# =====================================================================
# monitor_v2_v23.sh - v2.3 template-compiled execution layer live monitor.
# Shows: outer progress, per-task budget/stage, HERA decisions (branch/
# axis/intent), COMPILED/COMPILE_PATCH/SYNTHESIS events, LLM call detail,
# full run-log tail, ledger, proposals/outcomes/receipts, capabilities,
# daemon logs, GPU and processes.
#
# Usage:
#   bash monitor_v2_v23.sh                    # newest batch, live follow
#   bash monitor_v2_v23.sh --once              # single full snapshot, then exit
#   bash monitor_v2_v23.sh <STAMP>             # specific batch
#   bash monitor_v2_v23.sh <STAMP> <task>      # filter to one task
#   bash monitor_v2_v23.sh --once <STAMP> aptos2019-blindness-detection
#
# Env overrides:
#   STATE_ROOT   default /mnt/data/v2_state
#   INCOMING     default /mnt/data/stage42_delivery/incoming
# =====================================================================
set +e
INCOMING="${INCOMING:-/mnt/data/stage42_delivery/incoming}"
RUN_ROOT="${STATE_ROOT:-/mnt/data/v2_state}"
ONCE=0
STAMP=""
TASK_FILTER=""
for a in "$@"; do
  case "$a" in
    --once) ONCE=1 ;;
    *) if [ -z "$STAMP" ]; then STAMP="$a"; else TASK_FILTER="$a"; fi ;;
  esac
done

task_dir() { echo "$RUN_ROOT/run_v2_$1_$STAMP"; }
task_log() { echo "$INCOMING/run_v2_$1_$STAMP.log"; }
grepc() { grep -cE "$1" "$2" 2>/dev/null || echo 0; }
line_count() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
newest_file() {
  find "$1" -maxdepth 1 -name "$2" -type f -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -1 | cut -d' ' -f2-
}

rescan_tasks() {
  TASKS=()
  if [ -z "$STAMP" ]; then
    newest="$(ls -dt "$RUN_ROOT"/run_v2_*_* 2>/dev/null | head -1)"
    [ -n "$newest" ] && STAMP="$(basename "$newest" | grep -oE '[0-9]{8}T[0-9]{6}Z' | head -1)"
  fi
  [ -n "$STAMP" ] || return 0
  for d in "$RUN_ROOT"/run_v2_*_"$STAMP"; do
    [ -d "$d" ] || continue
    task="$(basename "$d")"
    task="${task#run_v2_}"
    task="${task%_$STAMP}"
    if [ -n "$TASK_FILTER" ] && [ "$task" != "$TASK_FILTER" ]; then
      continue
    fi
    TASKS+=("$task")
  done
}

OUTER_LOG="$(ls -t "$INCOMING"/run_v2_*_outer.log 2>/dev/null | head -1)"

py() {
  python3 - "$@" <<'PY'
import json, sys
from pathlib import Path

def clip(v, n=400):
    if not isinstance(v, str):
        v = json.dumps(v, ensure_ascii=False, default=str)
    v = v.replace("\n", " ").strip()
    return v if len(v) <= n else v[:n] + "..."

def load(p):
    if not p:
        return None
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None

mode = sys.argv[1] if len(sys.argv) > 1 else ""
args = sys.argv[2:]

if mode == "proposal":
    d = load(args[0] if args else "")
    if not d:
        print("  (no proposal json)")
        raise SystemExit
    print("  proposal=%s grant=%s child=%s axis=%s method=%s" % (
        d.get("proposal_id"), d.get("grant_id"), d.get("child_index"),
        d.get("mutation_axis"), d.get("method_id") or (d.get("plan") or {}).get("method_id")))
    if d.get("hypothesis"):
        print("  hypothesis: %s" % clip(d["hypothesis"], 420))
    if d.get("param_overrides"):
        print("  params: %s" % clip(d["param_overrides"], 320))
    if d.get("preprocessing"):
        print("  preprocessing: %s" % clip(d["preprocessing"], 320))
elif mode == "outcome":
    d = load(args[0] if args else "")
    if not d:
        print("  (no outcome json)")
        raise SystemExit
    print("  proposal=%s trial=%s verdict=%s metric=%s (%s) rc=%s" % (
        d.get("proposal_id"), d.get("trial_id"), d.get("verdict"),
        d.get("metric"), d.get("metric_name"), d.get("returncode")))
    if d.get("evidence"):
        print("  evidence: %s" % clip(d["evidence"], 420))
    if d.get("failure_reason"):
        print("  failure_reason: %s" % clip(d["failure_reason"], 320))
elif mode == "receipt":
    d = load(args[0] if args else "")
    if not d:
        print("  (no receipt json)")
        raise SystemExit
    print("  receipt=%s verdict=%s metric=%s trial=%s spec=%s" % (
        d.get("receipt_id"), d.get("verdict"), d.get("metric"),
        d.get("trial_id"), d.get("spec_id") or d.get("directive_hash")))
elif mode == "ledger":
    for line in open(args[0], encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            print("  %s" % clip(line, 300))
            continue
        ev = e.get("event") or e.get("type") or "?"
        keys = ("proposal_id", "trial_id", "verdict", "metric", "spec_id",
                "grant_id", "method_id", "branch", "intent", "receipt_id")
        extra = " ".join("%s=%s" % (k, e[k]) for k in keys if e.get(k) is not None)
        print("  [%s] %s" % (ev, clip(extra, 360)))
PY
}

chain_detail() {
  local task="$1" d log
  d="$(task_dir "$task")"
  log="$(task_log "$task")"
  echo ""
  echo "### $task"
  # ---- budget / stage ----
  python3 - "$d/stage_state.json" "$d/budget_state.json" <<'PY'
import json, sys, time
ss = bs = {}
try:
    ss = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    pass
try:
    bs = json.load(open(sys.argv[2], encoding="utf-8"))
except Exception:
    pass
stage = ss.get("current_stage", "-")
committed = bs.get("grants_committed", 0)
max_g = bs.get("max_grants", 0)
trials = bs.get("trials_reserved", 0)
max_t = bs.get("max_total_trials", 0)
dead = bs.get("wall_deadline_epoch") or 0
rem = max(0, int(dead - time.time()))
hh, mm = divmod(rem // 60, 60)
print("  stage=%-14s grants=%d/%d trials=%d/%d wall_left=%dh%02dm" %
      (stage, committed, max_g, trials, max_t, hh, mm))
for k in ("s1_hold", "s2_stagnation", "s2_regressions", "s3_grants", "s4_no_best"):
    if ss.get(k):
        print("  %s=%s" % (k, ss[k]))
if ss.get("entry_best") is not None:
    print("  entry_best=%s norm=%s" % (ss.get("entry_best"), ss.get("entry_norm")))
if ss.get("last_reason"):
    print("  last_reason=%s" % ss["last_reason"])
PY
  # ---- LLM health ----
  if [ -f "$log" ]; then
    echo "  llm_ok=$(grep -cE '\[llm\] OK' "$log") llm_fail=$(grep -cE '\[llm\] FAIL' "$log") codegen_fail=$(grep -cE '\[llm\] FAIL role=codegen' "$log") fallback=$(grep -cE 'LLM chose unknown branch' "$log")"
    # ---- HERA / grant decisions ----
    echo "  -- HERA decisions --"
    grep -E "HERA wrote new branch|\[prioritizer\]|grant plan:|grant: grant_" "$log" 2>/dev/null | tail -8 | sed 's/^/    /'
    # ---- template compilation / synthesis (run log + daemon logs) ----
    echo "  -- COMPILED / PATCH / SYNTHESIS --"
    _clines="$(grep -hE "COMPILED|COMPILE_PATCH|SYNTHESIS" "$log" "$d"/host_daemon_*.log 2>/dev/null | tail -8)"
    if [ -n "$_clines" ]; then
      echo "$_clines" | sed 's/^/    /'
    else
      echo "    (none yet)"
    fi
    # ---- results ----
    echo "  -- results --"
    grep -E "NEW BEST|receipt: verdict=|RESEARCH CYCLE COMPLETE|STAGE TRANSITION|restored scientific state" "$log" 2>/dev/null | tail -6 | sed 's/^/    /'
    # ---- LLM call detail ----
    echo "  -- LLM calls (tail) --"
    grep -E "\[llm\]" "$log" 2>/dev/null | tail -6 | sed 's/^/    /'
    # ---- FULL RUN LOG TAIL (raw) ----
    echo "  -- full log tail (last 25 lines) --"
    tail -n 25 "$log" 2>/dev/null | sed 's/^/    | /'
  else
    echo "  (no run log yet: $log)"
  fi
  # ---- ledger ----
  led="$d/experiment_ledger.jsonl"
  echo "  ledger_lines=$(line_count "$led") receipts=$(ls "$d/budget_receipts"/ 2>/dev/null | wc -l)"
  if [ -f "$led" ]; then
    echo "  -- ledger tail --"
    tail -n 5 "$led" > /tmp/v23_ledger_$$.jsonl 2>/dev/null
    py ledger /tmp/v23_ledger_$$.jsonl
    rm -f /tmp/v23_ledger_$$.jsonl
  fi
  # ---- protocol files (proposals / outcomes / seals) ----
  if [ -d "$d/protocol" ]; then
    echo "  -- protocol recent --"
    find "$d/protocol" -type f \( -name 'proposal_*.json' -o -name 'outcome_*.json' -o -name 'seal_spec_*.json' \) \
      -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -6 | cut -d' ' -f2- | while read -r f; do
        case "$f" in
          *proposal_*) echo "    [proposal] $(basename "$f")"; py proposal "$f" ;;
          *outcome_*)  echo "    [outcome]  $(basename "$f")"; py outcome "$f" ;;
          *seal_spec_*) echo "    [seal]     $(basename "$f") $(python3 -c "import json,sys; d=json.load(open('$f')); print('spec=%s method=%s' % (d.get('spec_id'), (d.get('plan') or {}).get('method_id') or (d.get('invocation') or {}).get('method_id')))" 2>/dev/null)" ;;
        esac
      done
  fi
  # ---- capabilities (Phase C) ----
  if [ -d "$d/capabilities" ]; then
    echo "  -- capabilities --"
    ls -la "$d/capabilities" 2>/dev/null | sed 's/^/    /' | tail -8
  fi
  # ---- run_report ----
  if [ -f "$d/run_report.json" ]; then
    python3 - "$d/run_report.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
print("  -- run_report --")
for k in ("status", "rounds_completed", "best_metric", "certified_best_metric",
          "stage", "submission_exists", "total_time_seconds"):
    print("    %-22s %s" % (k, r.get(k)))
PY
  fi
  # ---- submission ----
  if [ -f "$RUN_ROOT/submission/$task/submission.csv" ]; then
    echo "  submission=YES ($(stat -c %s "$RUN_ROOT/submission/$task/submission.csv" 2>/dev/null)B)"
  fi
  # ---- daemon ----
  dl="$(newest_file "$d" 'host_daemon_*.log')"
  if [ -n "$dl" ]; then
    echo "  -- daemon tail (${dl##*/}) --"
    tail -n 8 "$dl" 2>/dev/null | sed 's/^/    | /'
  fi
}

snapshot() {
  echo ""
  echo "===== V2.3 MONITOR $(date -u +%Y-%m-%dT%H:%M:%SZ) STAMP=$STAMP ====="
  if [ -n "$OUTER_LOG" ] && [ -f "$OUTER_LOG" ]; then
    echo "----- OUTER PROGRESS (${OUTER_LOG#$INCOMING/}) -----"
    grep -hE "MODE=|LLM_STATUS|LLM_ENV_FILE|V2_OFFLINE_TESTS|V2_INSTALL_VERIFY|\[1/7\]|\[5/7\]|\[6/7\]|\[7/7\]|FAILED=|RUN_SCRIPT=" "$OUTER_LOG" 2>/dev/null | tail -8
    echo "----- OUTER TAIL -----"
    tail -n 8 "$OUTER_LOG" 2>/dev/null | sed 's/^/  /'
  fi
  for task in "${TASKS[@]}"; do
    chain_detail "$task"
  done
  echo ""
  echo "----- GPU -----"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/  /'
  echo "----- PROCESSES -----"
  ps aux | grep -E "v2_closed_loop|v2_host_daemon" | grep -v grep \
    | awk '{printf "  pid=%s cpu=%s%% mem=%s%% %s\n", $2, $3, $4, substr($0, index($0,$11), 130)}'
  echo ""
}

signature() {
  for task in "${TASKS[@]}"; do
    d="$(task_dir "$task")"
    log="$(task_log "$task")"
    printf '%s grants=%s llmfail=%s compiled=%s led=%s report=%s\n' \
      "$task" \
      "$(grepc 'grant: grant_' "$log")" \
      "$(grepc '\[llm\] FAIL' "$log")" \
      "$(grepc 'COMPILED' "$log")" \
      "$(line_count "$d/experiment_ledger.jsonl")" \
      "$(stat -c %Y "$d/run_report.json" 2>/dev/null || echo 0)"
  done
}

current_log() {
  local newest="" l
  for task in "${TASKS[@]}"; do
    l="$(task_log "$task")"
    if [ -f "$l" ]; then
      if [ -z "$newest" ] || [ "$l" -nt "$newest" ]; then
        newest="$l"
      fi
    fi
  done
  echo "$newest"
}

annotate() {
  awk '{
    if ($0 ~ /COMPILED|COMPILE_PATCH|SYNTHESIS registered/) print "==> " $0;
    else if ($0 ~ /grant plan:|grant: grant_|HERA wrote new branch/) print "==> " $0;
    else if ($0 ~ /NEW BEST|RESEARCH CYCLE COMPLETE/) print ">>  " $0;
    else if ($0 ~ /STAGE TRANSITION/) print "==> " $0;
    else if ($0 ~ /receipt: verdict/) print "   " $0;
    else if ($0 ~ /\[llm\]/) print "!!  " $0;
    else if ($0 ~ /FATAL|Traceback/) print "ERR " $0;
    else print $0;
  }'
}

rescan_tasks
if [ "$ONCE" = 1 ]; then
  if [ "${#TASKS[@]}" -eq 0 ]; then
    echo "no state dirs under $RUN_ROOT (STAMP=${STAMP:-auto})"
    exit 1
  fi
  snapshot
  exit 0
fi

if [ "${#TASKS[@]}" -eq 0 ]; then
  echo "waiting for state dirs under $RUN_ROOT (STAMP=${STAMP:-auto}) ..."
  tries=0
  while [ "${#TASKS[@]}" -eq 0 ] && [ "$tries" -lt 600 ]; do
    rescan_tasks
    tries=$((tries + 1))
    if [ -n "$OUTER_LOG" ] && [ -f "$OUTER_LOG" ]; then
      tail -n 4 "$OUTER_LOG" 2>/dev/null | sed 's/^/outer> /'
    else
      echo "waiting for outer log ..."
    fi
    sleep 3
  done
  [ "${#TASKS[@]}" -eq 0 ] && { echo "still no state dirs after ~30min; check the outer log"; exit 1; }
  snapshot
fi

LAST_SIG=""
LAST_LOG=""
while :; do
  CL="$(current_log)"
  SIG="$(signature)"
  if [ "$SIG" != "$LAST_SIG" ]; then
    snapshot
    LAST_SIG="$SIG"
  fi
  if [ -n "$CL" ] && [ "$CL" != "$LAST_LOG" ]; then
    echo ""
    echo "----- FOLLOWING: ${CL#$INCOMING/} -----"
  fi
  LAST_LOG="$CL"
  if [ -n "$CL" ]; then
    timeout 8 tail -n 0 -F "$CL" 2>/dev/null | annotate
  else
    echo "waiting for task logs ..."
    sleep 3
  fi
  if [ -n "$OUTER_LOG" ] && [ -f "$OUTER_LOG" ] && \
      grep -qE "===== \[7/7\] DONE =====" "$OUTER_LOG"; then
    echo ""
    echo "===== OUTER RUN FINISHED ====="
    snapshot
    grep -E "FAILED=|GPU_ID=|GPU_MAP=" "$OUTER_LOG" 2>/dev/null | tail -3
    break
  fi
done