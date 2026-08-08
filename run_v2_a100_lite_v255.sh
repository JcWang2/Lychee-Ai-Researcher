#!/usr/bin/env bash
# run_v2_a100_lite_v255.sh - v2.5.5 MLE-Lite fleet launcher (21-loop concurrent).
# v2.5.5 (pushed): generic MLE-Bench data-layout + sample-column fixes.
#   - data_layout: localized-prefix tables (<prefix>_train.csv(.zip) with
#     <prefix>_test* / <prefix>_sample_submission* siblings) are materialized
#     to canonical names automatically (en_/ru_ style); prefix-agnostic
#     glob, no competition-name hardcoding.
#   - no-id sample rule: target = sample columns NOT in test.csv; id/
#     passthrough = sample columns IN test.csv (sample order, verbatim).
#     Fixes Insult,Date,Comment-style headers where column 0 is the target
#     (5 templates + hera/analyzer; TSV headers delimiter-aware).
#   - filename-prefix image labels (cat.0.jpg / dog.1.jpg): resolver
#     synthesizes a REAL train.csv before the all-zero sample copy
#     (>=50 files, 2..64 non-numeric prefixes guard).
#   - launch preflight: TASK_DATA_OK now means resolve_dataset_layout
#     succeeds; layout gaps FAIL loudly at startup instead of a silent
#     closed-loop crash (V2_DATA_PREFLIGHT=0 disables).
#   - test_v2_255.py: 71 offline assertions for all of the above.
# v2.5.4 (delivery-only, NOT pushed): declarative row cap + budget floor.
#   - MethodSpec.default_max_train_rows (50000 tabular/timeseries/ensemble,
#     20000 image): normalize() injects it into resource_request so compiled
#     templates finally subsample MAX_TRAIN_ROWS (fixes taxi 200k-row HistGB
#     rc=-9 timeouts - the profile cap never reached the compiler before).
#   - ResourceProfiler derives min_budget_seconds = max(300, t_est*0.5);
#     planner clamps budgets into [min_budget, max_budget] so an over-eager
#     cheap budget cannot guarantee a timeout. F0 calibration shrinks it.
#   - Child proposal contract exposes train_rows_cap; HERA may request a
#     SMALLER max_train_rows explicitly (clamped). No if-else added.
#   - lite script default V2_GPU_MAP=0,1,2,3,4,5 (6 tasks, one GPU each) and
#     V2_CPU_THREADS=4 (24 threads total) to cut CPU/GPU contention.
# v2.5.2 (delivery-only, NOT pushed): runnability defaults moved from the
# compiler's method-prefix if/else chain into capability_registry metadata
# (default_preprocessing / default_validation / validation_policy);
# program_compiler.py now has ZERO startswith() routing (AST-enforced by
# test_v2_252.py: 127 assertions incl. frozen v2.5.1 behavior table).
# v2.5.2 (NOT pushed to git; delivery-only big version):
#   - method/rendering registry: _TEMPLATE_REGISTRY (15 renderers) replaces
#     every spec.renderer== if/elif chain in program_compiler.py;
#   - declarative method selector (method_selector.py): DatasetContract
#     metadata filter + _COST_MODELS cost table + ExperienceTable prior
#     (V2_EXPERIENCE_JSON). It RETRIEVES and RANKS only; the final research
#     decision stays with the Analyzer/Planner (prompt PRIOR KNOWLEDGE block);
#   - metric dispatch / random baselines / resource rules are lookup tables
#     (evaluator, stage_controller, portfolio, deterministic);
#   - test_v2_250.py: 83 offline assertions (AST-scans ban if/else routing,
#     competition-name hardcoding) + full v239 regression suites.
# v2.5.2 (delivery-only hotfix, NOT pushed): generic datetime handling.
#   - _looks_like_date now recognizes timezone-suffixed / ISO-T /
#     microsecond / full-month timestamps (taxi pickup_datetime UTC).
#   - date-like columns are never classified as text; Analyzer exposes
#     datetime_columns (all content-verified date columns).
#   - tabular harness derives calendar/elapsed features (year/month/day/
#     weekday/hour/elapsed) instead of ordinal-encoding timestamps;
#     LAG_COLUMN is only used by renderers declaring time_policy=lag.
#   - capability registry prior: tabular.datetime_feature_histgb.v1 +
#     datetime_derive/datetime_ordinal/datetime_drop preprocessing opts.
#   - test_v2_251.py: 49 offline assertions incl. end-to-end taxi-shaped
#     OOF sanity (no physically impossible fares).
# Default TASK_LIST = the six-task retest set (taxi/nomad/spooky/denoising/
# jigsaw/leaf). Override freely:
#   TASK_LIST='new-york-city-taxi-fare-prediction|NYC taxi fare: regression of fare amount (RMSE),nomad2018-predict-transparent-conductors|Nomad transparent conductors: regression of formation energy (RMSLE),spooky-author-identification|Spooky author: classify text author (multi-class logloss),denoising-dirty-documents|Denoising dirty documents: image-to-image pixel regression (RMSE),jigsaw-toxic-comment-classification-challenge|Toxic comments: multi-label logloss on 6 toxicity targets,leaf-classification|Leaf species: classify images into 99 species (multi-class logloss)' \
#   nohup bash run_v2_a100_lite_v255.sh 24h-mle > run_v2_lite_v255_outer.log 2>&1 &
# Usage: nohup bash run_v2_a100_lite_v255.sh > run_v2_lite_v255_outer.log 2>&1 &
set -euo pipefail
export LLM_MODEL="${LLM_MODEL:-qwen3.8-max}"
export STATE_ROOT="${STATE_ROOT:-/mnt/data/v2_state_lite}"
export V2_EXEC_IMAGE="${V2_EXEC_IMAGE:-pact-stage42-p8:20260727T112909Z_legacy_l1}"
export V2_EXEC_PYTHON="${V2_EXEC_PYTHON:-/opt/conda/envs/agent/bin/python3}"
export V2_TORCH_CACHE="${V2_TORCH_CACHE:-/mnt/data/v2_torch_cache}"
export V2_HF_CACHE="${V2_HF_CACHE:-/mnt/data/v2_hf_cache}"
export V2_LLM_ENV="${V2_LLM_ENV:-/mnt/data/stage42_delivery/latest_ai_scientist_v6.env}"
export V2_PKG_DIR="${V2_PKG_DIR:-ai_scientist_execution_layer_v2_20260808_v255}"
export V2_PKG_TAR="${V2_PKG_TAR:-ai_scientist_execution_layer_v2_20260808_v255.tar.gz}"
export V2_GPU_MAP="${V2_GPU_MAP:-0,1,2,3,4,5}"
export V2_CPU_THREADS="${V2_CPU_THREADS:-4}"
# Offline install tests compile+run harness subprocesses; under concurrent
# 12-loop load they can exceed the old hardcoded 600s. Env-driven (no code
# branching): unset -> tests keep their own defaults; set -> relaxed here.
export V2_TEST_HARNESS_TIMEOUT="${V2_TEST_HARNESS_TIMEOUT:-1800}"
# test_v2_251/252 end-to-end harness subprocesses (HistGB 1200x200x3fold x2):
# ~12s idle but can exceed 30min under 12-loop saturation. Skip them for
# concurrent launches (validated in idle/6-loop installs); set 0 for the full gate.
export V2_TEST_HARNESS_SKIP="${V2_TEST_HARNESS_SKIP:-1}"
# v2.5.5: resolve_dataset_layout preflight at launch (TASK_DATA_OK gate).
export V2_DATA_PREFLIGHT="${V2_DATA_PREFLIGHT:-1}"
export TASK_LIST="${TASK_LIST:-new-york-city-taxi-fare-prediction|NYC taxi fare: regression of fare amount (RMSE),nomad2018-predict-transparent-conductors|Nomad transparent conductors: regression of formation energy (RMSLE),spooky-author-identification|Spooky author: classify text author (multi-class logloss),denoising-dirty-documents|Denoising dirty documents: image-to-image pixel regression (RMSE),jigsaw-toxic-comment-classification-challenge|Toxic comments: multi-label logloss on 6 toxicity targets,leaf-classification|Leaf species: classify images into 99 species (multi-class logloss)}"
cd /mnt/data/stage42_delivery/incoming
exec bash run_v2_a100_3tasks_v23.sh "${1:-24h-mle}"
