#!/usr/bin/env bash
# run_v2_a100_lite_v250.sh - v2.5.0 declarative-method-architecture retest on GPU 5.
# v2.5.0 (NOT pushed to git; delivery-only big version):
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
# Default TASK_LIST = the three image tasks for the v2.5 retest. Override
# freely, e.g. the 6-task set:
#   TASK_LIST='new-york-city-taxi-fare-prediction|NYC taxi fare: regression of fare amount (RMSE),nomad2018-predict-transparent-conductors|Nomad transparent conductors: regression of formation energy (RMSLE),spooky-author-identification|Spooky author: classify text author (multi-class logloss),denoising-dirty-documents|Denoising dirty documents: image-to-image pixel regression (RMSE),jigsaw-toxic-comment-classification-challenge|Toxic comments: multi-label logloss on 6 toxicity targets,leaf-classification|Leaf species: classify images into 99 species (multi-class logloss)' \
#   nohup bash run_v2_a100_lite_v250.sh 24h-mle > run_v2_lite_v250_outer.log 2>&1 &
# Requires: v2.5.0 tar.gz + run_v2_a100_3tasks_v23.sh already in incoming.
# Usage: nohup bash run_v2_a100_lite_v250.sh > run_v2_lite_v250_outer.log 2>&1 &
set -euo pipefail
export LLM_MODEL="${LLM_MODEL:-qwen3.8-max}"
export STATE_ROOT="${STATE_ROOT:-/mnt/data/v2_state_lite}"
export V2_EXEC_IMAGE="${V2_EXEC_IMAGE:-pact-stage42-p8:20260727T112909Z_legacy_l1}"
export V2_EXEC_PYTHON="${V2_EXEC_PYTHON:-/opt/conda/envs/agent/bin/python3}"
export V2_TORCH_CACHE="${V2_TORCH_CACHE:-/mnt/data/v2_torch_cache}"
export V2_HF_CACHE="${V2_HF_CACHE:-/mnt/data/v2_hf_cache}"
export V2_LLM_ENV="${V2_LLM_ENV:-/mnt/data/stage42_delivery/latest_ai_scientist_v6.env}"
export V2_PKG_DIR="${V2_PKG_DIR:-ai_scientist_execution_layer_v2_20260807_v250}"
export V2_PKG_TAR="${V2_PKG_TAR:-ai_scientist_execution_layer_v2_20260807_v250.tar.gz}"
export V2_GPU_MAP="${V2_GPU_MAP:-5,5,5}"
export TASK_LIST="${TASK_LIST:-aerial-cactus-identification|Cactus aerial photos: classify 32x32 RGB images containing cacti (AUC),aptos2019-blindness-detection|Diabetic retinopathy: grade retina images into 5 severity classes (QWK),dog-breed-identification|Dog breeds: classify 120 breeds from photos (multi-class logloss)}"
cd /mnt/data/stage42_delivery/incoming
exec bash run_v2_a100_3tasks_v23.sh "${1:-24h-mle}"
