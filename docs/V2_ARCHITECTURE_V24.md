# V2.4 Architecture ? from "runs through" to "wins"

Status: DRAFT (2026-08-07)
Base: v2.3.7 (git main @35a574d) + v2.3.8 in-flight (RLE/bbox/audio recognition,
      parallel worktree ai_scientist_execution_layer_v2_20260807_v238).
Branch policy: v2.4.0 must merge AFTER v2.3.8 lands; both are additive.

## 1. Problem statement

Live evidence across 9 tasks (aerial/aptos/dog/taxi/nomad/spooky/jigsaw/leaf/
denoising): every task now RUNS (valid submission, correct metric, rc=0) but
the score ceiling is far below what the data supports:

  - denoising-dirty-documents: stuck at 0.2494 RMSE = per-image mean baseline
    (template has no neighborhood operator; HERA cannot express median/NLM).
  - jigsaw / leaf / taxi / nomad: cheap probes land on sklearn-level models;
    no native GBDT, no transformers, no target encoding, no TTA, no stacking.
  - HERA prompts carry a SHALLOW profile (rows/classes/cols/missing list) and
    NO measured per-branch scores, NO difficulty evidence, NO cross-task
    experience -> the LLM guesses methods instead of reading evidence.

Root causes (three layers):
  L1 Analysis: profile is structural only; no target/feature diagnostics, no
     difficulty ladder, no leakage/order signals, no measured branch ranking.
  L2 Method space: 12 templates, mostly sklearn-level; per-family capacity
     ceiling far below SOTA (no native GBDT / transformer / median-NLM / TTA).
  L3 Iteration: no memory across tasks, no per-branch metric history in the
     prompt, no failure registry, stage machine cannot detect "method space
     exhausted" - it keeps re-rolling the same weak branch.

## 2. Pillars

### P1 DeepProfile (analysis capability, stdlib-only on host)
Extend AnalysisProfile with MEASURED diagnostics (bounded: sampled rows):

  target_diag:     n_classes, top1_share (imbalance), entropy, skew,
                   multi_target, unique_ratio
  feature_diag:    per-col missing_rate, cardinality, constant_cols,
                   duplicate_cols, numeric_share, high_card_cols (top 10)
  order_diag:      id_monotonic (row-id vs target correlation on sample),
                   time_range (min/max of time col), time_col_present
  difficulty_ladder: measured by a container probe (P2 infra, runs once at
                   startup after F0): constant/majority, linear, gbdt-mini
                   -> {constant, linear, gbdt} scores + headroom = gbdt -
                   constant. HERA prompt receives the ladder; stage
                   controller uses headroom to decide S2 vs S3.

Implementation: new module deep_profile.py (stdlib csv/math only, <= 30s,
fail-open). Analyzer calls it after layout resolution; results land in
AnalysisProfile.deep_diagnostics + difficulty_ladder (filled by probe).

### P2 MethodSpace v2.4 (templates + probe infra)
  - difficulty_probe capability (container, reuses F0 exec channel):
    constant/majority + LogisticRegression/Ridge + HistGB-mini on <=8k rows,
    <=60s, prints DIFFICULTY_PROBE <json>; stores state/difficulty_ladder.json.
  - tabular: tabular.gbdt.native.v1 (LightGBM/XGBoost when present in the
    exec image, else HistGB with categorical support + early stopping),
    tabular.encoding.target.v1 (target/ordinal encoding -> GBDT),
    tabular.features.interaction.v1 (top-k pairwise numeric products by
    cheap univariate screen).
  - text: text.transformer.finetune.v1 (distilbert when transformers is in
    the exec image, else falls back to text.tfidf.svd.linear.v1),
    text.tfidf.svd.linear.v1 (SVD 128 + linear; strong on short text).
  - image: extend timm templates: image_size options, class-balanced
    sampling, label smoothing, TTA flag, multi-label sigmoid head.
  - image_pixel: feature_mode stats|median|nlm|patch_cnn + kernel_size;
    median/NLM reuses the stdlib PNG decoder (v2.3.7); expected denoising
    gain 0.25 -> <=0.12.
  - timeseries: rolling-window stats (mean/std/min/max over 3 windows) +
    lags + calendar features when a date column exists.
  - ensemble: stack_oof (ridge stack over top-3 OOF predictions) replaces
    soft-vote as the default final ensemble.
  - every template declares cost_tier cheap|medium|expensive so HERA can
    match stage and remaining budget.

### P3 LearningLoop (iteration memory)
  - portfolio branches carry measured best_metric / trials / last_rc; the
    prioritizer prompt prints per-branch "best=0.2494 (4 trials)" so the LLM
    stops re-rolling losing branches (denoising case).
  - experience_store.json (per task) + shared cross-task memory file
    (versioned, committed to state root): entries
    (profile_fingerprint, method, params, metric, wall, rc). HERA prompt
    retrieves top-3 similar tasks (fingerprint k-NN) with their best methods.
  - failure registry: (method, layout_signature) -> fail count; broken
    ephemeral capabilities are demoted; new capabilities get a feasibility
    trial before a full grant.
  - stage machine v2.4: S1 = ladder + cheapest winner; S2 = local
    exploitation ranked by measured branch metrics; S3 = complexity ONLY
    when headroom evidence supports it; S4 = top-2 with full budget.
    New "space_exhausted" detector: all compatible branches tried >=2x with
    no gain -> force new-capability synthesis (P4) or ladder re-run.

### P4 Free-code channel (experimental, v2.5)
  V2_FREE_CODE=1: after templates exhausted AND headroom large, LLM writes a
  full solution; gates: py_compile + sandbox run on a 5% holdout + submission
  format validator. High risk; kept behind a flag, not in the default path.

## 3. MLE-Bench coverage matrix (81 comps in metrics_registry)

  Covered today (v2.3.7): 55  (tabular 14 / text 10 / image 30 / pixel 1)
  v2.3.8 (recognition, in-flight): adds image_mask (5), image_detection (6),
  audio (3) recognition + baseline renderers.
  v2.4 (method space): raises the ceiling for all of the above.
  Still open after v2.3.8+v2.4: 7 sequence/span/QA comps (tf2-qa, chaii,
  tweet-sentiment, text-normalization x2, stanford-covid, bms-molecular) +
  3 special-format comps (h&m, facebook-keyword, billion-word) -> v2.5
  (text.sequence template + free-code channel).

## 4. Milestones

  M1 (v2.4.0): deep_profile.py + contract fields + analyzer integration +
      prompt injection + offline tests.   <- this PR
  M2 (v2.4.1): difficulty probe capability + ladder into prompt/stage.
  M3 (v2.4.2): branch metric tracking + experience store + retrieval.
  M4 (v2.4.3): method space: native GBDT, target encoding, SVD, median/NLM,
      TTA/balance, rolling features, OOF stack.
  M5 (v2.4.4): integrate with v2.3.8 mask/detection/audio templates;
      smoke on 3 tasks; 24h validation on 6 tasks (3 old + 3 new).
  Each milestone: >=13 offline suites green -> smoke -> 24h on GPU5.

## 5. Non-goals (this release)

  - No free-form LLM code in the default path (P4, v2.5).
  - No per-competition hardcoding anywhere (same rule as v2.3.x).
  - No changes to the file-bus / daemon / receipt contract (stable ABI).
