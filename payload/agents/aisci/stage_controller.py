# -*- coding: utf-8 -*-
"""stage_controller.py - V2.2 four-stage soft research guidance.

Platform-side, competition-agnostic stage machine. It NEVER chooses a method
or an experiment: it only
  - derives a generic reference line (random baseline -> metric_norm),
  - tracks which of the four research periods the task is in,
  - soft-guides HERA via prompt blocks + intent whitelists,
  - clips the plan when the wall clock is nearly gone (S4 sprint).

Stages (per task, one state dir):
  S1_baseline      low-cost low-variance methods, valid submission + norm
  S2_enhancement   local attributable improvements around the incumbent
  S3_complex       new representations / inductive biases / structure
  S4_sprint        concentrated final training on the top candidate(s)

Incumbent assets flow through every stage - switching never resets the
research to baseline.
"""
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from hera.portfolio import estimate_grant_cost

STAGE_ORDER = ("S1_baseline", "S2_enhancement", "S3_complex", "S4_sprint")

STAGE_DEFAULT_INTENT = {
    "S1_baseline": "cheap_probe",
    "S2_enhancement": "local_exploitation",
    "S3_complex": "expensive_structural",
    "S4_sprint": "final_training",
}

STAGE_INTENTS = {
    "S1_baseline": {"feasibility", "repair", "cheap_probe"},
    "S2_enhancement": {"cheap_probe", "local_exploitation", "confirmation"},
    "S3_complex": {"expensive_structural", "local_exploitation", "cheap_probe"},
    "S4_sprint": {"final_training", "confirmation", "local_exploitation"},
}

STAGE_GOALS = {
    "S1_baseline":
        "Low-cost, low-variance, quickly verifiable executable methods; "
        "get ONE valid submission; norm >= 0.15 vs the reference line "
        "means baseline level.",
    "S2_enhancement":
        "Local, attributable improvements around the incumbent "
        "(params/features/preprocessing/CV). Stop when tuning turns "
        "negative (stagnation/regressions).",
    "S3_complex":
        "New representations / inductive biases / structural capability, "
        "proposed by HERA from the verified capability registry (cached "
        "weights only); carry the incumbent forward.",
    "S4_sprint":
        "Concentrate resources on high-value candidates: final training, "
        "confirmation, last safe improvements until the wall clock ends.",
}

# Generic reference lines, derived from the metric family + profile signals.
# Competition names never appear here.

# v2.5.0 declarative random-reference table: metric family -> baseline kind.
# Kinds are abstract computation types ("zero"/"half"/"one_over_k"/"log_k"/
# "target_scale"); metric names live ONLY in this data table.
_RANDOM_BASELINE_KIND = {
    "auc": "half",
    "qwk": "zero", "mcc": "zero", "spearman": "zero", "pearson": "zero",
    "kendall_tau": "zero", "map_at_k": "zero", "label_ranking_ap": "zero",
    "accuracy": "one_over_k", "f1_macro": "one_over_k",
    "f1_micro": "one_over_k", "f1_binary": "one_over_k", "f0_5": "one_over_k",
    "logloss": "log_k", "binary_logloss": "log_k",
    "weighted_logloss": "log_k", "kl_div": "log_k",
    "rmse": "target_scale", "mae": "target_scale", "log_mae": "target_scale",
    "rmsle": "target_scale", "mean_angular_error": "target_scale",
    "haversine": "target_scale",
}


def random_baseline(profile) -> Optional[float]:
    """Generic random-prediction reference line for the task's metric.

    Returns None when the metric family has no canonical random value
    (the caller then treats norm as unknown).
    """
    metric = str(getattr(profile, "metric_name", "") or "")
    kind = _RANDOM_BASELINE_KIND.get(metric)
    if kind is None:
        return None
    n_classes = max(2, int(getattr(profile, "n_classes", 0) or 0) or 2)
    if kind == "zero":
        return 0.0
    if kind == "half":
        return 0.5
    if kind == "one_over_k":
        return 1.0 / float(n_classes)
    if kind == "log_k":
        return float(math.log(n_classes))
    if kind == "target_scale":
        # Without a measured target scale we use 1.0 as the deterministic
        # reference; relative improvements are still comparable.
        stats = getattr(profile, "target_stats", None) or {}
        try:
            scale = float(stats.get("std") or stats.get("mean") or 1.0)
        except (TypeError, ValueError):
            scale = 1.0
        return max(1e-6, scale)
    return None


def metric_upper_bound(profile) -> Optional[float]:
    """Ideal metric value used to normalize progress into [0, 1]."""
    direction = str(getattr(profile, "metric_direction",
                            "higher_is_better") or "")
    if direction == "lower_is_better":
        return 0.0
    return 1.0


def metric_norm(best: Optional[float], profile) -> Optional[float]:
    """Normalized progress: (best - random) / (upper - random), flipped for
    minimize metrics. Clamped to [0, 1]; None when no reference exists."""
    if best is None:
        return None
    random = random_baseline(profile)
    upper = metric_upper_bound(profile)
    if random is None or upper is None:
        return None
    direction = str(getattr(profile, "metric_direction",
                            "higher_is_better") or "")
    try:
        best_f = float(best)
    except (TypeError, ValueError):
        return None
    if direction == "lower_is_better":
        denom = (random - upper)
        if abs(denom) < 1e-9:
            return None
        norm = (random - best_f) / denom
    else:
        denom = (upper - random)
        if abs(denom) < 1e-9:
            return None
        norm = (best_f - random) / denom
    return max(0.0, min(1.0, norm))


class StageController:
    """Four-stage soft guidance machine (per task, persisted)."""

    def __init__(self, state_dir, profile, resource: Optional[dict] = None,
                 max_grants: int = 128, total_wall_clock: int = 86400,
                 s1_hold_grants: int = 2,
                 s2_stagnation_exit: int = 5,
                 s2_regression_exit: int = 3,
                 s3_stagnation_exit: int = 5,
                 s4_stagnation_exit: int = 8,
                 s1_max_grants: int = 8,
                 stage_override: str = ""):
        self.state_dir = Path(state_dir)
        self.profile = profile
        self.resource = dict(resource or {})
        self.max_grants = max(1, int(max_grants))
        self.total_wall_clock = max(1, int(total_wall_clock))
        self.s1_hold_grants = max(1, int(s1_hold_grants))
        self.s2_stagnation_exit = max(2, int(s2_stagnation_exit))
        self.s2_regression_exit = max(1, int(s2_regression_exit))
        self.s3_stagnation_exit = max(2, int(s3_stagnation_exit))
        self.s4_stagnation_exit = max(2, int(s4_stagnation_exit))
        self.s1_max_grants = max(1, int(
            os.environ.get("V2_S1_MAX_GRANTS") or s1_max_grants))
        self.grants_seen = 0
        self._s1_hold = 0
        self._s2_stagnation = 0
        self._s2_regressions = 0
        self._s3_grants = 0
        self._s4_no_best = 0
        self._entry_best = None
        self._entry_norm = None
        self._last_reason = ""
        self.history: List[Dict[str, Any]] = self._load_history()
        self._restore_state()
        self.stage = self._initial_stage(stage_override)

    # ---- initialization ----
    def _initial_stage(self, override: str) -> str:
        env_stage = str(os.environ.get("STAGE_PROFILE", "") or "").strip()
        candidate = override or env_stage
        if candidate in STAGE_ORDER:
            return candidate
        if str(self._restored_stage or "") in STAGE_ORDER:
            return self._restored_stage
        return "S1_baseline"

    def _load_history(self) -> List[Dict[str, Any]]:
        path = self.state_dir / "stage_history.json"
        try:
            if path.is_file():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return [d for d in data if isinstance(d, dict)]
        except (OSError, ValueError):
            pass
        return []

    def _load_state(self) -> dict:
        """stage_state.json is a CHECKPOINT (not just audit history):
        current_stage + all counters + incumbent entry metric, restored on
        restart so the run never falls back from S2/S3 to S1."""
        path = self.state_dir / "stage_state.json"
        try:
            if path.is_file():
                data = json.loads(path.read_text(encoding="utf-8"))
                return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            pass
        return {}

    def _restore_state(self) -> None:
        state = self._load_state()
        self._restored_stage = str(state.get("current_stage") or "")
        self.grants_seen = max(0, int(state.get("grants_seen") or 0))
        self._s1_hold = max(0, int(state.get("s1_hold") or 0))
        self._s2_stagnation = max(0, int(state.get("s2_stagnation") or 0))
        self._s2_regressions = max(0, int(state.get("s2_regressions") or 0))
        self._s3_grants = max(0, int(state.get("s3_grants") or 0))
        self._s4_no_best = max(0, int(state.get("s4_no_best") or 0))
        try:
            if state.get("entry_best") is not None:
                self._entry_best = float(state["entry_best"])
        except (TypeError, ValueError):
            pass
        try:
            if state.get("entry_norm") is not None:
                self._entry_norm = float(state["entry_norm"])
        except (TypeError, ValueError):
            pass
        self._last_reason = str(state.get("last_reason") or "")

    def _state_dict(self) -> dict:
        return {
            "current_stage": self.stage,
            "grants_seen": self.grants_seen,
            "s1_hold": self._s1_hold,
            "s2_stagnation": self._s2_stagnation,
            "s2_regressions": self._s2_regressions,
            "s3_grants": self._s3_grants,
            "s4_no_best": self._s4_no_best,
            "entry_best": self._entry_best,
            "entry_norm": self._entry_norm,
            "last_reason": self._last_reason,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def _persist(self) -> None:
        """Atomic checkpoint after every stage-changing event."""
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.state_dir / "stage_history.json.tmp"
            tmp.write_text(
                json.dumps(self.history, indent=2, ensure_ascii=False),
                encoding="utf-8")
            os.replace(str(tmp), str(self.state_dir / "stage_history.json"))
            tmp2 = self.state_dir / "stage_state.json.tmp"
            tmp2.write_text(
                json.dumps(self._state_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8")
            os.replace(str(tmp2), str(self.state_dir / "stage_state.json"))
        except OSError:
            pass

    # ---- stage semantics ----
    def default_intent(self) -> str:
        return STAGE_DEFAULT_INTENT.get(self.stage, "cheap_probe")

    def allowed_intents(self) -> List[str]:
        return sorted(STAGE_INTENTS.get(self.stage, STAGE_INTENTS["S1_baseline"]))

    def intent_hints(self) -> dict:
        from hera.portfolio import INTENT_CHILD_RANGES
        allowed = self.allowed_intents()
        return {
            "allowed": allowed,
            "default": self.default_intent(),
            "child_ranges": {k: INTENT_CHILD_RANGES[k] for k in allowed},
        }

    def guidance_summary(self) -> str:
        return "%s: %s" % (self.stage, STAGE_GOALS.get(self.stage, ""))

    def prompt_block(self) -> str:
        return (
            "CURRENT RESEARCH STAGE: %s\n"
            "Stage goals: %s\n"
            "Allowed research intents for this stage: %s\n"
            "Preferred intent: %s\n"
            "Stage rule: you stay inside this stage's intent whitelist; "
            "the platform only maps intent -> child-trial count.\n"
            % (self.stage, STAGE_GOALS.get(self.stage, ""),
               ", ".join(self.allowed_intents()), self.default_intent())
        )

    def pre_grant_clip(self, remaining_wall_clock: float) -> str:
        """Wall-clock clipping BEFORE the planner runs (v2.2.1): if a full
        local-exploitation grant can no longer fit inside the remaining wall
        clock, force the S4 sprint NOW so the next grant is planned cheap
        (final training) instead of being refused by the budget guard with
        no chance to switch stage first."""
        remaining = float(remaining_wall_clock or 0)
        if self.stage not in ("S2_enhancement", "S3_complex"):
            return self.stage
        est = estimate_grant_cost(self.resource, 2, "local_exploitation")
        if remaining > 0 and remaining < est * 1.5:
            self._switch(
                "S4_sprint",
                "pre-grant wall-clock clipping: %.0fs left < %.0fs for "
                "one full grant" % (remaining, est * 1.5))
        return self.stage

    # ---- transitions ----
    def _switch(self, new_stage: str, reason: str) -> None:
        if new_stage == self.stage:
            return
        self.history.append({
            "from": self.stage,
            "to": new_stage,
            "reason": reason,
            "at_grant": self.grants_seen,
            "best_metric": self._entry_best,
            "metric_norm": self._entry_norm,
            "resource": dict(self.resource or {}),
        })
        self._last_reason = reason
        self.stage = new_stage
        self._s1_hold = 0
        self._s2_stagnation = 0
        self._s2_regressions = 0
        self._s3_grants = 0
        self._s4_no_best = 0
        self._persist()

    def on_grant_result(self, result: dict) -> str:
        """Feed one terminal grant outcome; returns the current stage.

        result keys: grants_used, remaining_wall_clock, best_metric,
        metric_norm, new_best, stagnation_count, submission_exists,
        regressions, intent.
        """
        self.grants_seen += 1
        best = result.get("best_metric")
        norm = result.get("metric_norm")
        new_best = bool(result.get("new_best"))
        stagnation = int(result.get("stagnation_count") or 0)
        submission = bool(result.get("submission_exists"))
        regressions = int(result.get("regressions") or 0)
        remaining = float(result.get("remaining_wall_clock") or 0)

        if best is not None:
            if self._entry_best is None:
                self._entry_best = best
                self._entry_norm = norm
            elif new_best:
                self._entry_best = best
                self._entry_norm = norm

        if self.stage == "S1_baseline":
            if submission and norm is not None and norm >= 0.15:
                self._s1_hold += 1
            else:
                self._s1_hold = 0
            if self._s1_hold >= self.s1_hold_grants:
                self._switch(
                    "S2_enhancement",
                    "S1 baseline met: submission + norm>=0.15 held %d grants"
                    % self._s1_hold)
            elif self.grants_seen >= self.s1_max_grants:
                # v2.2.1 fallback: metrics without a canonical reference
                # (dice/iou/jaccard/levenshtein/multilabel-auc) or any task
                # stuck in S1 must still exit instead of burning the budget.
                self._switch(
                    "S2_enhancement",
                    "S1 grant cap reached (%d): submission=%s norm=%s"
                    % (self.s1_max_grants, submission, norm))

        elif self.stage == "S2_enhancement":
            self._s2_stagnation = (0 if new_best else self._s2_stagnation + 1)
            self._s2_regressions = (0 if new_best
                                    else self._s2_regressions + regressions)
            if self._s2_stagnation >= self.s2_stagnation_exit or \
                    self._s2_regressions >= self.s2_regression_exit:
                self._switch(
                    "S3_complex",
                    "S2 enhancement exhausted: stagnation=%d regressions=%d"
                    % (self._s2_stagnation, self._s2_regressions))

        elif self.stage == "S3_complex":
            self._s3_grants += 1
            if new_best:
                self._switch("S4_sprint",
                             "S3 complex method produced a NEW BEST")
            elif self._s3_grants >= self.s3_stagnation_exit:
                self._switch(
                    "S4_sprint",
                    "S3 complex stage stagnated after %d grants; sprint on "
                    "the best candidate" % self._s3_grants)

        elif self.stage == "S4_sprint":
            self._s4_no_best = (0 if new_best else self._s4_no_best + 1)
            # S4 stays until the wall clock / budgets end; no further stage.

        # Wall-clock clipping: if a full complex grant can no longer fit,
        # force the sprint (final training on top-1 only).
        if self.stage in ("S2_enhancement", "S3_complex"):
            est = estimate_grant_cost(self.resource, 2, "local_exploitation")
            if remaining > 0 and remaining < est * 1.5:
                self._switch("S4_sprint",
                             "wall-clock clipping: %.0fs left < %.0fs for "
                             "one full grant" % (remaining, est * 1.5))
        # v2.2.1: checkpoint after EVERY grant, not only on stage switches,
        # so a restart keeps counters/entry_best (stage_state.json).
        self._persist()
        return self.stage

    def snapshot(self) -> dict:
        return {
            "stage": self.stage,
            "grants_seen": self.grants_seen,
            "default_intent": self.default_intent(),
            "allowed_intents": self.allowed_intents(),
            "s1_hold": self._s1_hold,
            "s2_stagnation": self._s2_stagnation,
            "s2_regressions": self._s2_regressions,
            "s3_grants": self._s3_grants,
            "s4_no_best": self._s4_no_best,
            "entry_best": self._entry_best,
            "entry_norm": self._entry_norm,
            "history": self.history,
        }