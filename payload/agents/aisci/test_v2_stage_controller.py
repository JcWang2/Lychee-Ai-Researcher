# -*- coding: utf-8 -*-
"""test_v2_stage_controller.py - V2.2 four-stage guidance tests."""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from stage_controller import (StageController, metric_norm, random_baseline)  # noqa: E402
from v2_contracts import AnalysisProfile  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("[OK] " + name)
    else:
        print("[FAIL] " + name + (" | " + detail if detail else ""))
        FAILURES.append(name)


def _profile(metric="accuracy", direction="higher_is_better",
             n_classes=2, task_type="classification"):
    return AnalysisProfile(competition="x", task_type=task_type,
                           metric_name=metric,
                           metric_direction=direction,
                           metric_alignment="exact",
                           metric_label=metric, n_classes=n_classes,
                           train_rows=1000, modality="tabular")


def test_random_baseline_formulas():
    acc = _profile("accuracy", "higher_is_better", n_classes=3)
    check("accuracy random = 1/n", abs(random_baseline(acc) - 1.0 / 3.0) < 1e-9,
          str(random_baseline(acc)))
    auc = _profile("auc", "higher_is_better", n_classes=2)
    check("auc random = 0.5", abs(random_baseline(auc) - 0.5) < 1e-9,
          str(random_baseline(auc)))
    qwk = _profile("qwk", "higher_is_better", n_classes=5)
    check("qwk random = 0", random_baseline(qwk) == 0.0,
          str(random_baseline(qwk)))
    ll = _profile("logloss", "lower_is_better", n_classes=120)
    check("logloss random = ln(n)", abs(random_baseline(ll) - 4.7875) < 1e-3,
          str(random_baseline(ll)))


def test_metric_norm_directions():
    acc = _profile("accuracy", "higher_is_better", n_classes=2)
    check("accuracy norm", abs(metric_norm(0.85, acc) - 0.7) < 1e-9,
          str(metric_norm(0.85, acc)))
    ll = _profile("logloss", "lower_is_better", n_classes=120)
    norm = metric_norm(3.4, ll)
    check("logloss norm flipped", norm is not None and 0.2 < norm < 0.35,
          str(norm))
    check("norm None without best", metric_norm(None, acc) is None)
    check("norm clamped", 0.0 <= metric_norm(999, acc) <= 1.0,
          str(metric_norm(999, acc)))


def _grants(controller, seq, submission=True):
    for i, (best, new_best, stag, reg) in enumerate(seq):
        controller.on_grant_result({
            "grants_used": i + 1,
            "remaining_wall_clock": 80000.0,
            "best_metric": best,
            "metric_norm": metric_norm(best, controller.profile),
            "new_best": new_best,
            "stagnation_count": stag,
            "submission_exists": submission,
            "regressions": reg,
            "intent": "cheap_probe",
        })


def test_s1_to_s2():
    tmp = Path(tempfile.mkdtemp(prefix="v2_stage_test_"))
    try:
        profile = _profile("accuracy", "higher_is_better", n_classes=2)
        ctl = StageController(tmp, profile, {"t_est_seconds": 600},
                              s1_hold_grants=2)
        check("starts S1", ctl.stage == "S1_baseline", ctl.stage)
        # one good grant -> still S1 (hold=1)
        _grants(ctl, [(0.85, True, 0, 0)])
        check("S1 holds after 1", ctl.stage == "S1_baseline", ctl.stage)
        _grants(ctl, [(0.87, True, 0, 0)])
        check("S1 -> S2 after 2 held", ctl.stage == "S2_enhancement",
              ctl.stage)
        hist = json.loads((tmp / "stage_history.json").read_text(
            encoding="utf-8"))
        check("stage history persisted", len(hist) >= 1,
              str(hist)[:200])
        check("S2 default intent", ctl.default_intent() == "local_exploitation",
              ctl.default_intent())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_s2_to_s3_on_stagnation():
    tmp = Path(tempfile.mkdtemp(prefix="v2_stage_test_"))
    try:
        profile = _profile("accuracy", "higher_is_better", n_classes=2)
        ctl = StageController(tmp, profile, {"t_est_seconds": 600},
                              s1_hold_grants=1, s2_stagnation_exit=3)
        # skip S1 quickly
        _grants(ctl, [(0.85, True, 0, 0)])
        check("S2 reached", ctl.stage == "S2_enhancement", ctl.stage)
        _grants(ctl, [(0.85, False, 1, 0), (0.85, False, 2, 0),
                      (0.85, False, 3, 1)])
        check("S2 -> S3 on stagnation", ctl.stage == "S3_complex", ctl.stage)
        check("S3 default intent", ctl.default_intent() == "expensive_structural",
              ctl.default_intent())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_s3_to_s4_on_new_best():
    tmp = Path(tempfile.mkdtemp(prefix="v2_stage_test_"))
    try:
        profile = _profile("accuracy", "higher_is_better", n_classes=2)
        ctl = StageController(tmp, profile, {"t_est_seconds": 600},
                              s1_hold_grants=1, s2_stagnation_exit=2,
                              s3_stagnation_exit=5)
        _grants(ctl, [(0.85, True, 0, 0)])
        _grants(ctl, [(0.86, True, 0, 0), (0.86, False, 1, 0),
                      (0.86, False, 2, 1)])
        check("S3 reached", ctl.stage == "S3_complex", ctl.stage)
        _grants(ctl, [(0.90, True, 0, 0)])
        check("S3 -> S4 on NEW BEST", ctl.stage == "S4_sprint", ctl.stage)
        check("S4 default intent", ctl.default_intent() == "final_training",
              ctl.default_intent())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_wall_clock_clipping():
    tmp = Path(tempfile.mkdtemp(prefix="v2_stage_test_"))
    try:
        profile = _profile("accuracy", "higher_is_better", n_classes=2)
        ctl = StageController(tmp, profile,
                              {"t_est_seconds": 600, "max_budget_seconds": 600},
                              s1_hold_grants=1)
        # reach S2, then report very little wall clock left
        _grants(ctl, [(0.85, True, 0, 0)])
        ctl.on_grant_result({
            "grants_used": 2,
            "remaining_wall_clock": 60.0,
            "best_metric": 0.86,
            "metric_norm": metric_norm(0.86, profile),
            "new_best": False,
            "stagnation_count": 1,
            "submission_exists": True,
            "regressions": 0,
            "intent": "local_exploitation",
        })
        check("wall clock clipping -> S4", ctl.stage == "S4_sprint",
              ctl.stage)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_stage_profile_override():
    tmp = Path(tempfile.mkdtemp(prefix="v2_stage_test_"))
    old = os.environ.get("STAGE_PROFILE")
    os.environ["STAGE_PROFILE"] = "S3_complex"
    try:
        profile = _profile("accuracy", "higher_is_better", n_classes=2)
        ctl = StageController(tmp, profile, {"t_est_seconds": 600})
        check("STAGE_PROFILE override", ctl.stage == "S3_complex", ctl.stage)
        hints = ctl.intent_hints()
        check("intent hints contain whitelist",
              "expensive_structural" in hints["allowed"]
              and hints["default"] == "expensive_structural",
              str(hints))
        check("prompt block mentions stage", "S3_complex" in ctl.prompt_block(),
              ctl.prompt_block()[:120])
    finally:
        if old is None:
            os.environ.pop("STAGE_PROFILE", None)
        else:
            os.environ["STAGE_PROFILE"] = old
        shutil.rmtree(tmp, ignore_errors=True)


def test_stage_state_restore():
    """v2.2.1: stage_state.json is a CHECKPOINT - restart must keep the
    stage and all counters (probe: S2 + stagnation before restart stays
    S2 + stagnation after restart, never falls back to S1)."""
    tmp = Path(tempfile.mkdtemp(prefix="v2_stage_restore_"))
    try:
        profile = _profile("accuracy", "higher_is_better", n_classes=2)
        ctl = StageController(tmp, profile, {"t_est_seconds": 600},
                              s1_hold_grants=1, s2_stagnation_exit=5)
        _grants(ctl, [(0.85, True, 0, 0)])           # S1 -> S2
        _grants(ctl, [(0.86, True, 0, 0),            # new best kept
                      (0.87, False, 1, 0),           # stagnation 1
                      (0.87, False, 2, 0)])          # stagnation 2
        check("reached S2 with counters",
              ctl.stage == "S2_enhancement" and ctl._s2_stagnation == 2,
              "%s stag=%d" % (ctl.stage, ctl._s2_stagnation))
        state_file = tmp / "stage_state.json"
        check("stage_state.json written", state_file.is_file())
        data = json.loads(state_file.read_text(encoding="utf-8"))
        check("state has full counters",
              data.get("current_stage") == "S2_enhancement"
              and data.get("s2_stagnation") == 2
              and data.get("grants_seen") == 4
              and data.get("entry_best") == 0.86,
              str(data))
        # simulate restart: NEW controller on the same state dir
        ctl2 = StageController(tmp, profile, {"t_est_seconds": 600},
                               s1_hold_grants=1, s2_stagnation_exit=5)
        check("restart keeps S2", ctl2.stage == "S2_enhancement", ctl2.stage)
        check("restart keeps stagnation counter",
              ctl2._s2_stagnation == 2, str(ctl2._s2_stagnation))
        check("restart keeps grants_seen", ctl2.grants_seen == 4,
              str(ctl2.grants_seen))
        check("restart keeps entry_best", ctl2._entry_best == 0.86,
              str(ctl2._entry_best))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_s1_max_grant_fallback_none_norm():
    """v2.2.1: metrics without a reference line (dice/iou/jaccard/...)
    must not stay in S1 forever; the S1 grant cap forces S2."""
    tmp = Path(tempfile.mkdtemp(prefix="v2_stage_s1cap_"))
    try:
        profile = _profile("dice", "higher_is_better", n_classes=2)
        check("dice has no random baseline", random_baseline(profile) is None,
              str(random_baseline(profile)))
        ctl = StageController(tmp, profile, {"t_est_seconds": 600},
                              s1_hold_grants=2, s1_max_grants=4)
        for i in range(5):
            ctl.on_grant_result({
                "grants_used": i + 1,
                "remaining_wall_clock": 80000.0,
                "best_metric": 0.5,
                "metric_norm": None,   # no reference line
                "new_best": i == 0,
                "stagnation_count": i,
                "submission_exists": True,
                "regressions": 0,
                "intent": "cheap_probe",
            })
        check("S1 cap forces S2 exit", ctl.stage == "S2_enhancement",
              ctl.stage)
        check("S1 cap reason recorded",
              "S1 grant cap" in (ctl._last_reason or ""), ctl._last_reason)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_pre_grant_clip():
    """v2.2.1: pre-grant clipping must switch S2/S3 -> S4 BEFORE the
    planner runs so the next grant is planned cheap."""
    tmp = Path(tempfile.mkdtemp(prefix="v2_stage_pregrant_"))
    try:
        profile = _profile("accuracy", "higher_is_better", n_classes=2)
        ctl = StageController(tmp, profile,
                              {"t_est_seconds": 600, "max_budget_seconds": 600},
                              s1_hold_grants=1)
        _grants(ctl, [(0.85, True, 0, 0)])           # S1 -> S2
        check("in S2 before clip", ctl.stage == "S2_enhancement", ctl.stage)
        ctl.pre_grant_clip(60.0)
        check("pre-grant clip -> S4", ctl.stage == "S4_sprint", ctl.stage)
        check("clip persisted", (tmp / "stage_state.json").is_file())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_seed_branch_provenance():
    """v2.2.1: platform safety seeds are marked, not agent discoveries."""
    from hera.portfolio import MethodPortfolio, PortfolioBranch
    profile = _profile("accuracy", "higher_is_better", n_classes=2)
    port = MethodPortfolio.default_for(profile)
    baseline = port.get_branch("baseline")
    check("seed branch origin",
          baseline.origin == "platform_safety_seed"
          and baseline.scientific_discovery is False,
          str(asdict_safe(baseline)))
    check("HERA branch defaults to discovery",
          PortfolioBranch().origin == "hera"
          and PortfolioBranch().scientific_discovery is True)


def asdict_safe(b):
    from dataclasses import asdict
    return asdict(b)


def test_intent_children_whitelist():
    from hera.portfolio import INTENT_CHILD_RANGES, resolve_children
    check("cheap_probe range", INTENT_CHILD_RANGES["cheap_probe"] == (2, 4))
    check("cheap_probe requested 5 clamped", resolve_children("cheap_probe", 5) == 3,
          str(resolve_children("cheap_probe", 5)))
    check("cheap_probe requested 2 kept", resolve_children("cheap_probe", 2) == 2)
    check("final_training always 1",
          resolve_children("final_training", 3) == 1)
    check("unknown intent fallback", resolve_children("bogus", 2) == 2,
          str(resolve_children("bogus", 2)))


if __name__ == "__main__":
    print("=== V2 stage-controller tests ===\n")
    test_random_baseline_formulas()
    test_metric_norm_directions()
    test_s1_to_s2()
    test_s2_to_s3_on_stagnation()
    test_s3_to_s4_on_new_best()
    test_wall_clock_clipping()
    test_stage_profile_override()
    test_stage_state_restore()
    test_s1_max_grant_fallback_none_norm()
    test_pre_grant_clip()
    test_seed_branch_provenance()
    test_intent_children_whitelist()
    print("\nRESULT=" + ("PASS" if not FAILURES else "FAIL:" + ",".join(FAILURES)))
    sys.exit(0 if not FAILURES else 1)