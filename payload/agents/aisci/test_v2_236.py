# -*- coding: utf-8 -*-
"""test_v2_236.py - v2.3.6 per-metric min_delta + recovery consistency.

Root cause fixed: a single global min_delta=0.01 blocked real improvements
forever for bounded/score metrics (aerial AUC 0.9997 vs 0.9972 is +0.0025,
never promotable) and let the certified pointer lag behind the ledger.
v2.3.6 makes the improvement threshold a property of the METRIC FAMILY and
restores the BETTER of (certified pointer, ledger best) with its code asset
on restart.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from metrics_registry import (DEFAULT_MIN_DELTA, METRIC_MIN_DELTA,  # noqa: E402
                              get_metric_spec, infer_metric_spec,
                              metric_min_delta)
from v2_closed_loop import ClosedLoop  # noqa: E402
from pact import (FileBus, HostSupervisorService, PactLedger,  # noqa: E402
                  PromotionManager, TrustedEvaluator)
from pact.executor import ExecOutcome  # noqa: E402
from v2_contracts import (PromotionRecord, ResearchPlan,  # noqa: E402
                          TrialReceipt, TrialSpec)

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("[OK] " + name)
    else:
        print("[FAIL] " + name + (" | " + detail if detail else ""))
        FAILURES.append(name)


def test_metric_min_delta_registry():
    spec = get_metric_spec("aerial-cactus-identification")
    check("auc min_delta=1e-4", spec.get("min_delta") == 1e-4,
          str(spec.get("min_delta")))
    check("qwk min_delta=1e-4",
          get_metric_spec("aptos2019-blindness-detection")["min_delta"] == 1e-4)
    check("logloss min_delta=1e-4",
          get_metric_spec("spooky-author-identification")["min_delta"] == 1e-4)
    check("mae min_delta=1e-3",
          get_metric_spec("ventilator-pressure-prediction")["min_delta"] == 1e-3)
    check("rmse min_delta=1e-3",
          get_metric_spec("new-york-city-taxi-fare-prediction")["min_delta"] == 1e-3)
    check("infer regression rmse 1e-3",
          infer_metric_spec("regression")["min_delta"] == 1e-3)
    check("infer classification accuracy 1e-4",
          infer_metric_spec("classification")["min_delta"] == 1e-4)
    check("unknown family keeps legacy default",
          metric_min_delta("brand_new_metric") == DEFAULT_MIN_DELTA
          and "brand_new_metric" not in METRIC_MIN_DELTA)
    check("default constant is 0.01", DEFAULT_MIN_DELTA == 0.01)


def test_closed_loop_is_better():
    """Real production shapes that the old 0.01 gate destroyed."""
    loop = object.__new__(ClosedLoop)
    loop.metric_spec = get_metric_spec("aerial-cactus-identification")
    check("aerial 0.9997 > 0.99718 is better",
          loop._is_better(0.9997, 0.99718))
    check("aerial sub-delta noise rejected",
          not loop._is_better(0.99718 + 0.00005, 0.99718))
    loop.metric_spec = get_metric_spec("aptos2019-blindness-detection")
    check("aptos 0.896 > 0.8919 is better",
          loop._is_better(0.896, 0.8919))
    check("aptos 0.89195 vs 0.8919 below delta rejected",
          not loop._is_better(0.89195, 0.8919))
    loop.metric_spec = get_metric_spec("spooky-author-identification")
    check("logloss 0.3749 < 0.37517 is better",
          loop._is_better(0.3749, 0.37517))
    check("logloss 0.37510 vs 0.37517 below delta rejected",
          not loop._is_better(0.37510, 0.37517))
    loop.metric_spec = get_metric_spec("new-york-city-taxi-fare-prediction")
    check("rmse 1.0 < 1.005 is better (1e-3)",
          loop._is_better(1.0, 1.005))
    check("rmse 1.0045 vs 1.005 below delta rejected",
          not loop._is_better(1.0045, 1.005))
    # legacy default preserved for specs without min_delta
    loop.metric_spec = {"metric_direction": "higher_is_better"}
    check("legacy default keeps 0.01",
          not loop._is_better(1.005, 1.0) and loop._is_better(1.011, 1.0))


def test_host_supervisor_min_delta():
    from pact.file_bus import FileBus as _FB
    tmp = Path(tempfile.mkdtemp(prefix="v2_236_host_"))
    try:
        bus = _FB(tmp / "state")
        dummy_exec = type("DummyExec", (), {"work_dir": str(tmp / "work")})()
        host = HostSupervisorService(
            bus=bus, executor=dummy_exec, bundler=None,
            evaluator=TrustedEvaluator(metric_name="rmse",
                                       metric_direction="lower_is_better"),
            promotion=PromotionManager(bus, metric_direction="lower_is_better"),
            implementer=None, competition="demo", metric_min_delta=1e-3)
        check("host min_delta=1e-3: rmse 1.0 vs 1.005 success",
              host._verdict(1.0, 1.005, 0) == "success")
        check("host min_delta=1e-3: 1.0045 vs 1.005 stagnant",
              host._verdict(1.0045, 1.005, 0) == "stagnant")
        check("host min_delta=1e-3: 1.0065 vs 1.005 regression",
              host._verdict(1.0065, 1.005, 0) == "regression")
        check("host min_delta=1e-3: 1.0045 not worse",
              not host._worse(1.0045, 1.005))
        default = HostSupervisorService(
            bus=bus, executor=dummy_exec, bundler=None,
            evaluator=TrustedEvaluator(metric_name="rmse",
                                       metric_direction="lower_is_better"),
            promotion=PromotionManager(bus, metric_direction="lower_is_better"),
            implementer=None, competition="demo")
        check("host default keeps 0.01 (legacy behavior)",
              default._verdict(1.005, 1.0, 0) == "stagnant")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_promotion_min_delta():
    tmp = Path(tempfile.mkdtemp(prefix="v2_236_promo_"))
    try:
        bus = FileBus(tmp / "state")
        pm = PromotionManager(bus, metric_direction="higher_is_better",
                              min_delta=1e-4)
        r1 = pm.promote("t1", 0.9997)
        check("first verified trial promoted", r1.decision == "promote")
        r2 = pm.promote("t2", 0.9972)
        check("sub-delta rejected", r2.decision == "reject")
        r3 = pm.promote("t3", 0.9999)
        check("above 1e-4 promoted", r3.decision == "promote",
              str(r3.decision))
        check("certified pointer advanced",
              r3.certified_best_metric == 0.9999)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_verifier_min_delta():
    tmp = Path(tempfile.mkdtemp(prefix="v2_236_verify_"))
    try:
        sub = tmp / "sub"
        code = ("import csv\n"
                "with open('submission.csv', 'w', newline='') as f:\n"
                "    w = csv.writer(f)\n"
                "    w.writerow(['Id', 'Prediction'])\n"
                "    for i in range(5):\n"
                "        w.writerow([i, 0])\n"
                "print('accuracy: 0.85005')\n")
        spec = TrialSpec.seal("demo", ResearchPlan(method_detail={"model": "stub"}), code)
        from pact.verifier import Verifier
        verifier = Verifier(sub, min_delta=1e-4)
        r1 = verifier.verify(spec, ExecOutcome(returncode=0,
                                               stdout="accuracy: 0.85020\n"),
                             best_metric=0.85)
        check("verifier min_delta=1e-4: +0.0002 success",
              r1.verdict == "success", r1.verdict)
        r2 = verifier.verify(spec, ExecOutcome(returncode=0,
                                               stdout="accuracy: 0.85005\n"),
                             best_metric=0.85)
        check("verifier min_delta=1e-4: +0.00005 stagnant",
              r2.verdict == "stagnant", r2.verdict)
        legacy = Verifier(sub)
        r3 = legacy.verify(spec, ExecOutcome(returncode=0,
                                             stdout="accuracy: 0.85020\n"),
                           best_metric=0.85)
        check("verifier default keeps 0.01 (legacy behavior)",
              r3.verdict == "stagnant", r3.verdict)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_recover_ledger_best_syncs_code():
    """Restart continuity (v2.3.6): when the ledger holds a better verified
    metric than the certified pointer (old min_delta rejected it), recovery
    restores the ledger best AND persists its code as the incumbent asset -
    metric and code must never drift apart across a restart."""
    import types as _types
    tmp = Path(tempfile.mkdtemp(prefix="v2_236_recover_"))
    try:
        state = tmp / "state"
        bus = FileBus(state)
        # Certified pointer is the OLD best (round-1 0.99718) because the
        # old 0.01 gate rejected 0.9997.
        bus.save_promotion(PromotionRecord(
            competition="aerial-cactus-identification",
            certified_best_trial_id="receipt_old",
            certified_best_metric=0.99718,
            incumbent_trial_id="receipt_old",
            incumbent_metric=0.99718,
            decision="promote", reason="old gate",
        ).to_dict())
        # Ledger holds the better verified trial (rc=0).
        ledger = PactLedger(state)
        receipt = TrialReceipt(
            receipt_id="receipt_new", spec_id="spec_new",
            competition="aerial-cactus-identification", round_num=2,
            returncode=0, stdout="", stderr="",
            metric=0.9997, metric_name="auc", verdict="success",
            evidence="Improved", code_hash="sha256:deadbeef",
            verified=True, submission_exists=True)
        ledger.append(receipt, ResearchPlan(round_num=2))
        # Host receipt store maps receipt_id -> spec_id; code lives in ws.
        bus.host_receipts.mkdir(parents=True, exist_ok=True)
        (bus.host_receipts / "receipt_receipt_new.json").write_text(
            json.dumps({"receipt_id": "receipt_new",
                        "spec_id": "spec_new"}), encoding="utf-8")
        bus.ws_code.mkdir(parents=True, exist_ok=True)
        (bus.ws_code / "trial_spec_new.py").write_text(
            "print('ledger best code')\n", encoding="utf-8")

        loop = object.__new__(ClosedLoop)
        loop.bus = bus
        loop.ledger = ledger
        loop.competition = "aerial-cactus-identification"
        loop.metric_spec = get_metric_spec("aerial-cactus-identification")
        loop.promotion = PromotionManager(
            bus, metric_direction="higher_is_better", min_delta=1e-4)
        loop.state_dir = state
        loop.guard = _types.SimpleNamespace(grants_used=0)
        loop._log = lambda msg: print("  [recover] " + msg)
        loop._recover_scientific_state()

        check("recovery restores ledger best 0.9997",
              loop.best_metric == 0.9997, str(loop.best_metric))
        check("recovery ties best to ledger receipt",
              loop.best_receipt_id == "receipt_new",
              str(loop.best_receipt_id))
        inc = json.loads((state / "incumbent_best.json").read_text(
            encoding="utf-8"))
        check("incumbent asset synced to ledger metric",
              inc.get("metric") == 0.9997, str(inc.get("metric")))
        code_path = Path(inc.get("code_path") or "")
        check("incumbent code asset exists",
              code_path.is_file() and "ledger best code" in code_path.read_text(
                  encoding="utf-8"))
        check("incumbent receipt_id matches",
              inc.get("receipt_id") == "receipt_new",
              str(inc.get("receipt_id")))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print("=== V2.3.6 min_delta + recovery tests ===\n")
    test_metric_min_delta_registry()
    test_closed_loop_is_better()
    test_host_supervisor_min_delta()
    test_promotion_min_delta()
    test_verifier_min_delta()
    test_recover_ledger_best_syncs_code()
    print("\nRESULT=" + ("PASS" if not FAILURES else "FAIL:" + ",".join(FAILURES)))
