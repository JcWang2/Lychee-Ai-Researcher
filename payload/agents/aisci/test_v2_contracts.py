# -*- coding: utf-8 -*-
"""test_v2_contracts.py - Data contract round-trip tests."""
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from v2_contracts import AnalysisProfile, ResearchPlan, TrialReceipt, TrialSpec  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("[OK] " + name)
    else:
        print("[FAIL] " + name + (" | " + detail if detail else ""))
        FAILURES.append(name)


def test_plan_roundtrip():
    plan = ResearchPlan(round_num=2, hypothesis="H", approach_type="exploit",
                        method_detail={"model": "lightgbm"})
    plan2 = ResearchPlan.from_dict(plan.to_dict())
    check("ResearchPlan roundtrip", plan2.to_dict() == plan.to_dict())


def test_spec_seal():
    plan = ResearchPlan(method_detail={"model": "xgb"})
    code = "import os\nprint('x')\n"
    spec = TrialSpec.seal("demo", plan, code)
    check("spec code_hash prefix", spec.code_hash.startswith("sha256:"))
    expected = "sha256:" + hashlib.sha256(code.encode("utf-8")).hexdigest()
    check("spec code_hash matches content", spec.code_hash == expected)
    check("spec plan snapshot", spec.plan_obj().method_detail == {"model": "xgb"})
    check("spec round_num carried", spec.round_num == plan.round_num)


def test_receipt_roundtrip():
    r = TrialReceipt(receipt_id="r1", spec_id="s1", verdict="success", metric=0.83)
    r2 = TrialReceipt.from_dict(r.to_dict())
    check("TrialReceipt roundtrip", r2.to_dict() == r.to_dict())


def test_profile_roundtrip():
    p = AnalysisProfile(competition="c", task_type="regression",
                        feature_columns=["a", "b"])
    p2 = AnalysisProfile.from_dict(p.to_dict())
    check("AnalysisProfile roundtrip", p2.to_dict() == p.to_dict())


if __name__ == "__main__":
    print("=== V2 contracts tests ===\n")
    test_plan_roundtrip()
    test_spec_seal()
    test_receipt_roundtrip()
    test_profile_roundtrip()
    print("\nRESULT=" + ("PASS" if not FAILURES else "FAIL:" + ",".join(FAILURES)))
    sys.exit(0 if not FAILURES else 1)
