# -*- coding: utf-8 -*-
"""v2.5.4 offline tests: declarative row cap + budget floor.

v2.5.4 (delivery-only hotfix, NOT pushed) fixes rc=-9 timeouts on large
dataset tasks (taxi 200k rows) with two declarative mechanisms:

  1) default_max_train_rows: per-capability runnability default (50000 for
     tabular/timeseries/ensemble, 20000 for image). normalize() injects it
     into resource_request.max_train_rows when HERA does not request one,
     so compiled templates finally subsample (MAX_TRAIN_ROWS was always 0
     -> full 200k rows -> HistGB overran cheap budgets under concurrency).
  2) min_budget_seconds: ResourceProfiler derives a trial-timeout floor
     (max(300, t_est * 0.5)) so an over-optimistic LLM budget can never
     schedule a trial that the platform's own runtime estimate says cannot
     finish. Planner clamps plan budgets into [min_budget, max_budget].
  3) The child-proposal contract now exposes train_rows_cap and accepts an
     OPTIONAL resource_request.max_train_rows (clamped to the cap), so the
     row cap is transparent to HERA, not a hidden override.

Contracts verified here:
  - AST: whole payload keeps ZERO method_id.startswith routing / zero
    renderer== chains / zero competition-name hardcoding.
  - Registry: the 10 data-heavy built-in capabilities declare the expected
    default_max_train_rows; text/pixel/mask/detection/audio stay 0.
  - normalize() injects the default only when absent; explicit requests
    pass through.
  - render() bakes the injected cap into the compiled code (MAX_TRAIN_ROWS).
  - ResourceProfiler: min_budget_seconds in [300, max_budget_seconds], and
    the taxi-shaped profile (200k tabular rows) has a floor > 300.
  - Planner: a fake LLM asking for 300s on the taxi profile is raised to
    the platform floor.

Run: python test_v2_254.py   (from the aisci payload dir)
"""
import ast
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from capability_registry import CapabilityRegistry  # noqa: E402
from hera.portfolio import ResourceProfiler  # noqa: E402
from hera.planner import Planner  # noqa: E402
from program_compiler import ProgramCompiler, _TEMPLATE_REGISTRY  # noqa: E402
from v2_contracts import AnalysisProfile, MethodInvocationV1  # noqa: E402

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("[OK] %s" % name)
    else:
        FAIL += 1
        print("[FAIL] %s | %s" % (name, detail))


# ------------------------------------------------------------- AST invariants
EXPECT_ROWCAP = {
    "tabular.linear.logistic.v1": 50000,
    "tabular.gbdt.histgb.v1": 50000,
    "tabular.datetime_feature_histgb.v1": 50000,
    "tabular.neural.mlp.v1": 50000,
    "timeseries.lag_histgb.v1": 50000,
    "ensemble.sklearn_soft_vote.v1": 50000,
    "image.embedding.timm.v1": 20000,
    "image.finetune.timm.v1": 20000,
    "image.finetune.timm.v2": 20000,
    "image.finetune.ensemble.v1": 20000,
}
ZERO_ROWCAP = {
    "text.embedding.tfidf.v1", "text.neural.mlp.v1",
    "image.pixel.baseline.v1", "image.mask.rle.baseline.v1",
    "image.detection.bbox.baseline.v1", "audio.tabular.baseline.v1",
}


def _logic_files():
    out = []
    for p in sorted(HERE.rglob("*.py")):
        rel = str(p.relative_to(HERE))
        if any(part.startswith(".v2_backup") or part == "__pycache__"
               for part in rel.split("/")):
            continue
        if p.name.startswith("test_"):
            continue
        out.append(p)
    return out


def test_ast_invariants():
    # mirrors test_v2_250 / test_v2_252 frozen scans
    bad_startswith = []
    for p in _logic_files():
        src = p.read_text(encoding="utf-8", errors="replace")
        if "method_id.startswith" in src:
            bad_startswith.append(str(p.relative_to(HERE)))
    check("no method_id.startswith routing in payload", not bad_startswith,
          "; ".join(bad_startswith[:3]))
    pc_src = (HERE / "program_compiler.py").read_text(encoding="utf-8")
    bad_renderer = [ln for ln in pc_src.splitlines()
                    if "spec.renderer ==" in ln or "renderer ==" in ln]
    check("compiler has no renderer == chains", not bad_renderer,
          "; ".join(bad_renderer[:3]))
    try:
        from metrics_registry import COMPETITION_METRICS  # noqa: E402
        comps = set((COMPETITION_METRICS or {}).keys())
    except Exception:  # noqa: BLE001
        comps = set()
    bad_names = []
    for p in _logic_files():
        if p.name == "metrics_registry.py":
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and node.value.strip().lower() in comps):
                bad_names.append("%s:%d" % (p.relative_to(HERE), node.lineno))
    check("no competition-name hardcoding in payload", not bad_names,
          "; ".join(bad_names[:5]))


# --------------------------------------------------------------- registry
def test_registry_row_caps():
    reg = CapabilityRegistry()
    for mid, cap in sorted(EXPECT_ROWCAP.items()):
        spec = reg.get(mid)
        check("row cap %s = %d" % (mid, cap),
              spec is not None and int(spec.default_max_train_rows or 0) == cap,
              str(getattr(spec, "default_max_train_rows", None)))
    for mid in sorted(ZERO_ROWCAP):
        spec = reg.get(mid)
        check("row cap %s = 0 (unchanged)" % mid,
              spec is not None and int(spec.default_max_train_rows or 0) == 0,
              str(getattr(spec, "default_max_train_rows", None)))


# ---------------------------------------------------------------- normalize
def test_normalize_injects_default():
    reg = CapabilityRegistry()
    pc = ProgramCompiler(reg)
    inv = MethodInvocationV1(method_id="tabular.gbdt.histgb.v1",
                             hypothesis="no rr")
    out = pc.normalize(inv)
    check("normalize injects 50000 for tabular histgb",
          int(out.resource_request.get("max_train_rows") or 0) == 50000,
          str(out.resource_request))
    inv2 = MethodInvocationV1(method_id="image.finetune.timm.v1",
                              hypothesis="img")
    out2 = pc.normalize(inv2)
    check("normalize injects 20000 for image finetune",
          int(out2.resource_request.get("max_train_rows") or 0) == 20000,
          str(out2.resource_request))
    inv3 = MethodInvocationV1(method_id="tabular.gbdt.histgb.v1",
                              hypothesis="explicit",
                              resource_request={"max_train_rows": 20000})
    out3 = pc.normalize(inv3)
    check("explicit max_train_rows wins",
          int(out3.resource_request.get("max_train_rows") or 0) == 20000,
          str(out3.resource_request))
    inv4 = MethodInvocationV1(method_id="text.embedding.tfidf.v1",
                              hypothesis="text no cap")
    out4 = pc.normalize(inv4)
    check("text keeps no default cap",
          int(out4.resource_request.get("max_train_rows") or 0) == 0,
          str(out4.resource_request))


def _taxi_profile():
    return AnalysisProfile(
        train_rows=200000, test_rows=9914, modality="tabular",
        task_type="regression", n_classes=0, feature_dim=18,
        metric_name="rmse", metric_direction="lower_is_better",
        text_columns=["key"], datetime_columns=["pickup_datetime"],
        time_column="pickup_datetime", feature_columns=[])


def _taxi_manifest():
    return {
        "layout": "flat", "train_csv": "train.csv", "test_csv": "test.csv",
        "sample_submission": "sample_submission.csv",
        "target_column": "fare_amount", "metric_name": "rmse",
        "task_type": "regression",
    }


def test_render_bakes_row_cap():
    reg = CapabilityRegistry()
    pc = ProgramCompiler(reg)
    prof = _taxi_profile()
    man = _taxi_manifest()
    inv = MethodInvocationV1(method_id="tabular.datetime_feature_histgb.v1",
                             hypothesis="taxi", params={"val_seed": 42})
    code, th = pc.render(inv, profile=prof, manifest=man)
    check("render succeeds", len(code) > 1000 and th.startswith("sha256:"), "")
    check("code bakes MAX_TRAIN_ROWS = 50000",
          "MAX_TRAIN_ROWS = int('50000'" in code,
          "MAX_TRAIN_ROWS = int('%s'" % (
              code.split("MAX_TRAIN_ROWS = int('")[1][:20]
              if "MAX_TRAIN_ROWS = int('" in code else "?"))
    check("code has subsample guard",
          "if MAX_TRAIN_ROWS > 0 and n > MAX_TRAIN_ROWS:" in code, "")
    inv2 = MethodInvocationV1(method_id="tabular.datetime_feature_histgb.v1",
                              hypothesis="taxi-small",
                              params={"val_seed": 42},
                              resource_request={"max_train_rows": 12000})
    code2, _ = pc.render(inv2, profile=prof, manifest=man)
    check("explicit smaller cap baked",
          "MAX_TRAIN_ROWS = int('12000'" in code2, "")


# ---------------------------------------------------------- resource profile
def test_resource_min_budget():
    prof = _taxi_profile()
    rp = ResourceProfiler(gpu_memory_mb=40960, cached_weights=[])
    res = rp.derive(prof)
    mb = int(res.get("min_budget_seconds") or 0)
    mx = int(res.get("max_budget_seconds") or 0)
    check("taxi profile derives max_budget", mx >= 900, "max=%d" % mx)
    check("taxi profile derives min_budget floor > 300",
          300 <= mb <= mx and mb > 300,
          "min=%d max=%d" % (mb, mx))
    small = AnalysisProfile(
        train_rows=2000, test_rows=500, modality="tabular",
        task_type="classification", n_classes=2, feature_dim=8,
        metric_name="accuracy", metric_direction="higher_is_better")
    res2 = rp.derive(small)
    mb2 = int(res2.get("min_budget_seconds") or 0)
    check("small dataset min_budget stays 300",
          mb2 == 300, "min=%d" % mb2)
    res3 = rp.derive(prof)
    check("derive is deterministic", res == res3, "")


# ------------------------------------------------------------------ planner
def test_planner_budget_floor():
    prof = _taxi_profile()
    rp = ResourceProfiler(gpu_memory_mb=40960, cached_weights=[])
    resource = rp.derive(prof)
    mb = int(resource.get("min_budget_seconds") or 0)

    def fake_llm(prompt):
        return '{"hypothesis": "h", "research_intent": "cheap_probe", ' \
               '"max_budget_seconds": 300}'

    pl = Planner(llm_call_fn=fake_llm)
    plan = pl.plan(profile=prof, evidence="", round_num=1, elapsed=10,
                   total_budget=86400, resource=resource,
                   stage_block="", intent_hints=None)
    check("planner clamps 300 up to min_budget",
          int(plan.max_budget_seconds) >= mb,
          "plan=%d min=%d" % (plan.max_budget_seconds, mb))

    def fake_llm_big(prompt):
        return '{"hypothesis": "h", "research_intent": "final_training", ' \
               '"max_budget_seconds": 99999}'

    pl2 = Planner(llm_call_fn=fake_llm_big)
    plan2 = pl2.plan(profile=prof, evidence="", round_num=1, elapsed=10,
                     total_budget=86400, resource=resource,
                     stage_block="", intent_hints=None)
    mx = int(resource.get("max_budget_seconds") or 0)
    check("planner clamps down to max_budget",
          int(plan2.max_budget_seconds) <= mx,
          "plan=%d max=%d" % (plan2.max_budget_seconds, mx))


def main():
    test_ast_invariants()
    test_registry_row_caps()
    test_normalize_injects_default()
    test_render_bakes_row_cap()
    test_resource_min_budget()
    test_planner_budget_floor()
    print("RESULT=%s ok=%d fail=%d" % ("PASS" if FAIL == 0 else "FAIL", PASS, FAIL))
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
