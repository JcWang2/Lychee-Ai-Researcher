# -*- coding: utf-8 -*-
"""test_v2_23.py - v2.3 template-compiled execution acceptance (Phase A+B+C).

Covers the v2.3 power boundary:
  Phase A (compiler-first): proposal carries MethodInvocationV1 ->
      deterministic render, 0 LLM calls, seal pins invocation_hash +
      template_hash, bit-for-bit replay.
  Phase B (capability registry): HERA chooses from the filtered registry;
      unknown/invalid choices fall back deterministically; no competition
      name in the decision path.
  Phase C (capability synthesis): a proven gap mints ONE bounded ephemeral
      adapter (MAX_SYNTHESIS_ACTIONS, persisted); reuse changes params
      only; failures mark the capability broken.
"""
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from v2_closed_loop import ClosedLoop  # noqa: E402
from v2_contracts import MethodInvocationV1, ResearchPlan  # noqa: E402
from capability_registry import (CapabilityRegistry, MethodSpec,  # noqa: E402
                                 load_ephemeral_path,
                                 load_synthesis_usage)
from program_compiler import ProgramCompiler  # noqa: E402
from pact import FileBus, HostSupervisorService  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("[OK] " + name)
    else:
        print("[FAIL] " + name + (" | " + detail if detail else ""))
        FAILURES.append(name)


PROFILE = types.SimpleNamespace(
    modality="tabular", task_type="classification", metric_name="accuracy",
    metric_direction="maximize", train_rows=120, test_rows=30, n_classes=2,
    feature_dim=8, image_width=0, image_height=0)

MANIFEST = {"task_type": "classification", "metric_name": "accuracy",
            "train_rows": 120}

GRANT = {
    "grant_id": "grant_v23_test",
    "directive_hash": "sha256:test",
    "selected_branch_id": "branch_logreg",
    "mutation_axis": "hyperparameter",
    "research_intent": "cheap_probe",
    "stage": "S1_baseline",
    "trial_budget": 3,
    "competition": "stub_comp",
    "task_prompt": "stub task",
    "status": "frozen",
}

_SYNTH_SOURCE = (
    "def build_model(params, seed):\n"
    "    from sklearn.linear_model import LogisticRegression\n"
    "    return LogisticRegression(C=params.get('alpha', 1.0), "
    "max_iter=params.get('max_iter', 200))\n"
)


def _make_loop(state_dir):
    loop = object.__new__(ClosedLoop)
    loop.registry = CapabilityRegistry(
        ephemeral_path=load_ephemeral_path(state_dir))
    loop.compiler = ProgramCompiler(loop.registry)
    loop.planner = types.SimpleNamespace(llm_call=None)
    loop._manifest = dict(MANIFEST)
    loop.state_dir = Path(state_dir)
    return loop


def _first_compatible(loop, modality="tabular", task_type="classification",
                      metric="accuracy"):
    specs = loop.registry.compatible(modality, task_type, metric)
    return specs[0].method_id if specs else ""


# ---------------------------------------------------------------- Phase B
def test_proposer_valid_invocation():
    tmp = Path(tempfile.mkdtemp(prefix="v23_prop_"))
    try:
        loop = _make_loop(tmp)

        def fake_llm(prompt):
            return json.dumps({
                "method_id": "tabular.gbdt.histgb.v1",
                "params": {"learning_rate": 0.07, "max_iter": 600},
                "preprocessing": ["missing_value_native"],
                "validation": "single_holdout",
                "hypothesis": "HistGB: lower LR + more iters should beat "
                              "logistic baseline",
            })

        loop.planner.llm_call = fake_llm
        proposer = loop._agent_proposer(dict(GRANT), PROFILE)
        out = proposer(1, dict(GRANT), "child 0 evidence")
        inv = out.get("invocation") or {}
        check("valid invocation returned", bool(inv.get("method_id")),
              str(inv)[:200])
        check("HERA method kept",
              inv.get("method_id") == "tabular.gbdt.histgb.v1", str(inv))
        check("HERA params kept", inv["params"].get("learning_rate") == 0.07,
              str(inv.get("params")))
        check("preprocessing kept",
              "missing_value_native" in inv.get("preprocessing", []))
        check("validation kept", inv.get("validation") == "single_holdout")
        check("invocation hash computable",
              MethodInvocationV1.from_dict(inv).compute_hash()
              .startswith("sha256:"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_proposer_preprocessing_alias_normalized():
    """HERA/LLM synonyms (impute_mean/standardize/kfold) are normalized onto
    canonical registry tokens so valid research choices are honored instead
    of being rejected by schema strictness."""
    tmp = Path(tempfile.mkdtemp(prefix="v23_prop_alias_"))
    try:
        loop = _make_loop(tmp)

        def fake_llm(prompt):
            return json.dumps({
                "method_id": "tabular.linear.logistic.v1",
                "params": {"C": 0.5},
                "preprocessing": ["impute_mean", "standardize"],
                "validation": "kfold",
                "hypothesis": "alias test",
            })

        loop.planner.llm_call = fake_llm
        proposer = loop._agent_proposer(dict(GRANT), PROFILE)
        out = proposer(1, dict(GRANT), "child 0 evidence")
        inv = out.get("invocation") or {}
        check("alias invocation returned",
              inv.get("method_id") == "tabular.linear.logistic.v1", str(inv))
        pre = inv.get("preprocessing") or []
        check("preprocessing aliases normalized",
              "missing_value_impute" in pre and "standard_scaling" in pre,
              str(pre))
        check("validation alias normalized",
              inv.get("validation") == "stratified_kfold", str(inv))
        ok, reason = loop.compiler.validate(
            MethodInvocationV1.from_dict(inv), PROFILE, MANIFEST)
        check("alias invocation validates", ok, reason)
        code, th = loop.compiler.render(
            MethodInvocationV1.from_dict(inv), PROFILE, MANIFEST)
        check("alias invocation renders",
              "make_estimator" in code and th.startswith("sha256:"),
              str(len(code)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def test_proposer_leave_oof_alias_normalized():
    """HERA 'leave'/'missing:leave' (gbdt missing param) and 'oof' validation
    are normalized onto canonical registry tokens instead of being rejected."""
    tmp = Path(tempfile.mkdtemp(prefix="v23_prop_leave_"))
    try:
        loop = _make_loop(tmp)

        def fake_llm(prompt):
            return json.dumps({
                "method_id": "tabular.gbdt.histgb.v1",
                "params": {"max_iter": 500},
                "preprocessing": ["missing:leave"],
                "validation": "oof",
                "hypothesis": "leave/oof alias test",
            })

        loop.planner.llm_call = fake_llm
        proposer = loop._agent_proposer(dict(GRANT), PROFILE)
        out = proposer(1, dict(GRANT), "child 0 evidence")
        inv = out.get("invocation") or {}
        check("leave/oof invocation returned",
              inv.get("method_id") == "tabular.gbdt.histgb.v1", str(inv))
        pre = inv.get("preprocessing") or []
        check("missing:leave normalized",
              "missing_value_native" in pre, str(pre))
        check("oof validation normalized",
              inv.get("validation") == "stratified_kfold", str(inv))
        ok, reason = loop.compiler.validate(
            MethodInvocationV1.from_dict(inv), PROFILE, MANIFEST)
        check("leave/oof invocation validates", ok, reason)
        code, th = loop.compiler.render(
            MethodInvocationV1.from_dict(inv), PROFILE, MANIFEST)
        check("leave/oof invocation renders",
              "make_estimator" in code and th.startswith("sha256:"),
              str(len(code)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def test_proposer_imputation_alias_normalized():
    """'imputation' and kfold variants map to canonical tokens (v2.3.1)."""
    tmp = Path(tempfile.mkdtemp(prefix="v23_prop_impute_"))
    try:
        loop = _make_loop(tmp)

        def fake_llm(prompt):
            return json.dumps({
                "method_id": "tabular.linear.logistic.v1",
                "params": {"C": 0.4},
                "preprocessing": ["imputation", "standardize"],
                "validation": "kfold_cv",
                "hypothesis": "imputation alias test",
            })

        loop.planner.llm_call = fake_llm
        proposer = loop._agent_proposer(dict(GRANT), PROFILE)
        out = proposer(1, dict(GRANT), "child 0 evidence")
        inv = out.get("invocation") or {}
        check("imputation invocation returned",
              inv.get("method_id") == "tabular.linear.logistic.v1", str(inv))
        pre = inv.get("preprocessing") or []
        check("imputation normalized",
              "missing_value_impute" in pre and "standard_scaling" in pre,
              str(pre))
        check("kfold_cv validation normalized",
              inv.get("validation") == "stratified_kfold", str(inv))
        ok, reason = loop.compiler.validate(
            MethodInvocationV1.from_dict(inv), PROFILE, MANIFEST)
        check("imputation invocation validates", ok, reason)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_proposer_partial_fallback_keeps_method():
    """v2.3.1: an invalid preprocessing token sanitizes ONLY that field;
    HERA's method + params survive instead of a full deterministic fallback."""
    tmp = Path(tempfile.mkdtemp(prefix="v23_prop_partial_"))
    try:
        loop = _make_loop(tmp)

        def fake_llm(prompt):
            return json.dumps({
                "method_id": "tabular.gbdt.histgb.v1",
                "params": {"max_iter": 500, "learning_rate": 0.07},
                "preprocessing": ["bogus_token"],
                "validation": "oof",
                "hypothesis": "partial fallback test",
            })

        loop.planner.llm_call = fake_llm
        proposer = loop._agent_proposer(dict(GRANT), PROFILE)
        out = proposer(1, dict(GRANT), "child 0 evidence")
        inv = out.get("invocation") or {}
        check("partial fallback keeps method",
              inv.get("method_id") == "tabular.gbdt.histgb.v1", str(inv))
        check("partial fallback keeps params",
              inv.get("params", {}).get("max_iter") == 500
              and inv.get("params", {}).get("learning_rate") == 0.07,
              str(inv.get("params")))
        pre = inv.get("preprocessing") or []
        check("bogus preprocessing dropped",
              "bogus_token" not in pre, str(pre))
        check("oof validation normalized",
              inv.get("validation") == "stratified_kfold", str(inv))
        ok, reason = loop.compiler.validate(
            MethodInvocationV1.from_dict(inv), PROFILE, MANIFEST)
        check("partial fallback validates", ok, reason)
        code, th = loop.compiler.render(
            MethodInvocationV1.from_dict(inv), PROFILE, MANIFEST)
        check("partial fallback renders",
              "make_estimator" in code and th.startswith("sha256:"),
              str(len(code)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def test_proposer_unknown_method_fallback():
    tmp = Path(tempfile.mkdtemp(prefix="v23_prop_"))
    try:
        loop = _make_loop(tmp)

        def fake_llm(prompt):
            return json.dumps({
                "method_id": "tabular.definitely.not.real.v9",
                "params": {},
                "preprocessing": [],
                "validation": "stratified_kfold",
                "hypothesis": "unknown method probe",
            })

        loop.planner.llm_call = fake_llm
        proposer = loop._agent_proposer(dict(GRANT), PROFILE)
        out = proposer(2, dict(GRANT), "")
        inv = out.get("invocation") or {}
        expected = _first_compatible(loop)
        check("fallback to first compatible",
              inv.get("method_id") == expected,
              "got %r expected %r" % (inv.get("method_id"), expected))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_proposer_llm_error_fallback():
    tmp = Path(tempfile.mkdtemp(prefix="v23_prop_"))
    try:
        loop = _make_loop(tmp)

        def fake_llm(prompt):
            raise RuntimeError("llm down")

        loop.planner.llm_call = fake_llm
        proposer = loop._agent_proposer(dict(GRANT), PROFILE)
        out = proposer(3, dict(GRANT), "")
        inv = out.get("invocation") or {}
        expected = _first_compatible(loop)
        check("crash fallback to first compatible",
              inv.get("method_id") == expected,
              "got %r expected %r" % (inv.get("method_id"), expected))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_proposer_schema_reject_fallback():
    tmp = Path(tempfile.mkdtemp(prefix="v23_prop_"))
    try:
        loop = _make_loop(tmp)

        def fake_llm(prompt):
            return json.dumps({
                "method_id": "tabular.gbdt.histgb.v1",
                "params": {"learning_rate": 99.0},   # out of schema range
                "preprocessing": [],
                "validation": "stratified_kfold",
                "hypothesis": "out-of-range probe",
            })

        loop.planner.llm_call = fake_llm
        proposer = loop._agent_proposer(dict(GRANT), PROFILE)
        out = proposer(1, dict(GRANT), "")
        inv = out.get("invocation") or {}
        expected = _first_compatible(loop)
        check("schema reject falls back deterministically",
              inv.get("method_id") == expected,
              "got %r expected %r" % (inv.get("method_id"), expected))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- Phase C
def test_proposer_capability_gap_synthesis():
    tmp = Path(tempfile.mkdtemp(prefix="v23_synth_"))
    try:
        loop = _make_loop(tmp)
        calls = []

        def fake_llm(prompt):
            calls.append(prompt)
            if "CAPABILITY SYNTHESIS" in prompt:
                return json.dumps({
                    "method_id": "my_special_net",
                    "family": "ephemeral",
                    "supported_modalities": ["tabular"],
                    "supported_tasks": ["classification"],
                    "metric_outputs": {"accuracy": "class"},
                    "parameter_schema": {
                        "alpha": {"type": "float", "min": 0.0, "max": 1.0,
                                  "default": 0.5},
                        "max_iter": {"type": "int", "min": 50, "max": 1000,
                                     "default": 200}},
                    "preprocessing_options": ["missing_value_impute"],
                    "validation_schemes": ["stratified_kfold",
                                           "single_holdout"],
                    "source_code": _SYNTH_SOURCE,
                    "description": "special net adapter",
                })
            return json.dumps({
                "capability_gap": "registry cannot express my special net",
                "hypothesis": "special net should capture interactions",
            })

        loop.planner.llm_call = fake_llm
        proposer = loop._agent_proposer(dict(GRANT), PROFILE)
        out = proposer(1, dict(GRANT), "")
        inv = out.get("invocation") or {}
        check("ephemeral method used",
              inv.get("method_id") == "ephemeral.my_special_net",
              str(inv)[:200])
        check("synthesis usage persisted",
              load_synthesis_usage(tmp).get("used") == 1,
              str(load_synthesis_usage(tmp)))
        spec = loop.registry.get("ephemeral.my_special_net")
        check("ephemeral spec registered", spec is not None)
        check("ephemeral flag", spec is not None and spec.ephemeral)
        check("ephemeral persisted to disk",
              (tmp / "capabilities" / "ephemeral_specs.json").is_file())
        inv_obj = MethodInvocationV1.from_dict(inv)
        code, th = loop.compiler.render(inv_obj, PROFILE, MANIFEST)
        check("ephemeral renders",
              "def build_model" in code and "make_estimator" in code)
        check("ephemeral template hash", th.startswith("sha256:"), th)
        check("two LLM calls (propose + synthesis)", len(calls) == 2,
              str(len(calls)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_proposer_synthesis_budget_exhausted():
    old = os.environ.get("MAX_SYNTHESIS_ACTIONS")
    os.environ["MAX_SYNTHESIS_ACTIONS"] = "1"
    try:
        tmp = Path(tempfile.mkdtemp(prefix="v23_synth_"))
        try:
            loop = _make_loop(tmp)

            def fake_llm(prompt):
                if "CAPABILITY SYNTHESIS" in prompt:
                    return json.dumps({
                        "method_id": "adapter_a",
                        "source_code": _SYNTH_SOURCE,
                    })
                return json.dumps({
                    "capability_gap": "gap %s" % prompt[:10],
                    "hypothesis": "gap probe",
                })

            loop.planner.llm_call = fake_llm
            proposer = loop._agent_proposer(dict(GRANT), PROFILE)
            out1 = proposer(1, dict(GRANT), "")
            check("first gap mints one adapter",
                  (out1.get("invocation") or {}).get("method_id")
                  == "ephemeral.adapter_a",
                  str(out1.get("invocation"))[:200])
            out2 = proposer(2, dict(GRANT), "")
            inv2 = out2.get("invocation") or {}
            check("second gap exhausted -> no mint",
                  not str(inv2.get("method_id", "")).startswith("ephemeral."),
                  str(inv2)[:200])
            check("usage capped at 1",
                  load_synthesis_usage(tmp).get("used") == 1,
                  str(load_synthesis_usage(tmp)))
            expected = _first_compatible(loop)
            check("exhausted falls back deterministically",
                  inv2.get("method_id") == expected,
                  "got %r expected %r" % (inv2.get("method_id"), expected))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    finally:
        if old is None:
            os.environ.pop("MAX_SYNTHESIS_ACTIONS", None)
        else:
            os.environ["MAX_SYNTHESIS_ACTIONS"] = old


def test_ephemeral_reuse_changes_params_only():
    tmp = Path(tempfile.mkdtemp(prefix="v23_synth_"))
    try:
        registry = CapabilityRegistry(
            ephemeral_path=load_ephemeral_path(tmp))
        compiler = ProgramCompiler(registry)
        spec = MethodSpec(
            method_id="ephemeral.reuse",
            family="ephemeral",
            supported_modalities=["tabular"],
            supported_tasks=["classification"],
            metric_outputs={"accuracy": "class"},
            parameter_schema={
                "alpha": {"type": "float", "min": 0.0, "max": 1.0,
                          "default": 0.5}},
            preprocessing_options=["missing_value_impute"],
            validation_schemes=["stratified_kfold", "single_holdout"],
            renderer="ephemeral_sklearn", resource_model="ephemeral_sklearn_v1",
            gpu=False, ephemeral=True,
            source_code=_SYNTH_SOURCE,
            template_hash="sha256:" + hashlib.sha256(
                _SYNTH_SOURCE.encode("utf-8")).hexdigest())
        registry.register_ephemeral(spec)
        inv1 = compiler.normalize(MethodInvocationV1(
            method_id="ephemeral.reuse", params={"alpha": 0.1},
            hypothesis="h1"))
        inv2 = compiler.normalize(MethodInvocationV1(
            method_id="ephemeral.reuse", params={"alpha": 0.9},
            hypothesis="h2"))
        code1, th1 = compiler.render(inv1, PROFILE, MANIFEST)
        code2, th2 = compiler.render(inv2, PROFILE, MANIFEST)
        code1b, th1b = compiler.render(inv1, PROFILE, MANIFEST)
        check("same template hash on reuse", th1 == th2 == th1b)
        check("bit-for-bit replay same params", code1 == code1b)
        check("params differ between children",
              '"alpha": 0.1' in code1 and '"alpha": 0.9' in code2)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- Phase A
def test_materialize_spec_template_zero_llm():
    tmp = Path(tempfile.mkdtemp(prefix="v23_host_"))
    try:
        state = tmp / "state"
        bus = FileBus(state)
        registry = CapabilityRegistry(
            ephemeral_path=load_ephemeral_path(state))
        compiler = ProgramCompiler(registry)
        calls = []

        class _Impl:
            def implement(self, *a, **k):
                calls.append((a, k))
                return "print('legacy should not run')"

        host = HostSupervisorService(
            bus=bus,
            executor=types.SimpleNamespace(work_dir=str(tmp / "work")),
            bundler=None,
            evaluator=types.SimpleNamespace(
                metric_direction="higher_is_better"),
            promotion=None,
            implementer=_Impl(),
            compiler=compiler,
            registry=registry,
            competition="stub_comp",
            state_dir=state)
        inv = compiler.normalize(MethodInvocationV1(
            method_id="tabular.gbdt.histgb.v1",
            params={"learning_rate": 0.07, "max_iter": 400},
            preprocessing=["missing_value_native"],
            validation="stratified_kfold",
            hypothesis="v23 materialize test"))
        proposal = {"proposal_id": "proposal_v23_1",
                    "invocation": inv.to_dict()}
        plan = ResearchPlan(hypothesis="v23 materialize test",
                            method_detail={})
        spec = host._materialize_spec(proposal, dict(GRANT), PROFILE, plan)
        check("zero LLM implementer calls", len(calls) == 0, str(calls))
        check("invocation pinned",
              bool(spec.invocation)
              and spec.invocation.get("method_id")
              == "tabular.gbdt.histgb.v1")
        check("invocation hash", spec.invocation_hash.startswith("sha256:"),
              spec.invocation_hash)
        check("template hash", spec.template_hash.startswith("sha256:"),
              spec.template_hash)
        check("code hash pins code",
              spec.code_hash == "sha256:" + hashlib.sha256(
                  spec.code.encode("utf-8")).hexdigest())
        check("compiled marker",
              "[compiled]" in spec.code and "make_estimator" in spec.code)
        seal = spec.seal_record()
        check("seal pins hashes",
              seal["invocation_hash"] == spec.invocation_hash
              and seal["template_hash"] == spec.template_hash)
        check("spec json on bus",
              (bus.workspace / "code"
               / ("spec_" + spec.spec_id + ".json")).is_file())
        code2, th2 = compiler.render(
            MethodInvocationV1.from_dict(spec.invocation), PROFILE, MANIFEST)
        check("bit-for-bit replay",
              code2 == spec.code and th2 == spec.template_hash)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_materialize_spec_legacy_fallback():
    tmp = Path(tempfile.mkdtemp(prefix="v23_host_"))
    try:
        state = tmp / "state"
        bus = FileBus(state)
        registry = CapabilityRegistry(
            ephemeral_path=load_ephemeral_path(state))
        compiler = ProgramCompiler(registry)
        calls = []

        class _Impl:
            def implement(self, *a, **k):
                calls.append(1)
                return "print('legacy ok')"

        host = HostSupervisorService(
            bus=bus,
            executor=types.SimpleNamespace(work_dir=str(tmp / "work")),
            bundler=None,
            evaluator=types.SimpleNamespace(
                metric_direction="higher_is_better"),
            promotion=None,
            implementer=_Impl(),
            compiler=compiler,
            registry=registry,
            competition="stub_comp",
            state_dir=state)
        proposal = {"proposal_id": "proposal_v23_legacy"}
        plan = ResearchPlan(hypothesis="legacy fallback", method_detail={})
        spec = host._materialize_spec(proposal, dict(GRANT), PROFILE, plan)
        check("legacy implementer called once", len(calls) == 1, str(calls))
        check("legacy spec has no invocation", not spec.invocation)
        check("legacy spec has no template hash", spec.template_hash == "")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_patch_params_deterministic_halving():
    registry = CapabilityRegistry()
    compiler = ProgramCompiler(registry)
    inv = compiler.normalize(MethodInvocationV1(
        method_id="tabular.gbdt.histgb.v1", hypothesis="h"))
    check("defaults applied", inv.params.get("max_iter") == 300,
          str(inv.params))
    patched, note = compiler.patch_params(inv, "crash")
    check("patch found", patched is not None, note)
    check("halved max_iter", patched.params.get("max_iter") == 150,
          str(patched.params))
    check("patch round advanced",
          patched.resource_request.get("patch_round") == 1,
          str(patched.resource_request))
    ok, reason = compiler.validate(patched, PROFILE, MANIFEST)
    check("patched invocation still valid", ok, reason)
    p2, n2 = compiler.patch_params(patched, "crash")
    check("second patch halves learning_rate",
          p2 is not None and p2.params.get("learning_rate") == 0.025, n2)
    code, th = compiler.render(patched, PROFILE, MANIFEST)
    check("patched renders", "[compiled]" in code and th.startswith("sha256:"),
          th)


def test_registry_broken_capability_excluded():
    tmp = Path(tempfile.mkdtemp(prefix="v23_broken_"))
    try:
        registry = CapabilityRegistry(
            ephemeral_path=load_ephemeral_path(tmp))
        registry.set_broken("tabular.linear.logistic.v1",
                            "crashed at trial time")
        compat = registry.compatible("tabular", "classification", "accuracy")
        check("broken capability excluded",
              all(s.method_id != "tabular.linear.logistic.v1"
                  for s in compat),
              str([s.method_id for s in compat]))
        check("broken persisted",
              registry.get("tabular.linear.logistic.v1").broken)
        registry2 = CapabilityRegistry(
            ephemeral_path=load_ephemeral_path(tmp))
        check("builtin broken is per-instance (no cross-instance pollution)",
              not registry2.get("tabular.linear.logistic.v1").broken)
        ephem = MethodSpec(
            method_id="ephemeral.broken_probe",
            family="ephemeral",
            supported_modalities=["tabular"],
            supported_tasks=["classification"],
            metric_outputs={"accuracy": "class"},
            parameter_schema={},
            preprocessing_options=[],
            validation_schemes=["stratified_kfold"],
            source_code="def build_model():\n    return None\n",
            description="broken probe")
        registry2.register_ephemeral(ephem)
        registry2.set_broken("ephemeral.broken_probe", "crashed")
        registry3 = CapabilityRegistry(
            ephemeral_path=load_ephemeral_path(tmp))
        check("ephemeral broken survives reload",
              registry3.get("ephemeral.broken_probe") is not None
              and registry3.get("ephemeral.broken_probe").broken)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_no_competition_name_in_decision():
    """Power boundary: capability filtering never reads competition name."""
    tmp = Path(tempfile.mkdtemp(prefix="v23_bound_"))
    try:
        registry = CapabilityRegistry(
            ephemeral_path=load_ephemeral_path(tmp))
        for comp in ("aerial-cactus-identification",
                     "aptos2019-blindness-detection",
                     "dog-breed-identification",
                     "some-future-mlebench-comp"):
            specs = registry.compatible("tabular", "classification",
                                        "accuracy")
            check("compat for %s is name-independent" % comp,
                  len(specs) == len(registry.compatible(
                      "tabular", "classification", "accuracy")))
        prompt = registry.prompt_summary("tabular", "classification",
                                         "accuracy", max_chars=2400)
        check("prompt has no competition name",
              "aerial-cactus" not in prompt and "dog-breed" not in prompt
              and "aptos" not in prompt)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_image_template_missing_cache_falls_back():
    """v2.3.2: a missing/removed cache dir must NOT crash a compiled image
    trial - load_arrays falls back to raw decode and still returns arrays.
    Regression for the 2026-08-06 FileNotFoundError chain (rc=1)."""
    import numpy as np
    from PIL import Image
    tmp = Path(tempfile.mkdtemp(prefix="v23_imgcache_"))
    old_env = os.environ.get("V2_CACHE_DIRS")
    try:
        loop = _make_loop(tmp)
        inv = loop.compiler.normalize(MethodInvocationV1(
            method_id="image.embedding.timm.v1",
            params={"image_size": 64, "batch_size": 16, "max_iter": 100,
                    "model_name": "efficientnet_b0", "val_seed": 42},
            preprocessing=["cached_image_arrays"],
            validation="single_holdout"))
        img_profile = types.SimpleNamespace(
            modality="image", task_type="classification", metric_name="accuracy",
            metric_direction="maximize", train_rows=2, test_rows=1, n_classes=2,
            feature_dim=0, image_width=64, image_height=64)
        manifest = {"task_type": "classification", "metric_name": "accuracy",
                    "metric_direction": "maximize", "train_rows": 2}
        code, th = loop.compiler.render(inv, img_profile, manifest)
        check("image template renders", "load_arrays" in code, th[:80])
        check("image template guards cache load", "cache unreadable" in code,
              "missing guard in template")
        # run ONLY the cached-arrays machinery with a MISSING cache dir:
        # must fall back to raw decode instead of raising FileNotFoundError.
        start = code.index("_IMG_EXTS =")
        end = code.index('X_train = load_arrays("train")')
        body = code[start:end]
        data = tmp / "data"
        (data / "train").mkdir(parents=True)
        (data / "test").mkdir(parents=True)
        for name in ("a", "b"):
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
                data / "train" / (name + ".png"))
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
            data / "test" / ("c.png"))
        (data / "train.csv").write_text("id,label\na,0\nb,1\n",
                                        encoding="utf-8")
        (data / "test.csv").write_text("id\nc\n", encoding="utf-8")
        ns = {"os": os, "json": json, "np": np,
              "log": lambda msg: print("[compiled] %s" % msg, flush=True)}
        exec(compile(body, "<imgcache>", "exec"), ns)
        ns["id_col"] = "id"
        ns["IMAGE_SIZE"] = 64
        ns["TRAIN_CSV"] = str(data / "train.csv")
        ns["TEST_CSV"] = str(data / "test.csv")
        ns["TRAIN_IMAGES"] = str(data / "train")
        ns["TEST_IMAGES"] = str(data / "test")
        os.environ["V2_CACHE_DIRS"] = json.dumps(
            {"64": str(data / "data_cache" / "gone" / "64")})
        X = ns["load_arrays"]("train")
        check("missing cache falls back to raw decode (train)",
              X.shape == (2, 64, 64, 3), str(X.shape))
        Xt = ns["load_arrays"]("test")
        check("missing cache falls back to raw decode (test)",
              Xt.shape == (1, 64, 64, 3), str(Xt.shape))
        # restored cache dir is still used preferentially (cache hit)
        ok_dir = data / "data_cache" / "k" / "64"
        ok_dir.mkdir(parents=True)
        np.save(ok_dir / "train_X.npy",
                np.zeros((2, 64, 64, 3), dtype=np.uint8))
        (ok_dir / "train_ids.json").write_text(
            json.dumps(["a", "b"]), encoding="utf-8")
        os.environ["V2_CACHE_DIRS"] = json.dumps({"64": str(ok_dir)})
        X2 = ns["load_arrays"]("train")
        check("restored cache dir used (cache hit)",
              X2.shape == (2, 64, 64, 3), str(X2.shape))
    finally:
        if old_env is None:
            os.environ.pop("V2_CACHE_DIRS", None)
        else:
            os.environ["V2_CACHE_DIRS"] = old_env
        shutil.rmtree(tmp, ignore_errors=True)




def test_text_and_timeseries_capability_space():
    """v2.3.2: text modality and timeseries task_type must NEVER land in an
    empty capability space; text/timeseries renderers exist and bake the
    content evidence (text columns / lag column) into the code."""
    tmp = Path(tempfile.mkdtemp(prefix="v23_textcap_"))
    try:
        loop = _make_loop(tmp)
        txt = [x.method_id for x in loop.registry.compatible(
            "text", "classification", "accuracy")]
        check("text capability space non-empty",
              "text.embedding.tfidf.v1" in txt
              and "text.neural.mlp.v1" in txt, str(txt))
        ts = [x.method_id for x in loop.registry.compatible(
            "tabular", "timeseries", "rmse")]
        check("timeseries capability space non-empty",
              "timeseries.lag_histgb.v1" in ts
              and "tabular.gbdt.histgb.v1" in ts, str(ts))
        prof = types.SimpleNamespace(
            modality="text", task_type="classification", metric_name="accuracy",
            metric_direction="maximize", train_rows=120, test_rows=30,
            n_classes=2, feature_dim=1, image_width=0, image_height=0,
            text_columns=["comment_text"], time_column="")
        manifest = {"task_type": "classification", "metric_name": "accuracy",
                    "train_rows": 120}
        for mid, token in (("text.embedding.tfidf.v1", "TfidfVectorizer"),
                           ("text.neural.mlp.v1", "_DenseAdapter")):
            inv = loop.compiler.normalize(MethodInvocationV1(method_id=mid))
            code, th = loop.compiler.render(inv, prof, manifest)
            check("%s renders" % mid,
                  token in code and th.startswith("sha256:"), th[:60])
            check("text columns baked into code", '"comment_text"' in code, "")
        prof_ts = types.SimpleNamespace(
            modality="tabular", task_type="timeseries", metric_name="rmse",
            metric_direction="minimize", train_rows=120, test_rows=30,
            n_classes=0, feature_dim=1, image_width=0, image_height=0,
            text_columns=[], time_column="date")
        manifest_ts = {"task_type": "timeseries", "metric_name": "rmse",
                       "train_rows": 120}
        inv = loop.compiler.normalize(MethodInvocationV1(
            method_id="timeseries.lag_histgb.v1"))
        code, th = loop.compiler.render(inv, prof_ts, manifest_ts)
        check("timeseries lag renders",
              "time_holdout" in code and "LAG_COLUMN = 'date'" in code,
              th[:60])
        # tabular methods must still accept timeseries (compat fallback)
        inv = loop.compiler.normalize(MethodInvocationV1(
            method_id="tabular.linear.logistic.v1"))
        code, th = loop.compiler.render(inv, prof_ts, manifest_ts)
        check("tabular method accepts timeseries",
              "LogisticRegression" in code, th[:40])
        # v2.3.3: tabular modality with a stray text column renders tabular
        # mode (text features dropped; dense tabular models stay dense)
        prof_mixed = types.SimpleNamespace(
            modality="tabular", task_type="classification",
            metric_name="accuracy", metric_direction="maximize",
            train_rows=120, test_rows=30, n_classes=2, feature_dim=11,
            image_width=0, image_height=0,
            text_columns=["Name"], time_column="")
        inv = loop.compiler.normalize(MethodInvocationV1(
            method_id="tabular.linear.logistic.v1"))
        code, th = loop.compiler.render(inv, prof_mixed, manifest)
        check("mixed tabular renders tabular mode",
              "MODALITY = 'tabular'" in code
              and 'MODALITY == "text"' in code, th[:40])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_text_compiled_code_end_to_end():
    """v2.3.2: rendered text harness (TF-IDF + LogisticRegression) trains on
    a tiny CSV and writes oof.csv + submission.csv."""
    tmp = Path(tempfile.mkdtemp(prefix="v23_texte2e_"))
    try:
        loop = _make_loop(tmp)
        data = tmp / "data"
        data.mkdir()
        texts = ["great product works well", "terrible quality breaks fast",
                 "nice and cheap", "waste of money do not buy"]
        with io.open(data / "train.csv", "w", encoding="utf-8",
                     newline="") as fh:
            fh.write("id,comment_text,target\n")
            for i in range(60):
                fh.write("%d,%s,%d\n" % (i, texts[i % 4], i % 2))
        with io.open(data / "test.csv", "w", encoding="utf-8",
                     newline="") as fh:
            fh.write("id,comment_text\n")
            for i in range(8):
                fh.write("%d,%s\n" % (i, texts[i % 4]))
        with io.open(data / "sample_submission.csv", "w", encoding="utf-8",
                     newline="") as fh:
            fh.write("id,target\n")
            for i in range(8):
                fh.write("%d,0\n" % i)
        prof = types.SimpleNamespace(
            modality="text", task_type="classification", metric_name="accuracy",
            metric_direction="maximize", train_rows=60, test_rows=8,
            n_classes=2, feature_dim=1, image_width=0, image_height=0,
            text_columns=["comment_text"], time_column="")
        manifest = {"task_type": "classification", "metric_name": "accuracy",
                    "train_rows": 60}
        inv = loop.compiler.normalize(MethodInvocationV1(
            method_id="text.embedding.tfidf.v1",
            params={"max_features": 1000, "max_iter": 200, "val_seed": 7}))
        code, th = loop.compiler.render(inv, prof, manifest)
        script = tmp / "trial.py"
        script.write_text(code, encoding="utf-8")
        env = dict(os.environ)
        env.update({
            "TRAIN_CSV": str(data / "train.csv"),
            "TEST_CSV": str(data / "test.csv"),
            "SAMPLE_SUBMISSION": str(data / "sample_submission.csv"),
            "TARGET_COLUMN": "target",
            "TASK_TYPE": "classification",
        })
        r = subprocess.run([sys.executable, str(script)], capture_output=True,
                           text=True, timeout=300, env=env, cwd=str(tmp))
        detail = ((r.stdout or "")[-600:] + (r.stderr or "")[-600:])
        check("text trial exits 0", r.returncode == 0, detail)
        oof = tmp / "oof.csv"
        sub = tmp / "submission.csv"
        check("text oof written",
              oof.is_file() and oof.read_text(encoding="utf-8").strip()
              .splitlines()[0] == "true,pred", "")
        check("text submission written",
              sub.is_file() and len(sub.read_text(encoding="utf-8").strip()
                                    .splitlines()) == 9, "")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_timeseries_compiled_code_end_to_end():
    """v2.3.2: lag-feature renderer sorts by the detected date column,
    uses time-ordered holdout and writes oof.csv + submission.csv."""
    import datetime as _dt
    tmp = Path(tempfile.mkdtemp(prefix="v23_tse2e_"))
    try:
        loop = _make_loop(tmp)
        data = tmp / "data"
        data.mkdir()
        base = _dt.date(2020, 1, 1)
        with io.open(data / "train.csv", "w", encoding="utf-8",
                     newline="") as fh:
            # v2.3.3: a stray free-text column must be excluded from the
            # lag feature loop (all-NaN lags would drop every row).
            fh.write("date,sales,note\n")
            for i in range(80):
                d = base + _dt.timedelta(days=i)
                fh.write("%s,%.1f,note text %d\n"
                         % (d.isoformat(), 100.0 + i + (i % 4) * 2.0, i))
        with io.open(data / "test.csv", "w", encoding="utf-8",
                     newline="") as fh:
            fh.write("date,sales,note\n")
            for i in range(5):
                d = base + _dt.timedelta(days=80 + i)
                fh.write("%s,0.0,note text 0\n" % d.isoformat())
        with io.open(data / "sample_submission.csv", "w", encoding="utf-8",
                     newline="") as fh:
            fh.write("id,sales\n")
            for i in range(5):
                fh.write("%d,0.0\n" % i)
        prof = types.SimpleNamespace(
            modality="tabular", task_type="timeseries", metric_name="rmse",
            metric_direction="minimize", train_rows=80, test_rows=5,
            n_classes=0, feature_dim=1, image_width=0, image_height=0,
            text_columns=["note"], time_column="date")
        manifest = {"task_type": "timeseries", "metric_name": "rmse",
                    "train_rows": 80}
        inv = loop.compiler.normalize(MethodInvocationV1(
            method_id="timeseries.lag_histgb.v1",
            params={"max_lag": 3, "rolling_window": 3, "max_iter": 100,
                    "val_seed": 7}))
        code, th = loop.compiler.render(inv, prof, manifest)
        script = tmp / "trial.py"
        script.write_text(code, encoding="utf-8")
        env = dict(os.environ)
        env.update({
            "TRAIN_CSV": str(data / "train.csv"),
            "TEST_CSV": str(data / "test.csv"),
            "SAMPLE_SUBMISSION": str(data / "sample_submission.csv"),
            "TARGET_COLUMN": "sales",
            "TASK_TYPE": "timeseries",
        })
        r = subprocess.run([sys.executable, str(script)], capture_output=True,
                           text=True, timeout=300, env=env, cwd=str(tmp))
        detail = ((r.stdout or "")[-600:] + (r.stderr or "")[-600:])
        check("timeseries trial exits 0", r.returncode == 0, detail)
        check("timeseries oof written", (tmp / "oof.csv").is_file(), "")
        check("timeseries submission written",
              (tmp / "submission.csv").is_file()
              and len((tmp / "submission.csv").read_text(encoding="utf-8")
                      .strip().splitlines()) == 6, "")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_proposer_valid_invocation()
    test_proposer_preprocessing_alias_normalized()
    test_proposer_leave_oof_alias_normalized()
    test_proposer_imputation_alias_normalized()
    test_proposer_partial_fallback_keeps_method()
    test_proposer_unknown_method_fallback()
    test_proposer_llm_error_fallback()
    test_proposer_schema_reject_fallback()
    test_proposer_capability_gap_synthesis()
    test_proposer_synthesis_budget_exhausted()
    test_ephemeral_reuse_changes_params_only()
    test_materialize_spec_template_zero_llm()
    test_materialize_spec_legacy_fallback()
    test_patch_params_deterministic_halving()
    test_registry_broken_capability_excluded()
    test_no_competition_name_in_decision()
    test_image_template_missing_cache_falls_back()
    test_text_and_timeseries_capability_space()
    test_text_compiled_code_end_to_end()
    test_timeseries_compiled_code_end_to_end()
    if FAILURES:
        print("FAILURES=%d: %s" % (len(FAILURES), ", ".join(FAILURES)))
        sys.exit(1)
    print("ALL_V23_TESTS=PASS")