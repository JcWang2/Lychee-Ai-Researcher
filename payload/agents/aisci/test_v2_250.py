# -*- coding: utf-8 -*-
"""v2.5.0 offline tests: declarative method architecture (zero if/else routing).

Frozen v2.5 contracts verified here:
  1) NO competition name appears in any logic module (metrics_registry.py is
     the only official data table; test files are exempt).
  2) NO if/elif chain selects a renderer/template by name: the compiler must
     dispatch through _TEMPLATE_REGISTRY (one entry per renderer).
  3) Method selection is declarative: MethodSelector filters by metadata and
     ranks by cost table + experience prior (_COST_MODELS); it never decides.
     The final research decision stays with the planner/Analyzer (the prompt
     says so and the selector API returns ranked candidates, not a pick).
  4) Metric dispatch is declarative: evaluator uses _COMPUTE_HANDLERS /
     _PROBABILITY_HANDLERS; stage_controller uses _RANDOM_BASELINE_KIND.
  5) Resource rules are tables: portfolio derives budgets from
     modality-resource tables, never modality branches.
  6) Registry completeness: every capability renderer has a template entry
     and renders deterministically.
  7) Selector generalization: any synthetic contract (modality x metric x
     scale) gets non-empty compatible candidates; results depend only on the
     contract, never on a task identity.

Run: python test_v2_250.py   (from the aisci payload dir)
"""
import ast
import hashlib
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from capability_registry import CapabilityRegistry  # noqa: E402
from hera.planner import Planner  # noqa: E402
from hera.portfolio import ResourceProfiler  # noqa: E402
from method_selector import (DatasetContract, ExperienceTable,  # noqa: E402
                             MethodSelector)
from pact.evaluator import (_COMPUTE_HANDLERS, _PROBABILITY_HANDLERS,  # noqa: E402
                            TrustedEvaluator)
from program_compiler import ProgramCompiler, _TEMPLATE_REGISTRY  # noqa: E402
from stage_controller import _RANDOM_BASELINE_KIND  # noqa: E402
from v2_contracts import (AnalysisProfile, MethodInvocationV1,  # noqa: E402
                          ResearchPlan)

import metrics_registry as mr  # noqa: E402

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


# ------------------------------------------------------------ 1) no comp names
def test_no_competition_names():
    comps = set((getattr(mr, "COMPETITION_METRICS", {}) or {}).keys())
    bad = []
    for path in HERE.rglob("*.py"):
        rel = str(path.relative_to(HERE))
        if any(part.startswith(".v2_backup") or part == "__pycache__"
               for part in rel.split("/")):
            continue  # install backups / caches are not logic modules
        if path.name == "metrics_registry.py" or path.name.startswith("test_"):
            continue
        src = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                v = node.value.strip().lower()
                if v in comps:
                    bad.append("%s:%d %r" % (rel, node.lineno, node.value))
    check("no competition names in logic code", not bad, "; ".join(bad[:5]))


# ------------------------------------------------------------ 2) no renderer if/elif
def test_no_renderer_dispatch_chain():
    pc_src = (HERE / "program_compiler.py").read_text(encoding="utf-8")
    bad = [ln for ln in pc_src.splitlines()
           if ("spec.renderer ==" in ln or "renderer ==" in ln)]
    check("compiler has no renderer == branches", not bad, "; ".join(bad[:3]))
    check("compiler uses _TEMPLATE_REGISTRY",
          "_TEMPLATE_REGISTRY.get(spec.renderer)" in pc_src)


# ------------------------------------------------------------ 3) selector declarative
def test_selector_declarative():
    sel = MethodSelector()
    contracts = [
        DatasetContract(modality="image", task_type="classification",
                        metric_family="qwk", n_rows=3295, n_classes=5,
                        gpu_available=True, budget_seconds=1800,
                        has_pretrained=True, image_cache=True),
        DatasetContract(modality="text", task_type="classification",
                        metric_family="logloss", n_rows=17621,
                        n_classes=3, budget_seconds=900),
        DatasetContract(modality="tabular", task_type="regression",
                        metric_family="rmse", n_rows=200000,
                        budget_seconds=1800),
        DatasetContract(modality="audio", task_type="classification",
                        metric_family="mean_auc_multilabel", n_rows=8000,
                        budget_seconds=900),
        DatasetContract(modality="image_pixel", task_type="regression",
                        metric_family="rmse", n_rows=200000,
                        gpu_available=True, budget_seconds=1800),
        DatasetContract(modality="image_mask", task_type="segmentation",
                        metric_family="dice", n_rows=5000,
                        gpu_available=True, budget_seconds=1800),
        DatasetContract(modality="image_detection", task_type="detection",
                        metric_family="map_at_k", n_rows=3000,
                        gpu_available=True, budget_seconds=1800),
        DatasetContract(modality="tabular", task_type="timeseries",
                        metric_family="rmse", n_rows=60000,
                        budget_seconds=1200),
        DatasetContract(modality="text", task_type="classification",
                        metric_family="mean_auc_multilabel", n_rows=9000,
                        text_columns=2, budget_seconds=900),
    ]
    for c in contracts:
        cands = sel.candidates(c)
        check("selector candidates non-empty for %s/%s/%s"
              % (c.modality, c.task_type, c.metric_family),
              len(cands) > 0, "got %d" % len(cands))
        for m in cands[:3]:
            spec = sel.registry.get(m.method_id)
            ok = spec is not None and sel.compatible(spec, c)
            check("selector candidate %s compatible with %s"
                  % (m.method_id, c.modality), ok, m.method_id)
    # contract-only: same contract twice -> identical ranking
    c = contracts[0]
    a = [(m.method_id, m.score) for m in sel.candidates(c)]
    b = [(m.method_id, m.score) for m in sel.candidates(c)]
    check("selector is contract-only (deterministic)", a == b, str(a))
    # selector never decides: multiple candidates with a score, no "pick"
    check("selector returns ranked candidates (not a decision)",
          len(a) >= 2, str(a))
    # experience prior influences ranking but never excludes
    exp = ExperienceTable()
    exp.record(c, a[0][0], lift=0.2, cost_ratio=0.5)
    sel2 = MethodSelector(experience=exp)
    b2 = [(m.method_id, m.score) for m in sel2.candidates(c)]
    a_scores = dict(a)
    b_scores = dict(b2)
    check("experience prior changes scores (data-driven)",
          abs(b_scores.get(a[0][0], 0.0) - a_scores[a[0][0]]) > 1e-9,
          str((a_scores.get(a[0][0]), b_scores.get(a[0][0]))))
    check("experience table round-trips",
          exp.rows and exp.prior(c, a[0][0])[2] == 1, str(exp.rows)[:80])


# ------------------------------------------------------------ 4) evaluator declarative
def test_evaluator_declarative():
    ev_src = (HERE / "pact" / "evaluator.py").read_text(encoding="utf-8")
    check("evaluator uses _COMPUTE_HANDLERS",
          "_COMPUTE_HANDLERS.get(name)" in ev_src)
    check("evaluator uses _PROBABILITY_HANDLERS",
          "_PROBABILITY_HANDLERS.get(self.metric_name)" in ev_src)
    for k in ("logloss", "weighted_logloss", "kl_div",
              "mean_auc_multilabel", "binary_logloss", "map_at_k",
              "label_ranking_ap"):
        check("evaluator handler for %s" % k, k in _COMPUTE_HANDLERS, k)
    for k in ("logloss", "weighted_logloss", "kl_div",
              "mean_auc_multilabel"):
        check("probability handler for %s" % k, k in _PROBABILITY_HANDLERS, k)
    # metric dispatch still works end-to-end
    ev = TrustedEvaluator(metric_name="logloss")
    r = ev.evaluate(None, returncode=1)
    check("evaluator rc failure path unchanged", r.metric is None,
          str(r.evidence))


# ------------------------------------------------------------ 5) stage controller declarative
def test_stage_declarative():
    sc_src = (HERE / "stage_controller.py").read_text(encoding="utf-8")
    check("stage_controller uses _RANDOM_BASELINE_KIND",
          "kind = _RANDOM_BASELINE_KIND.get(metric)" in sc_src)
    for k in ("auc", "qwk", "logloss", "rmse", "accuracy"):
        check("random baseline kind for %s" % k, k in _RANDOM_BASELINE_KIND, k)


# ------------------------------------------------------------ 6) registry completeness
def _profile_for_spec(spec):
    """Minimal AnalysisProfile/manifest matching a capability's metadata."""
    modality = (spec.supported_modalities or ["tabular"])[0]
    task_type = (spec.supported_tasks or ["classification"])[0]
    metric_name = next(iter(spec.metric_outputs or {"logloss": "proba"}))
    prof = AnalysisProfile(
        competition="some-future-mlebench-comp",
        modality=modality, task_type=task_type, metric_name=metric_name,
        metric_direction="min", train_rows=2000, n_classes=5,
        text_columns=["text"] if modality == "text" else [],
        time_column="datetime" if task_type == "timeseries" else "")
    return prof, {"metric_name": metric_name, "metric_direction": "min"}


def test_registry_completeness():
    reg = CapabilityRegistry()
    pc = ProgramCompiler(reg)
    missing = []
    used_renderers = set()
    for spec in reg.all():
        if getattr(spec, "ephemeral", False):
            continue
        used_renderers.add(spec.renderer)
        if spec.renderer not in _TEMPLATE_REGISTRY:
            missing.append(spec.method_id + "->" + spec.renderer)
    check("all capability renderers registered", not missing,
          "; ".join(missing[:5]))
    unused = [rid for rid in _TEMPLATE_REGISTRY if rid not in used_renderers]
    check("every template renderer is used by a capability", not unused,
          "; ".join(unused[:5]))
    # every renderer renders deterministically through its real method spec
    rendered = set()
    for spec in reg.all():
        if getattr(spec, "ephemeral", False) or spec.renderer in rendered:
            continue
        rendered.add(spec.renderer)
        try:
            prof, manifest = _profile_for_spec(spec)
            val = (spec.validation_schemes or ["single_holdout"])[0]
            inv = MethodInvocationV1(method_id=spec.method_id,
                                     hypothesis="registry-completeness",
                                     params={}, validation=val)
            code1, th1 = pc.render(inv, profile=prof, manifest=manifest)
            code2, th2 = pc.render(inv, profile=prof, manifest=manifest)
            ok = (len(code1) > 500 and th1 == th2
                  and th1.startswith("sha256:"))
        except Exception as exc:  # noqa: BLE001
            ok = False
            th1 = repr(exc)
        check("renderer %s renders deterministically" % spec.renderer,
              ok, str(th1)[:120])
    # probability templates carry the row-norm + 9-decimal contract
    prof, manifest = _profile_for_spec(reg.get("image.finetune.timm.v2"))
    inv_img = MethodInvocationV1(method_id="image.finetune.timm.v2",
                                 hypothesis="demo_img",
                                 params={"val_seed": 42},
                                 validation="single_holdout")
    code_img, _ = pc.render(inv_img, profile=prof, manifest=manifest)
    check("finetune v2 has _norm_proba_row", "_norm_proba_row" in code_img)
    check("finetune v2 has %.9f", '"%.9f"' in code_img)
    prof_tab, manifest_tab = _profile_for_spec(
        reg.get("tabular.gbdt.histgb.v1"))
    inv_tab = MethodInvocationV1(method_id="tabular.gbdt.histgb.v1",
                                 hypothesis="demo_tab",
                                 params={"val_seed": 42})
    code_tab, _ = pc.render(inv_tab, profile=prof_tab,
                            manifest=manifest_tab)
    check("tabular harness has _norm_proba_row",
          "_norm_proba_row" in code_tab)


# ------------------------------------------------------------ 7) planner prior injection
def test_planner_prior_injection():
    from v2_contracts import AnalysisProfile
    prof = AnalysisProfile(
        competition="some-future-mlebench-comp",
        task_type="classification", modality="image",
        metric_name="qwk", metric_label="quadratic weighted kappa",
        train_rows=3295, n_classes=5,
        data_notes="image rows=3295/367 classes=5",
        text_columns=[], time_column="")
    planner = Planner(llm_call_fn=lambda p: "{}")
    block = planner.prior_block(prof, {"max_budget_seconds": 1800,
                                       "gpu_memory_mb": 40960,
                                       "cached_weights": ["timm/x"]})
    check("planner prior block non-empty", len(block) > 50, block[:80])
    check("planner prior lists a method id", "image." in block, block[:120])
    prompt = planner.build_plan_prompt(
        prof, evidence="none", round_num=1, elapsed=0, total_budget=86400,
        resource={"max_budget_seconds": 1800, "gpu_memory_mb": 40960,
                  "cached_weights": ["timm/x"]})
    check("prompt contains PRIOR KNOWLEDGE", "PRIOR KNOWLEDGE" in prompt)
    check("prompt keeps decision with the planner",
          "final research decision is always yours" in prompt)


# ------------------------------------------------------------ 8) portfolio tables
def test_portfolio_tables():
    pf_src = (HERE / "hera" / "portfolio.py").read_text(encoding="utf-8")
    for const in ("_MODALITY_BASE_RESOURCE", "_BATCH_GPU_TABLE",
                  "_EPOCHS_BONUS_FIXED", "_EPOCHS_BONUS_GPU",
                  "_SEED_MODEL_BY_TASK"):
        check("portfolio has %s" % const, const in pf_src, const)
    prof_src = (HERE / "pact" / "deterministic.py").read_text(
        encoding="utf-8")
    check("deterministic uses _BINARY_POS_FREQ_METRICS",
          "_BINARY_POS_FREQ_METRICS" in prof_src)


if __name__ == "__main__":
    test_no_competition_names()
    test_no_renderer_dispatch_chain()
    test_selector_declarative()
    test_evaluator_declarative()
    test_stage_declarative()
    test_registry_completeness()
    test_planner_prior_injection()
    test_portfolio_tables()
    print("RESULT=%s ok=%d fail=%d" % ("PASS" if FAIL == 0 else "FAIL",
                                        PASS, FAIL))
    sys.exit(0 if FAIL == 0 else 1)
