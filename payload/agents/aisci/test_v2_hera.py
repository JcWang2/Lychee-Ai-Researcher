# -*- coding: utf-8 -*-
"""test_v2_hera.py - HERA analyzer/planner/interpreter tests (stub LLM)."""
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from hera import Analyzer, Interpreter, Planner  # noqa: E402
from v2_contracts import TrialReceipt  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("[OK] " + name)
    else:
        print("[FAIL] " + name + (" | " + detail if detail else ""))
        FAILURES.append(name)


def _make_data_dir(tmp):
    data = tmp / "data"
    data.mkdir()
    (data / "train.csv").write_text(
        "age,fare,target\n30,10.5,1\n40,20.0,0\n25,5.5,1\n", encoding="utf-8")
    (data / "test.csv").write_text("age,fare\n31,11.0\n", encoding="utf-8")
    return data


def _make_mlebench_data_dir(tmp):
    data = tmp / "mle_task"
    public = data / "prepared" / "public"
    private = data / "prepared" / "private"
    public.mkdir(parents=True)
    private.mkdir(parents=True)
    (public / "train.csv").write_text(
        "image,label\nimg_1.jpg,0\nimg_2.jpg,1\nimg_3.jpg,1\n",
        encoding="utf-8")
    (public / "sample_submission.csv").write_text(
        "image,label\nimg_1.jpg,0\nimg_2.jpg,0\n",
        encoding="utf-8")
    (private / "test.csv").write_text(
        "image\nimg_1.jpg\nimg_2.jpg\n",
        encoding="utf-8")
    (public / "train").mkdir()
    (public / "test").mkdir()
    return data, public / "sample_submission.csv"


def test_analyzer_classification():
    tmp = Path(tempfile.mkdtemp(prefix="v2_hera_test_"))
    try:
        data = _make_data_dir(tmp)
        profile = Analyzer(data, "Predict whether a passenger survived").profile("demo")
        check("task type classification", profile.task_type == "classification",
              profile.task_type)
        check("train rows", profile.train_rows == 3, str(profile.train_rows))
        check("test rows", profile.test_rows == 1, str(profile.test_rows))
        check("target column found", profile.target_column == "target",
              profile.target_column)
        check("columns listed", "age" in profile.feature_columns)
        check("data notes non-empty", bool(profile.data_notes))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_analyzer_mlebench_prepared_layout():
    tmp = Path(tempfile.mkdtemp(prefix="v2_hera_mlebench_test_"))
    try:
        data, sample = _make_mlebench_data_dir(tmp)
        profile = Analyzer(data, "image classification",
                           sample_path=str(sample)).profile("mle_demo")
        check("mlebench task type classification",
              profile.task_type == "classification", profile.task_type)
        check("mlebench train rows", profile.train_rows == 3,
              str(profile.train_rows))
        check("mlebench test rows", profile.test_rows == 2,
              str(profile.test_rows))
        check("mlebench target column", profile.target_column == "label",
              profile.target_column)
        check("mlebench layout note", "layout=mlebench_prepared" in profile.data_notes,
              profile.data_notes)
        check("mlebench sample note", str(sample) in profile.data_notes,
              profile.data_notes)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_planner_stub_llm():
    tmp = Path(tempfile.mkdtemp(prefix="v2_hera_test_"))
    try:
        data = _make_data_dir(tmp)
        profile = Analyzer(data, "regression task: predict price").profile("demo")

        def stub(prompt):
            if "Return a JSON plan" in prompt:
                return json.dumps({
                    "hypothesis": "Stub hypothesis",
                    "approach_type": "exploit",
                    "expected_improvement": "small",
                    "risk": "Medium",
                    "method_detail": {"model": "xgboost", "features": "all"},
                    "max_budget_seconds": 60,
                })
            return "{}"

        planner = Planner(llm_call_fn=stub)
        plan = planner.plan(profile, evidence="", round_num=1, elapsed=1,
                            total_budget=100)
        check("hypothesis parsed", plan.hypothesis == "Stub hypothesis", plan.hypothesis)
        check("approach parsed", plan.approach_type == "exploit", plan.approach_type)
        check("method parsed", plan.method_detail.get("model") == "xgboost")
        check("no code in plan (code belongs to PACT)", not hasattr(plan, "code") or plan.code == "")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_planner_parses_intent_and_children():
    """v2.2.1 regression: the planner must NOT drop research_intent/children
    (the acceptance probe: LLM returns final_training/1 -> plan carries both)."""
    tmp = Path(tempfile.mkdtemp(prefix="v2_hera_intent_"))
    try:
        data = _make_data_dir(tmp)
        profile = Analyzer(data, "demo").profile("demo")

        def stub(prompt):
            if "Return a JSON plan" in prompt:
                return json.dumps({
                    "hypothesis": "H",
                    "approach_type": "explore",
                    "expected_improvement": "x",
                    "risk": "Low",
                    "research_intent": "final_training",
                    "children": 1,
                    "method_detail": {"model": "random_forest"},
                    "max_budget_seconds": 600,
                })
            return "{}"

        plan = Planner(llm_call_fn=stub).plan(
            profile, evidence="", round_num=1, elapsed=1, total_budget=100)
        check("planner intent parsed",
              plan.research_intent == "final_training", plan.research_intent)
        check("planner children parsed",
              plan.method_detail.get("children") == 1,
              str(plan.method_detail.get("children")))

        # invalid intent -> empty (stage validates later)
        def stub_bad(prompt):
            return json.dumps({"research_intent": "bogus",
                               "method_detail": {}})
        plan2 = Planner(llm_call_fn=stub_bad).plan(
            profile, evidence="", round_num=1, elapsed=1, total_budget=100)
        check("planner invalid intent -> empty",
              plan2.research_intent == "", plan2.research_intent)
        check("planner children not recorded when missing",
              plan2.method_detail.get("children") is None,
              str(plan2.method_detail))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_analyzer_regression_target_stats():
    """v2.2.1: _target_stats must measure mean/std/min/max for numeric
    targets so the regression reference line is not stuck at 1.0."""
    tmp = Path(tempfile.mkdtemp(prefix="v2_hera_regr_"))
    try:
        data = tmp / "data"
        data.mkdir()
        (data / "train.csv").write_text(
            "x,y\n1,1.0\n2,2.0\n3,3.0\n4,4.0\n", encoding="utf-8")
        (data / "test.csv").write_text("x\n5\n", encoding="utf-8")
        profile = Analyzer(data, "regression").profile("demo")
        stats = profile.target_stats or {}
        check("regression mean measured",
              abs(float(stats.get("mean") or 0) - 2.5) < 1e-6, str(stats))
        check("regression std measured",
              abs(float(stats.get("std") or 0) - 1.118034) < 1e-3, str(stats))
        check("regression min/max measured",
              float(stats.get("min") or -1) == 1.0
              and float(stats.get("max") or -1) == 4.0, str(stats))
        from stage_controller import random_baseline
        from v2_contracts import AnalysisProfile
        rmse_profile = AnalysisProfile(
            competition="x", task_type="regression",
            metric_name="rmse", metric_direction="lower_is_better",
            metric_alignment="exact", metric_label="rmse",
            n_classes=0, target_stats=dict(profile.target_stats))
        base = random_baseline(rmse_profile)
        check("regression baseline uses measured scale",
              base is not None and abs(base - 1.118034) < 1e-3,
              str(base))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_planner_no_llm_fallback():
    tmp = Path(tempfile.mkdtemp(prefix="v2_hera_test_"))
    try:
        data = _make_data_dir(tmp)
        profile = Analyzer(data, "demo").profile("demo")
        planner = Planner(llm_call_fn=lambda p: "{}")
        plan = planner.plan(profile, evidence="", round_num=1, elapsed=1,
                            total_budget=100)
        check("fallback plan present", "Fallback" in plan.hypothesis)
        check("fallback approach type", plan.approach_type == "explore")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resource_profiles():
    """v2.2 resource profiles: generic, competition-name-free, persist."""
    from hera.portfolio import MethodPortfolio, ResourceProfiler, \
        resource_profile_for
    from v2_contracts import AnalysisProfile
    tmp = Path(tempfile.mkdtemp(prefix="v2_res_test_"))
    try:
        data = _make_data_dir(tmp)
        profile = Analyzer(data, "demo").profile("demo")
        generic = resource_profile_for("unknown-comp", "classification")
        check("resource generic budget",
              isinstance(generic.get("max_budget_seconds"), int)
              and 300 <= generic["max_budget_seconds"] <= 7200,
              str(generic))
        check("resource generic folds", 1 <= generic.get("max_folds", 0) <= 5,
              str(generic))
        # Competition names must NOT change resource derivation.
        named = resource_profile_for("dog-breed-identification",
                                     "classification")
        check("resource ignores competition name",
              named["max_budget_seconds"] == generic["max_budget_seconds"]
              and named["max_folds"] == generic["max_folds"],
              str(named))
        # Profile signals DO drive the derivation (image modality, rows).
        img = AnalysisProfile(competition="anything",
                              task_type="classification",
                              modality="image", train_rows=20000,
                              image_width=192, image_height=192,
                              image_channels=3, n_classes=120)
        res = ResourceProfiler().derive(img)
        check("image profile budget scales with rows",
              res["max_budget_seconds"] >= 1200, str(res))
        check("image profile size cap sane",
              res["image_size_max"] is not None
              and 96 <= res["image_size_max"] <= 384,
              str(res.get("image_size_max")))
        check("image profile folds capped", 1 <= res["max_folds"] <= 5,
              str(res["max_folds"]))
        check("resource derived_from has no competition",
              "competition" not in str(res.get("derived_from") or {}),
              str(res.get("derived_from")))
        port_path = tmp / "portfolio.json"
        port = MethodPortfolio.load_or_default(profile, port_path)
        port.resource_profile = dict(res)
        port.save()
        again = MethodPortfolio.load_or_default(profile, port_path)
        check("resource persists",
              again.resource_profile.get("max_folds") == res["max_folds"]
              and again.resource_profile.get("max_budget_seconds")
              == res["max_budget_seconds"],
              str(again.resource_profile))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_planner_resource_budget():
    """Planner clamps max_budget_seconds to the competition resource cap."""
    from hera.portfolio import resource_profile_for
    tmp = Path(tempfile.mkdtemp(prefix="v2_plan_res_test_"))
    try:
        data = _make_data_dir(tmp)
        profile = Analyzer(data, "demo").profile("demo")
        res = resource_profile_for("aerial-cactus-identification",
                                   "classification")
        stub = lambda p: json.dumps({  # noqa: E731
            "hypothesis": "H", "approach_type": "exploit",
            "method_detail": {"model": "xgb"},
            "max_budget_seconds": 9999})
        plan = Planner(llm_call_fn=stub).plan(
            profile, evidence="", round_num=1, elapsed=1, total_budget=86400,
            resource=res)
        check("planner clamps to resource budget",
              plan.max_budget_seconds <= res["max_budget_seconds"],
              str(plan.max_budget_seconds))
        check("plan carries resource profile",
              plan.method_detail.get("resource_profile", {}).get("max_folds")
              == res["max_folds"],
              str(plan.method_detail)[:200])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_memory_strategy_pool():
    from hera.memory import ScientificMemory
    from v2_contracts import ResearchPlan, TrialReceipt
    tmp = Path(tempfile.mkdtemp(prefix="v2_mem_test_"))
    try:
        mem = ScientificMemory(tmp)
        plan = ResearchPlan(round_num=1, hypothesis="H", approach_type="exploit",
                            method_detail={"model": "xgb"})
        receipt = TrialReceipt(competition="demo", round_num=1,
                               verdict="success", metric=0.9)
        mem.update(plan, receipt, task_description="demo task", best_before=0.8)
        check("strategy pool written on success", (tmp / "strategy_pool.json").exists())
        check("evidence nodes written", (tmp / "evidence_nodes.json").exists())
        check("causal edges written", (tmp / "causal_edges.json").exists())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_interpreter():
    interp = Interpreter(llm_call_fn=lambda p: "{}")
    receipt = TrialReceipt(round_num=1, verdict="success", metric=0.85)
    result = interp.interpret(receipt, best_before=0.80, stagnation_count=0,
                              max_rounds=10, elapsed=10, total_budget=100)
    check("delta computed", result.delta is not None and abs(result.delta - 0.05) < 1e-6, str(result.delta))
    check("continue default", result.stop_decision == "continue", result.stop_decision)
    stuck = interp.interpret(receipt, best_before=0.85, stagnation_count=6,
                             max_rounds=10, elapsed=10, total_budget=100)
    check("hard stop at stagnation 6", stuck.stop_decision == "stop", stuck.stop_decision)
    end = interp.interpret(receipt, best_before=0.80, stagnation_count=0,
                           max_rounds=1, elapsed=10, total_budget=100)
    check("stop at max rounds", end.stop_decision == "stop", end.stop_decision)




def test_portfolio_hera_writes_branches():
    from hera.portfolio import MethodPortfolio
    from hera.prioritization import Prioritizer
    from v2_contracts import ResearchPlan
    tmp = Path(tempfile.mkdtemp(prefix="v2_portfolio_test_"))
    try:
        data = _make_data_dir(tmp)
        profile = Analyzer(data, "demo").profile("demo")
        port_path = tmp / "portfolio.json"

        # seed portfolio: deterministic starter branches, persisted on save
        port = MethodPortfolio.load_or_default(profile, port_path)
        check("portfolio seeded", "baseline" in port.branch_ids()
              and "feature_engineering" in port.branch_ids(),
              ",".join(port.branch_ids()))
        port.save()
        check("portfolio persisted", port_path.is_file())

        # HERA writes a new branch; invalid/duplicate branches are rejected
        check("portfolio add branch",
              port.add_branch({
                  "branch_id": "torch_cnn",
                  "model_family": "torch_cnn",
                  "description": "CNN with augmentation",
                  "allowed_mutation_axes": ["architecture", "hyperparameter"],
                  "defaults": {"model": "torch_cnn", "epochs": 5},
              }) == "")
        check("portfolio rejects duplicate",
              port.add_branch({"branch_id": "torch_cnn",
                               "allowed_mutation_axes": ["model"]}) != "")
        check("portfolio rejects bad axis",
              port.add_branch({"branch_id": "bad_axis",
                               "allowed_mutation_axes": ["magic"]}) != "")
        port.save()

        # reload merges persisted + discovered branches
        reloaded = MethodPortfolio.load_or_default(profile, port_path)
        check("portfolio reload keeps discovered",
              "torch_cnn" in reloaded.branch_ids(),
              ",".join(reloaded.branch_ids()))

        # prioritizer: LLM picks a branch and HERA writes a new one
        plan = ResearchPlan(round_num=1, hypothesis="H", approach_type="explore",
                            method_detail={"features": "all"})

        def stub(prompt):
            return json.dumps({
                "selected_branch_id": "torch_cnn",
                "mutation_axis": "architecture",
                "reason": "test",
                "new_branches": [{
                    "branch_id": "ensemble_stack",
                    "model_family": "ensemble",
                    "description": "Stacked ensemble",
                    "allowed_mutation_axes": ["ensemble", "hyperparameter"],
                    "defaults": {"model": "stack", "cv": 3},
                }],
            })

        prio = Prioritizer(llm_call_fn=stub)
        ticket = prio.prioritize(profile, reloaded, plan, trial_budget=3)
        check("prioritizer selects branch",
              ticket.selected_branch_id == "torch_cnn", ticket.selected_branch_id)
        check("prioritizer validates axis",
              ticket.mutation_axis == "architecture", ticket.mutation_axis)
        check("prioritizer writes new branch",
              "ensemble_stack" in reloaded.branch_ids(),
              ",".join(reloaded.branch_ids()))
        check("plan merged with branch direction",
              plan.method_detail.get("branch_id") == "torch_cnn"
              and plan.method_detail.get("model") == "torch_cnn"
              and plan.method_detail.get("epochs") == 5,
              str(plan.method_detail))
        reloaded.save()
        again = MethodPortfolio.load_or_default(profile, port_path)
        check("portfolio persists HERA growth",
              "ensemble_stack" in again.branch_ids(),
              ",".join(again.branch_ids()))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def test_prioritizer_platform_facts_in_prompt():
    """rc4: measured platform facts (cache sizes / F0 / incumbent kind)
    reach the HERA decision prompt as facts - no method guidance."""
    from hera.portfolio import MethodPortfolio
    from hera.prioritization import Prioritizer
    from v2_contracts import ResearchPlan
    tmp = Path(tempfile.mkdtemp(prefix="v2_platform_facts_"))
    try:
        data = _make_data_dir(tmp)
        profile = Analyzer(data, "demo").profile("demo")
        port = MethodPortfolio.load_or_default(profile, tmp / "p.json")
        prio = Prioritizer(llm_call_fn=lambda prompt: "{}")
        facts = "PLATFORM FACTS: prebuilt image caches at sizes [64, 128]; measured F0 cost ~= 12s"
        prompt = prio.build_ticket_prompt(
            profile, port, ResearchPlan(hypothesis="H"), 3,
            research_intent="cheap_probe", stage="S1_baseline",
            platform_facts=facts)
        check("platform facts reach ticket prompt", facts in prompt,
              prompt[:1200])
        check("facts marked as measurable",
              "Measured platform facts" in prompt, prompt[:1200])
        check("no platform method mandate",
              "must use" not in prompt.lower(), prompt[:1200])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _late_sof_jpeg(w, h, extra=300):
    """Minimal JPEG whose SOF0 (width/height) sits far past byte 64, like
    real photos with EXIF/ICC segments. Regression for the modality bug:
    a 64-byte header read could never reach the SOF marker, so real JPEG
    tasks were misclassified as tabular."""
    app0 = (b"\xff\xe0" + (16).to_bytes(2, "big")
            + b"JFIF\x00" + b"\x01\x02\x00\x00\x01\x00\x01\x00\x00")
    app1_payload = b"Exif\x00\x00" + b"\x11" * max(0, extra - 6)
    app1 = b"\xff\xe1" + (2 + len(app1_payload)).to_bytes(2, "big") + app1_payload
    sof_payload = (b"\x08" + h.to_bytes(2, "big") + w.to_bytes(2, "big")
                   + b"\x03" + b"\x01\x11\x00" * 3)
    sof = b"\xff\xc0" + (2 + len(sof_payload)).to_bytes(2, "big") + sof_payload
    return b"\xff\xd8" + app0 + app1 + sof + b"\xff\xd9"


def test_analyzer_nested_image_layout_modality():
    """aerial/dog style: images under train/ (flat or class subdirs) with
    bare-id CSVs; modality MUST come from the recursive magic-verified
    image file count AND real dims must be probed even when the JPEG SOF
    marker sits far past byte 64 (generic, no competition names)."""
    tmp = Path(tempfile.mkdtemp(prefix="v2_nested_img_"))
    try:
        data = tmp / "nested_task"
        public = data / "prepared" / "public"
        private = data / "prepared" / "private"
        public.mkdir(parents=True)
        private.mkdir(parents=True)
        (public / "train.csv").write_text(
            "id,label\n" + "".join("img_%04d.jpg,%d\n" % (i, i % 2)
                                    for i in range(60)),
            encoding="utf-8")
        (public / "test.csv").write_text("id\nimg_0001.jpg\n",
                                         encoding="utf-8")
        (private / "test.csv").write_text("id,label\nimg_0001.jpg,0\n",
                                          encoding="utf-8")
        (public / "sample_submission.csv").write_text(
            "id,label\nimg_0001.jpg,0\n", encoding="utf-8")
        (public / "train" / "class_a").mkdir(parents=True)
        (public / "train" / "class_b").mkdir(parents=True)
        jpg = _late_sof_jpeg(120, 80)
        for i in range(60):
            sub = public / "train" / ("class_a" if i % 2 == 0 else "class_b")
            (sub / ("img_%04d.jpg" % i)).write_bytes(jpg)
        (public / "test").mkdir()
        (public / "test" / "img_0001.jpg").write_bytes(jpg)
        from data_layout import resolve_dataset_layout
        layout = resolve_dataset_layout(data)
        check("nested layout resolves to whole train dir",
              layout.train_image_dir is not None
              and layout.train_image_dir.name == "train",
              str(layout.train_image_dir))
        profile = Analyzer(data, "classify images").profile("nested_demo")
        check("nested image modality", profile.modality == "image",
              profile.modality)
        check("recursive image count", profile.image_file_count >= 60,
              str(profile.image_file_count))
        check("nested dims probe (late SOF)", profile.image_width == 120
              and profile.image_height == 80,
              "%dx%d" % (profile.image_width, profile.image_height))
        check("image notes", "Modality: image" in profile.data_notes,
              profile.data_notes)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_analyzer_image_modality_default_dims():
    """Magic-verified image dir with unparseable headers: modality must
    still be image (file-count evidence) and dims fall back to 64x64 so
    cache/resource derivation keep working."""
    tmp = Path(tempfile.mkdtemp(prefix="v2_img_default_dims_"))
    try:
        data = tmp / "dims_task"
        public = data / "prepared" / "public"
        private = data / "prepared" / "private"
        public.mkdir(parents=True)
        private.mkdir(parents=True)
        (public / "train.csv").write_text(
            "id,label\n" + "".join("img_%04d.png,%d\n" % (i, i % 2)
                                    for i in range(60)),
            encoding="utf-8")
        (public / "test.csv").write_text("id\nimg_0001.png\n",
                                         encoding="utf-8")
        (private / "test.csv").write_text("id,label\nimg_0001.png,0\n",
                                          encoding="utf-8")
        (public / "sample_submission.csv").write_text(
            "id,label\nimg_0001.png,0\n", encoding="utf-8")
        # PNG magic only, no parseable IHDR -> count works, dims probe fails
        (public / "train_images").mkdir()
        magic = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24
        for i in range(60):
            (public / "train_images" / ("img_%04d.png" % i)).write_bytes(magic)
        (public / "test_images").mkdir()
        (public / "test_images" / "img_0001.png").write_bytes(magic)
        profile = Analyzer(data, "classify images").profile("dims_demo")
        check("magic-only image modality", profile.modality == "image",
              profile.modality)
        check("image count from magic", profile.image_file_count >= 60,
              str(profile.image_file_count))
        check("default dims applied", profile.image_width == 64
              and profile.image_height == 64,
              "%dx%d" % (profile.image_width, profile.image_height))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_analyzer_aptos_style_image_layout():
    """aptos uses train_images/ + bare ids without extension; generic
    layout detection must still find the image dirs (images=True)."""
    from data_layout import resolve_dataset_layout
    tmp = Path(tempfile.mkdtemp(prefix="v2_aptos_layout_"))
    try:
        data = tmp / "aptos_task"
        public = data / "prepared" / "public"
        private = data / "prepared" / "private"
        public.mkdir(parents=True)
        private.mkdir(parents=True)
        (public / "train.csv").write_text(
            "id_code,diagnosis\n2a2274bcb00a,0\nb1b2b3c4d5e6,1\n",
            encoding="utf-8")
        (public / "test.csv").write_text(
            "id_code\nx1x2x3x4x5x6\n", encoding="utf-8")
        (private / "test.csv").write_text(
            "id_code,diagnosis\nx1x2x3x4x5x6,2\n", encoding="utf-8")
        (public / "sample_submission.csv").write_text(
            "id_code,diagnosis\nx1x2x3x4x5x6,0\n", encoding="utf-8")
        (public / "train_images").mkdir()
        (public / "test_images").mkdir()
        (public / "train_images" / "2a2274bcb00a.png").write_bytes(b"x")
        (public / "train_images" / "b1b2b3c4d5e6.png").write_bytes(b"x")
        (public / "test_images" / "x1x2x3x4x5x6.png").write_bytes(b"x")
        layout = resolve_dataset_layout(data)
        check("aptos style train image dir detected",
              layout.train_image_dir is not None
              and layout.train_image_dir.name == "train_images",
              str(layout.train_image_dir))
        check("aptos style test image dir detected",
              layout.test_image_dir is not None
              and layout.test_image_dir.name == "test_images",
              str(layout.test_image_dir))
        check("aptos style manifest images=True",
              bool(layout.manifest().get("train_images")),
              str(layout.manifest()))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)



def _make_text_data_dir(tmp, rows=60):
    """Free-text review column + short-enum description + numeric length."""
    data = tmp / "text_task"
    public = data / "prepared" / "public"
    private = data / "prepared" / "private"
    public.mkdir(parents=True)
    private.mkdir(parents=True)
    texts = [
        "this product is amazing and works perfectly every single day",
        "the battery died after only two weeks of normal use unfortunately",
        "great value for the money highly recommend to all my friends",
        "terrible build quality the screen cracked in my pocket somehow",
        "works as expected fast shipping and good packaging overall",
        "not worth the price at all returned it after three days",
    ]
    descs = ["red", "blue", "green", "black", "white", "yellow"]
    lines = ["comment_text,description,length,target"]
    for i in range(rows):
        lines.append("%s,%s,%d,%d"
                     % (texts[i % len(texts)], descs[i % len(descs)],
                        10 + i, i % 2))
    (public / "train.csv").write_text("\n".join(lines) + "\n",
                                      encoding="utf-8")
    test_lines = ["comment_text,description,length"]
    for i in range(10):
        test_lines.append("%s,%s,%d"
                          % (texts[i % len(texts)], descs[i % len(descs)],
                             20 + i))
    (public / "test.csv").write_text("\n".join(test_lines) + "\n",
                                     encoding="utf-8")
    (private / "test.csv").write_text("id\n0\n1\n", encoding="utf-8")
    (public / "sample_submission.csv").write_text(
        "id,target\n0,0\n1,0\n", encoding="utf-8")
    return data, public / "sample_submission.csv"


def _make_id_code_dir(tmp):
    """Near-unique id codes + short enums must stay tabular."""
    data = tmp / "tabular_task"
    public = data / "prepared" / "public"
    private = data / "prepared" / "private"
    public.mkdir(parents=True)
    private.mkdir(parents=True)
    descs = ["low", "medium", "high", "critical"]
    lines = ["code,description,target"]
    for i in range(60):
        lines.append("id_%08d,%s,%d" % (i, descs[i % 4], i % 2))
    (public / "train.csv").write_text("\n".join(lines) + "\n",
                                      encoding="utf-8")
    (public / "test.csv").write_text("code,description\nid_00000000,low\n",
                                     encoding="utf-8")
    (private / "test.csv").write_text("id\nx\n", encoding="utf-8")
    (public / "sample_submission.csv").write_text(
        "id,target\nx,0\n", encoding="utf-8")
    return data, public / "sample_submission.csv"


def _make_timeseries_dir(tmp):
    """Date column + numeric series (daily store sales)."""
    import datetime as _dt
    data = tmp / "ts_task"
    public = data / "prepared" / "public"
    private = data / "prepared" / "private"
    public.mkdir(parents=True)
    private.mkdir(parents=True)
    base = _dt.date(2020, 1, 1)
    lines = ["date,store,sales"]
    for i in range(90):
        d = base + _dt.timedelta(days=i)
        lines.append("%s,S1,%.1f" % (d.isoformat(), 100.0 + i + (i % 5) * 3.0))
    (public / "train.csv").write_text("\n".join(lines) + "\n",
                                      encoding="utf-8")
    (public / "test.csv").write_text("date,store\n2020-04-01,S1\n",
                                     encoding="utf-8")
    (private / "test.csv").write_text("id\nx\n", encoding="utf-8")
    (public / "sample_submission.csv").write_text(
        "id,sales\nx,0\n", encoding="utf-8")
    return data, public / "sample_submission.csv"


def _make_mixed_data_dir(tmp):
    """v2.3.3: spaceship-titanic-like table - one space-dense prose
    column (Name) among many numeric/categorical features. The prose
    column must be recorded but must NOT hijack modality to text."""
    data = tmp / "mixed_task"
    public = data / "prepared" / "public"
    private = data / "prepared" / "private"
    public.mkdir(parents=True)
    private.mkdir(parents=True)
    planets = ["Europa", "Earth", "Mars"]
    dests = ["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"]
    first = ["John", "Mary", "Wei", "Sofia", "Liam", "Aria"]
    last = ["Smith", "Chen", "Garcia", "Kim", "Mueller", "Okafor"]
    lines = ["PassengerId,HomePlanet,CryoSleep,Cabin,Destination,Age,VIP,"
             "RoomService,FoodCourt,ShoppingMall,Spa,VRDeck,Name,Transported"]
    for i in range(60):
        lines.append("%d,%s,%s,%s,%s,%d,%s,%d,%d,%d,%d,%d,%s %s,%d"
                     % (1000 + i, planets[i % 3], "True" if i % 2 else "False",
                        ["B/1/P", "C/3/S", "A/2/T"][i % 3], dests[i % 3],
                        20 + i % 60, "True" if i % 3 == 0 else "False",
                        i * 3, i * 5, i * 7, i * 11, i * 13,
                        first[i % 6], last[(i + 1) % 6], i % 2))
    (public / "train.csv").write_text("\n".join(lines) + "\n",
                                      encoding="utf-8")
    test_lines = ["PassengerId,HomePlanet,CryoSleep,Cabin,Destination,Age,VIP,"
                  "RoomService,FoodCourt,ShoppingMall,Spa,VRDeck,Name"]
    for i in range(10):
        test_lines.append("%d,%s,%s,%s,%s,%d,%s,%d,%d,%d,%d,%d,%s %s"
                          % (2000 + i, planets[i % 3],
                             "True" if i % 2 else "False",
                             ["B/1/P", "C/3/S", "A/2/T"][i % 3], dests[i % 3],
                             20 + i, "False", i, i, i, i, i,
                             first[i % 6], last[(i + 1) % 6]))
    (public / "test.csv").write_text("\n".join(test_lines) + "\n",
                                     encoding="utf-8")
    (private / "test.csv").write_text("id,Transported\n0,0\n1,1\n",
                                      encoding="utf-8")
    (public / "sample_submission.csv").write_text(
        "id,Transported\n0,0\n1,1\n", encoding="utf-8")
    return data, public / "sample_submission.csv"


def test_analyzer_text_modality_content_based():
    """v2.3.2: free-text columns detected from CONTENT, not just names;
    short-enum 'description' and numeric columns stay non-text."""
    tmp = Path(tempfile.mkdtemp(prefix="v2_hera_text_"))
    try:
        data, sample = _make_text_data_dir(tmp)
        profile = Analyzer(data, "classify whether a review is positive",
                           sample_path=str(sample)).profile("text_demo")
        check("content text modality", profile.modality == "text",
              profile.modality)
        check("text column detected", "comment_text" in profile.text_columns,
              str(profile.text_columns))
        check("short enum not text", "description" not in profile.text_columns,
              str(profile.text_columns))
        check("numeric column not text", "length" not in profile.text_columns,
              str(profile.text_columns))
        check("text notes", "Text columns (content-verified)" in profile.data_notes,
              profile.data_notes)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_analyzer_id_codes_stay_tabular():
    """v2.3.2: near-unique id/code columns are keys, never text."""
    tmp = Path(tempfile.mkdtemp(prefix="v2_hera_tab_"))
    try:
        data, sample = _make_id_code_dir(tmp)
        profile = Analyzer(data, "classify severity",
                           sample_path=str(sample)).profile("tab_demo")
        check("id codes stay tabular", profile.modality == "tabular",
              profile.modality)
        check("no text columns", profile.text_columns == [],
              str(profile.text_columns))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_analyzer_timeseries_evidence():
    """v2.3.2: task_type=timeseries needs prompt hint AND date evidence."""
    tmp = Path(tempfile.mkdtemp(prefix="v2_hera_ts_"))
    try:
        data, sample = _make_timeseries_dir(tmp)
        profile = Analyzer(data, "forecast future daily sales",
                           sample_path=str(sample)).profile("ts_demo")
        check("timeseries task type", profile.task_type == "timeseries",
              profile.task_type)
        check("time column detected", profile.time_column == "date",
              profile.time_column)
        check("modality stays tabular", profile.modality == "tabular",
              profile.modality)
        # same data WITHOUT a temporal prompt must NOT be timeseries
        profile2 = Analyzer(data, "predict sales value",
                            sample_path=str(sample)).profile("ts_demo2")
        check("no prompt hint -> not timeseries",
              profile2.task_type == "regression", profile2.task_type)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_analyzer_mixed_text_column_stays_tabular():
    """v2.3.3: a space-dense prose column (Name) in a mostly-tabular
    table must be recorded as a text column but modality stays tabular so
    HERA keeps the tabular method space (spaceship-titanic style)."""
    tmp = Path(tempfile.mkdtemp(prefix="v2_hera_mixed_"))
    try:
        data, sample = _make_mixed_data_dir(tmp)
        profile = Analyzer(data, "predict whether a passenger is transported",
                           sample_path=str(sample)).profile("mixed_demo")
        check("mixed data stays tabular", profile.modality == "tabular",
              profile.modality)
        check("name column recorded as text", "Name" in profile.text_columns,
              str(profile.text_columns))
        check("mixed notes", "Mixed data:" in profile.data_notes,
              profile.data_notes)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print("=== V2 HERA tests ===\n")
    test_analyzer_classification()
    test_analyzer_mlebench_prepared_layout()
    test_analyzer_aptos_style_image_layout()
    test_analyzer_nested_image_layout_modality()
    test_analyzer_image_modality_default_dims()
    test_planner_stub_llm()
    test_planner_parses_intent_and_children()
    test_analyzer_regression_target_stats()
    test_analyzer_text_modality_content_based()
    test_analyzer_id_codes_stay_tabular()
    test_analyzer_timeseries_evidence()
    test_analyzer_mixed_text_column_stays_tabular()
    test_planner_no_llm_fallback()
    test_planner_resource_budget()
    test_resource_profiles()
    test_memory_strategy_pool()
    test_interpreter()
    test_portfolio_hera_writes_branches()
    test_prioritizer_platform_facts_in_prompt()
    print("\nRESULT=" + ("PASS" if not FAILURES else "FAIL:" + ",".join(FAILURES)))
    sys.exit(0 if not FAILURES else 1)
