# -*- coding: utf-8 -*-
"""v2.3.9 offline tests: HERA upper-bound capabilities + PACT execution
hardening for the FULL MLE-Bench surface (no competition names anywhere).

Covers:
  1) capability registry: image.finetune.timm.v2 (LR schedule / augment /
     AMP / TTA / label smoothing) and image.finetune.ensemble.v1
     (model_names x seeds, max 4 members) with generic image contracts;
  2) compiler renders both templates deterministically: %.9f probability
     output, generic probability-row normalization (_norm_proba_row), and
     torch.nn imports; legacy image templates got the same fix;
  3) resource profiler: cached timm/HF weights + big GPU raise the image
     budget/epoch ceiling generically (modality-driven, never task-named);
  4) PACT executor: --shm-size=1g + huggingface-hub cache mount +
     preflight now globs HF hub repos into the pretrained whitelist;
  5) publisher self-check: probability submissions are clipped +
     renormalized before delivery (no NaN/negative/off-by-rounding rows);
     regression outputs are left untouched;
  6) closed-loop finalize: official-grade re-eval is fail-open and recorded
     in RESULT_SUMMARY (skipped when mlebench is unavailable);
  7) executed fine-tune/ensemble harness end-to-end on tiny synthetic
     images (skipped when timm+torchvision are both absent).

Run: python test_v2_239.py   (from the aisci payload dir)
"""
import csv
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from capability_registry import CapabilityRegistry  # noqa: E402
from hera.portfolio import ResourceProfiler  # noqa: E402
from pact.executor import Executor  # noqa: E402
from pact.publisher import ControlledPublisher  # noqa: E402
from program_compiler import ProgramCompiler  # noqa: E402
from v2_contracts import (AnalysisProfile, MethodInvocationV1,  # noqa: E402
                          PromotionRecord, ResearchPlan, TrialSpec)

FAILURES = []
SKIPPED = []


def check(name, cond, detail=""):
    if cond:
        print("[OK] " + name)
    else:
        print("[FAIL] " + name + (" | " + str(detail)[:400] if detail else ""))
        FAILURES.append(name)


def skip(name, reason):
    print("[SKIP] " + name + " | " + reason)
    SKIPPED.append(name)


def _write_csv(path, rows):
    with io.open(str(path), "w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(rows)


def _img_profile(rows=5000, ncls=50, modality="image"):
    return AnalysisProfile(competition="demo", modality=modality,
                           task_type="classification", train_rows=rows,
                           test_rows=500, image_width=64, image_height=64,
                           feature_columns=[], n_classes=ncls)


# ------------------------------------------------------------- 1 registry
def test_registry_v239():
    reg = CapabilityRegistry()
    v2 = reg.get("image.finetune.timm.v2")
    check("registry: finetune.v2 exists", v2 is not None and not v2.broken, "")
    if v2 is not None:
        check("registry: finetune.v2 renderer", v2.renderer == "image_finetune_timm_v2", v2.renderer)
        check("registry: finetune.v2 gpu", v2.gpu, "")
        check("registry: finetune.v2 modality image", "image" in (v2.supported_modalities or []), "")
        for key in ("lr_schedule", "augment", "amp", "tta_flip", "label_smoothing"):
            check("registry: finetune.v2 param %s" % key, key in (v2.parameter_schema or {}), "")
        ps = (v2.parameter_schema or {}).get("image_size", {})
        check("registry: finetune.v2 image_size up to 384",
              ps.get("max", 0) >= 384, str(ps))
        eps = (v2.parameter_schema or {}).get("epochs", {})
        check("registry: finetune.v2 epochs up to 12",
              eps.get("max", 0) >= 12, str(eps))
    ens = reg.get("image.finetune.ensemble.v1")
    check("registry: ensemble.v1 exists", ens is not None and not ens.broken, "")
    if ens is not None:
        check("registry: ensemble.v1 renderer", ens.renderer == "image_finetune_ensemble", ens.renderer)
        check("registry: ensemble.v1 gpu", ens.gpu, "")
        mns = (ens.parameter_schema or {}).get("model_names", {})
        seeds = (ens.parameter_schema or {}).get("seeds", {})
        check("registry: ensemble model_names list<=3", mns.get("type") == "list" and mns.get("max_len") == 3, str(mns))
        check("registry: ensemble seeds list<=2", seeds.get("type") == "list" and seeds.get("max_len") == 2, str(seeds))


# ------------------------------------------------------------- 2 compiler
def test_compiler_renders_v239():
    reg = CapabilityRegistry()
    comp = ProgramCompiler(reg)
    prof = _img_profile()
    manifest = {"metric_name": "logloss", "metric_direction": "min"}
    inv = MethodInvocationV1(
        method_id="image.finetune.timm.v2", hypothesis="h",
        params={"model_name": "efficientnet_b0", "image_size": 128,
                "epochs": 4, "lr": 3e-4, "batch_size": 32,
                "weight_decay": 1e-4, "early_stop_patience": 3,
                "val_seed": 42, "lr_schedule": "cosine", "augment": "flip",
                "amp": True, "tta_flip": True, "label_smoothing": 0.05,
                "max_rows": 15000},
        preprocessing=["cached_image_arrays", "imagenet_norm",
                       "pretrained_weight_cache", "flip_augment"],
        validation="single_holdout",
        resource_request={"max_train_rows": 15000, "epochs": 4,
                          "image_size": 128, "batch_size": 32})
    code, th = comp.render(inv, profile=prof, manifest=manifest)
    check("compiler: finetune.v2 renders", len(code) > 1000 and th.startswith("sha256:"), th[:24])
    check("compiler: finetune.v2 no tokens", "@@" not in code, "")
    check("compiler: finetune.v2 nn import", "import torch.nn as nn" in code, "")
    check("compiler: finetune.v2 %.9f", '"%.9f"' in code, "")
    check("compiler: finetune.v2 row norm", "_norm_proba_row" in code, "")
    check("compiler: finetune.v2 flip aug", "Image.FLIP_LEFT_RIGHT" in code, "")
    check("compiler: finetune.v2 cosine schedule", "CosineAnnealingLR" in code, "")

    inv2 = MethodInvocationV1(
        method_id="image.finetune.ensemble.v1", hypothesis="h2",
        params={"model_names": ["efficientnet_b0", "resnet18"],
                "seeds": [42, 7], "image_size": 128, "epochs": 3,
                "lr": 3e-4, "batch_size": 24, "weight_decay": 1e-4,
                "early_stop_patience": 3, "val_seed": 42,
                "lr_schedule": "cosine", "augment": "strong", "amp": True,
                "tta_flip": True, "label_smoothing": 0.0},
        preprocessing=["cached_image_arrays", "imagenet_norm",
                       "pretrained_weight_cache"],
        validation="single_holdout",
        resource_request={"max_train_rows": 10000, "epochs": 3,
                          "image_size": 128, "batch_size": 24})
    code2, th2 = comp.render(inv2, profile=prof, manifest=manifest)
    check("compiler: ensemble renders", len(code2) > 1000 and th2.startswith("sha256:"), th2[:24])
    check("compiler: ensemble no tokens", "@@" not in code2, "")
    check("compiler: ensemble nn import", "import torch.nn as nn" in code2, "")
    check("compiler: ensemble %.9f", '"%.9f"' in code2, "")
    check("compiler: ensemble row norm", "_norm_proba_row" in code2, "")
    check("compiler: ensemble model list", "model_names" in code2, "")
    check("compiler: ensemble caps members", "len(combos) >= 4" in code2, "")

    # legacy image templates carry the same submission-quality fix
    inv3 = MethodInvocationV1(
        method_id="image.embedding.timm.v1", hypothesis="h3",
        params={"model_name": "efficientnet_b0", "image_size": 128, "C": 1.0,
                "max_iter": 1000, "batch_size": 64, "max_rows": 20000,
                "val_seed": 42, "folds": 3},
        preprocessing=["cached_image_arrays", "imagenet_norm",
                       "pretrained_weight_cache"],
        validation="stratified_kfold",
        resource_request={"max_train_rows": 20000, "folds": 3,
                          "image_size": 128})
    code3, _ = comp.render(inv3, profile=prof, manifest=manifest)
    check("compiler: legacy embed %.9f", '"%.9f"' in code3, "")
    check("compiler: legacy embed row norm", "_norm_proba_row" in code3, "")
    # tabular/text harness carries the same probability-quality fix
    tab = AnalysisProfile(competition="demo", modality="tabular",
                          task_type="classification", train_rows=100,
                          feature_columns=["a", "b"], n_classes=2)
    inv4 = MethodInvocationV1(
        method_id="tabular.linear.logistic.v1", hypothesis="h4",
        params={"C": 1.0, "max_iter": 1000, "scaling": "standard",
                "missing": "mean", "val_seed": 42, "folds": 5},
        preprocessing=["missing_value_impute", "standard_scaling"],
        validation="stratified_kfold",
        resource_request={"max_train_rows": 100, "folds": 5})
    code4, _ = comp.render(inv4, profile=tab, manifest=manifest)
    check("compiler: tabular harness %.9f", '"%.9f"' in code4, "")
    check("compiler: tabular harness row norm", "_norm_proba_row" in code4, "")
    # wrong-modality rejection is generic
    tab2 = AnalysisProfile(competition="demo", modality="tabular",
                           task_type="regression", train_rows=100,
                           feature_columns=["a", "b"], n_classes=0)
    ok, reason = comp.validate(MethodInvocationV1(
        method_id="image.finetune.timm.v2", params={}), profile=tab2)
    check("compiler: finetune.v2 rejected for tabular", not ok, reason)


# ------------------------------------------------------------- 3 resources
def test_resource_profiler_pretrained_boost():
    plain = ResourceProfiler(gpu_memory_mb=40960).derive(_img_profile())
    cached = ResourceProfiler(gpu_memory_mb=40960,
                              cached_weights=["resnet18", "efficientnet_b0",
                                              "resnet50", "convnext_tiny",
                                              "mobilenetv3_large_100",
                                              "vit_tiny_patch16_224"]).derive(
                                  _img_profile())
    check("resource: cached image epochs_max <= 12",
          cached["epochs_max"] <= 12, str(cached["epochs_max"]))
    check("resource: cached image epochs boosted",
          cached["epochs_max"] > plain["epochs_max"], "%s vs %s" % (
              cached["epochs_max"], plain["epochs_max"]))
    check("resource: cached image budget boosted",
          cached["max_budget_seconds"] > plain["max_budget_seconds"],
          "%s vs %s" % (cached["max_budget_seconds"],
                        plain["max_budget_seconds"]))
    # tabular epoch ceiling must NOT move (modality-driven, not name-driven)
    tab = AnalysisProfile(competition="demo", modality="tabular",
                          task_type="regression", train_rows=10000,
                          feature_columns=["f%d" % i for i in range(20)],
                          n_classes=0)
    t_plain = ResourceProfiler(gpu_memory_mb=40960).derive(tab)
    t_cached = ResourceProfiler(gpu_memory_mb=40960,
                                cached_weights=["a"] * 10).derive(tab)
    check("resource: tabular epochs unchanged",
          t_cached["epochs_max"] == t_plain["epochs_max"],
          "%s vs %s" % (t_cached["epochs_max"], t_plain["epochs_max"]))
    # small GPU / no weights stays conservative
    small = ResourceProfiler(gpu_memory_mb=8192).derive(_img_profile())
    check("resource: small-gpu epochs conservative",
          small["epochs_max"] <= 8, str(small["epochs_max"]))


# ------------------------------------------------------------- 4 executor
def test_executor_docker_cmd_v239():
    tmp = Path(tempfile.mkdtemp(prefix="v239_exec_"))
    try:
        pub = tmp / "prepared" / "public"
        priv = tmp / "prepared" / "private"
        pub.mkdir(parents=True)
        priv.mkdir(parents=True)
        (pub / "train.csv").write_text("id,label\n1,0\n", encoding="utf-8")
        (pub / "test.csv").write_text("id\n2\n", encoding="utf-8")
        (pub / "sample_submission.csv").write_text("id,label\n2,0\n",
                                                   encoding="utf-8")
        (priv / "test.csv").write_text("id,label\n2,0\n", encoding="utf-8")
        from data_layout import resolve_dataset_layout
        manifest = resolve_dataset_layout(tmp / "prepared").manifest()
        manifest.update({"train_csv": str(pub / "train.csv"),
                         "test_csv": str(pub / "test.csv"),
                         "target_column": "label",
                         "task_type": "classification"})
        torch_cache = tmp / "torch_cache"
        hf_cache = tmp / "hf_cache"
        torch_cache.mkdir()
        hf_cache.mkdir()
        ex = Executor(tmp / "work", exec_image="exec:v2", docker_bin="docker",
                      manifest=manifest, torch_cache=str(torch_cache),
                      hf_cache=str(hf_cache))
        spec = TrialSpec.seal("demo", ResearchPlan(), "x")
        cmd = ex.docker_cmd(spec, "round_1_x.py", str(tmp / "work"), {})
        joined = " ".join(cmd)
        check("executor: shm-size 1g", "--shm-size=1g" in joined, joined)
        check("executor: hf cache mounted",
              "-v %s:/root/.cache/huggingface" % hf_cache in joined, joined)
        check("executor: torch cache mounted",
              "-v %s:/root/.cache/torch" % torch_cache in joined, joined)
        # hf cache defaults to a v2_hf_cache sibling of the torch cache even
        # when the directory does not exist yet (docker -v creates it), and
        # the host V2_HF_CACHE env must not leak into this unit test
        # (launchers export it, e.g. V2_HF_CACHE=/mnt/data/v2_hf_cache).
        import unittest.mock as mock
        sibling = Path(torch_cache).parent / "v2_hf_cache"
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("V2_HF_CACHE", None)
            ex2 = Executor(tmp / "work2", exec_image="exec:v2",
                           docker_bin="docker", manifest=manifest,
                           torch_cache=str(torch_cache))
        check("executor: hf cache sibling default",
              ex2.hf_cache == str(sibling), str(ex2.hf_cache))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_executor_timeout_bytes_mode():
    # v2.3.9 regression: subprocess.run(text=True) + timeout crashes on
    # CPython 3.12 (b''.join(str_seq) -> TypeError inside _check_timeout),
    # which killed the host daemon and lost the outcome on long trials
    # (new-york-city-taxi-fare-prediction, 2026-08-07: 19 committed trials,
    # 0 receipts). The executor must run in bytes mode and surface a
    # timed_out ExecOutcome instead of raising.
    # v2.5.0: this unit test must exercise the HOST subprocess path; the
    # launcher exports V2_EXEC_IMAGE, which Executor.__init__ would pick up
    # and switch to container mode (then _mount_root() fail-closes on a
    # manifest-less unit test). Pop the ambient launcher env like the
    # V2_HF_CACHE sibling test below does.
    import unittest.mock as mock
    tmp = Path(tempfile.mkdtemp(prefix="v239_to_"))
    try:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("V2_EXEC_IMAGE", None)
            os.environ.pop("V2_DOCKER_BIN", None)
            ex = Executor(tmp / "work", exec_image="", docker_bin="docker",
                          python_bin=sys.executable)
        code = "import time\ntime.sleep(60)\n"
        spec = TrialSpec.seal("demo", ResearchPlan(), code)
        outcome = ex.run(spec, timeout_seconds=2)
        check("executor: timeout returns timed_out",
              outcome.timed_out and outcome.returncode == -9
              and "TIMEOUT" in outcome.stderr, str(outcome))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_preflight_hf_glob():
    import unittest.mock as mock
    tmp = Path(tempfile.mkdtemp(prefix="v239_pf_"))
    try:
        pub = tmp / "prepared" / "public"
        priv = tmp / "prepared" / "private"
        pub.mkdir(parents=True)
        priv.mkdir(parents=True)
        (pub / "train.csv").write_text("id,label\n1,0\n", encoding="utf-8")
        (priv / "test.csv").write_text("id,label\n2,0\n", encoding="utf-8")
        from data_layout import resolve_dataset_layout
        manifest = resolve_dataset_layout(tmp / "prepared").manifest()
        manifest.update({"train_csv": str(pub / "train.csv"),
                         "target_column": "label"})
        ex = Executor(tmp / "work", exec_image="exec:v2", docker_bin="docker",
                      manifest=manifest, torch_cache=str(tmp / "tc"),
                      hf_cache=str(tmp / "hc"))
        captured = {}

        def fake_run(cmd, *a, **k):
            if "--version" in cmd:
                return subprocess.CompletedProcess(cmd, 0, b"20.10.0", b"")
            captured["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, b"PREFLIGHT_OK\n", b"")

        with mock.patch("pact.executor.subprocess.run", side_effect=fake_run):
            pf = ex.preflight()
        script = " ".join(captured.get("cmd") or [])
        check("preflight: runs to completion", pf.get("status") == "ok", str(pf))
        check("preflight: hf glob in script",
              "huggingface/hub/models--*" in script, script[:400])
        check("preflight: hf line in script",
              "PREFLIGHT_HF_PRETRAINED" in script, script[:400])
        check("preflight: hf mount in cmd",
              "%s:/root/.cache/huggingface" % (tmp / "hc") in script, "")
        check("preflight: torch mount in cmd",
              "%s:/root/.cache/torch" % (tmp / "tc") in script, "")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------- 5 publisher
def test_publisher_sanitize():
    from pact.publisher import PublishError
    tmp = Path(tempfile.mkdtemp(prefix="v239_pub_"))
    try:
        bundles = tmp / "bundles"
        bundles.mkdir()
        sub_dir = tmp / "submission"
        sub_dir.mkdir()
        poison = tmp / "poison.csv"
        _write_csv(poison, [
            ["id", "c1", "c2", "c3"],
            ["a", "0.5", "0.5", "0.5"],      # sum 1.5 (repaired)
            ["b", "0.3", "-0.1", "0.9"],     # negative (repaired)
            ["c", "0.33", "0.33", "nan"],    # NaN (repaired)
            ["d", "0.4", "0.4", "0.2"],      # clean (rewritten %.9f)
            ["e", "0.2", "0.5", "0.3"],
            ["f", "0.1", "0.8", "0.1"],
            ["g", "0.25", "0.25", "0.5"],
            ["h", "0.6", "0.35", "0.05"],
        ])
        from v2_contracts import CandidateBundle
        bundle = CandidateBundle(bundle_id="b1", trial_id="trial_1",
                                 submission_path=str(poison))
        (bundles / "bundle_b1.json").write_text(
            json.dumps(bundle.to_dict()), encoding="utf-8")

        class FakeBus:
            host_bundles = bundles

            def load_promotion(self):
                return PromotionRecord(competition="c",
                                       certified_best_trial_id="trial_1").to_dict()

        pub = ControlledPublisher(FakeBus(), sub_dir)
        dst = pub.publish_certified()
        rows = list(csv.reader(io.open(str(dst), encoding="utf-8")))
        check("publisher: sanitized file exists", dst.is_file(), "")
        check("publisher: row count kept", len(rows) == 9, str(len(rows)))
        bad = 0
        sums_ok = True
        for r in rows[1:]:
            vals = [float(r[i]) for i in range(1, 4)]
            if abs(sum(vals) - 1.0) > 1e-9 or min(vals) < 0.0:
                sums_ok = False
            if any(v != v for v in vals):
                bad += 1
        check("publisher: all rows sum to 1", sums_ok, str(rows))
        check("publisher: no NaN left", bad == 0, "")
        check("publisher: 9-decimal formatting",
              all(len(r[1].split(".")[1]) == 9 for r in rows[1:]), str(rows[1]))

        # regression outputs are untouched
        reg = tmp / "reg.csv"
        _write_csv(reg, [["id", "fare"], ["a", "3.5"], ["b", "100.0"]])
        bundle2 = CandidateBundle(bundle_id="b2", trial_id="trial_2",
                                  submission_path=str(reg))
        (bundles / "bundle_b2.json").write_text(
            json.dumps(bundle2.to_dict()), encoding="utf-8")

        class FakeBus2(FakeBus):
            def load_promotion(self):
                return PromotionRecord(competition="c",
                                       certified_best_trial_id="trial_2").to_dict()

        pub2 = ControlledPublisher(FakeBus2(), sub_dir)
        dst2 = pub2.publish_certified()
        raw = dst2.read_text(encoding="utf-8")
        check("publisher: regression untouched",
              "3.5" in raw and "100.0" in raw and "9" not in raw.split("\n")[1], raw)
        # nothing certified -> no-op error
        class FakeBus3(FakeBus):
            def load_promotion(self):
                return None
        try:
            ControlledPublisher(FakeBus3(), tmp / "sub2").publish_certified()
            check("publisher: no-certified raises", False, "")
        except PublishError:
            check("publisher: no-certified raises", True, "")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------- 6 finalize
def test_finalize_official_grade_fail_open():
    from v2_closed_loop import ClosedLoop
    obj = ClosedLoop.__new__(ClosedLoop)
    obj._log = lambda *a, **k: None

    def boom(sub):
        raise RuntimeError("mlebench not importable in control env")
    obj._official_grade = boom
    res = ClosedLoop._finalize_official_grade(obj, Path("missing.csv"))
    check("finalize: no submission -> skipped",
          res.get("status") == "skipped" and "no submission" in res.get("reason", ""), str(res))
    res2 = ClosedLoop._finalize_official_grade(obj, Path(__file__))
    check("finalize: grade failure fail-open",
          res2.get("status") == "skipped" and "mlebench" in res2.get("reason", ""), str(res2))

    obj._official_grade = lambda sub: {"score": 0.9123, "any_medal": True,
                                       "valid_submission": True,
                                       "competition_id": "demo"}
    res3 = ClosedLoop._finalize_official_grade(obj, Path(__file__))
    check("finalize: grade ok recorded",
          res3.get("status") == "ok" and res3.get("score") == 0.9123 and
          res3.get("any_medal") is True, str(res3))


# ------------------------------------------------------------- 7 execution
def _make_tiny_image_dataset(tmp):
    """8x8 RGB PNGs; train/test dirs + CSVs (generic image classification)."""
    pub = tmp / "prepared" / "public"
    priv = tmp / "prepared" / "private"
    (pub / "train").mkdir(parents=True)
    (pub / "test").mkdir(parents=True)
    priv.mkdir(parents=True)
    from PIL import Image
    import numpy as np
    rng = np.random.RandomState(0)
    for i in range(1, 9):
        arr = (rng.rand(8, 8, 3) * 255).astype("uint8")
        Image.fromarray(arr).save(pub / "train" / ("img%02d.png" % i))
    for i in range(1, 4):
        arr = (rng.rand(8, 8, 3) * 255).astype("uint8")
        Image.fromarray(arr).save(pub / "test" / ("t%02d.png" % i))
    _write_csv(pub / "train.csv",
               [["id", "label"]] +
               [["img%02d" % i, "cat" if i % 2 else "dog"] for i in range(1, 9)])
    _write_csv(pub / "test.csv", [["id"]] + [["t%02d" % i] for i in range(1, 4)])
    _write_csv(pub / "sample_submission.csv",
               [["id", "cat", "dog"]] +
               [["t%02d" % i, "0.5", "0.5"] for i in range(1, 4)])
    _write_csv(priv / "test.csv",
               [["id", "label"]] + [["t%02d" % i, "cat"] for i in range(1, 4)])
    return pub


def _run_harness(code, work_dir, env):
    proc = subprocess.run([sys.executable, "-c", code],
                          cwd=str(work_dir), env=dict(os.environ, **env),
                          capture_output=True, text=True, timeout=900)
    return proc


def test_harness_execution():
    has_timm = importlib.util.find_spec("timm") is not None
    has_tv = importlib.util.find_spec("torchvision") is not None
    if not (has_timm or has_tv):
        skip("harness: finetune execution",
             "timm and torchvision both missing in this interpreter")
        return
    tmp = Path(tempfile.mkdtemp(prefix="v239_harness_"))
    try:
        reg = CapabilityRegistry()
        comp = ProgramCompiler(reg)
        pub = _make_tiny_image_dataset(tmp)
        prof = _img_profile(rows=8, ncls=2)
        manifest = {"metric_name": "logloss", "metric_direction": "min"}
        env = {"TRAIN_CSV": str(pub / "train.csv"),
               "TEST_CSV": str(pub / "test.csv"),
               "SAMPLE_SUBMISSION": str(pub / "sample_submission.csv"),
               "TARGET_COLUMN": "label",
               "TASK_TYPE": "classification",
               "TRAIN_IMAGES": str(pub / "train"),
               "TEST_IMAGES": str(pub / "test"),
               "V2_CACHE_DIRS": "{}"}
        # --- finetune v2 ---
        inv = MethodInvocationV1(
            method_id="image.finetune.timm.v2", hypothesis="h",
            params={"model_name": "resnet18", "image_size": 64, "epochs": 2,
                    "lr": 1e-3, "batch_size": 16, "weight_decay": 1e-4,
                    "early_stop_patience": 2, "val_seed": 42,
                    "lr_schedule": "cosine", "augment": "none", "amp": False,
                    "tta_flip": True, "label_smoothing": 0.0},
            preprocessing=["cached_image_arrays", "imagenet_norm",
                           "pretrained_weight_cache"],
            validation="single_holdout",
            resource_request={"max_train_rows": 8, "epochs": 2,
                              "image_size": 64, "batch_size": 16})
        code, _ = comp.render(inv, profile=prof, manifest=manifest)
        wd = tmp / "wd_v2"
        wd.mkdir()
        r = _run_harness(code, wd, env)
        check("harness finetune.v2: exit 0", r.returncode == 0,
              (r.stdout or "")[-500:] + (r.stderr or "")[-500:])
        if r.returncode == 0:
            check("harness finetune.v2: oof written", (wd / "oof.csv").is_file(), "")
            check("harness finetune.v2: submission written",
                  (wd / "submission.csv").is_file(), "")
            rows = list(csv.reader(io.open(str(wd / "submission.csv"), encoding="utf-8")))
            ok = len(rows) == 4 and rows[0][1:] == ["cat", "dog"]
            if ok:
                for rw in rows[1:]:
                    vals = [float(rw[1]), float(rw[2])]
                    if abs(sum(vals) - 1.0) > 1e-9:
                        ok = False
            check("harness finetune.v2: proba rows sum to 1", ok, str(rows))
        # --- ensemble (1 member to keep the smoke fast) ---
        inv2 = MethodInvocationV1(
            method_id="image.finetune.ensemble.v1", hypothesis="h2",
            params={"model_names": ["resnet18"], "seeds": [42],
                    "image_size": 64, "epochs": 2, "lr": 1e-3,
                    "batch_size": 16, "weight_decay": 1e-4,
                    "early_stop_patience": 2, "val_seed": 42,
                    "lr_schedule": "cosine", "augment": "none", "amp": False,
                    "tta_flip": True, "label_smoothing": 0.0},
            preprocessing=["cached_image_arrays", "imagenet_norm",
                           "pretrained_weight_cache"],
            validation="single_holdout",
            resource_request={"max_train_rows": 8, "epochs": 2,
                              "image_size": 64, "batch_size": 16})
        code2, _ = comp.render(inv2, profile=prof, manifest=manifest)
        wd2 = tmp / "wd_ens"
        wd2.mkdir()
        r2 = _run_harness(code2, wd2, env)
        check("harness ensemble: exit 0", r2.returncode == 0,
              (r2.stdout or "")[-500:] + (r2.stderr or "")[-500:])
        if r2.returncode == 0:
            rows2 = list(csv.reader(io.open(str(wd2 / "submission.csv"), encoding="utf-8")))
            ok2 = len(rows2) == 4
            if ok2:
                for rw in rows2[1:]:
                    vals = [float(rw[1]), float(rw[2])]
                    if abs(sum(vals) - 1.0) > 1e-9:
                        ok2 = False
            check("harness ensemble: proba rows sum to 1", ok2, str(rows2))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print("=== V2.3.9 tests ===\n")
    test_registry_v239()
    test_compiler_renders_v239()
    test_resource_profiler_pretrained_boost()
    test_executor_docker_cmd_v239()
    test_executor_timeout_bytes_mode()
    test_preflight_hf_glob()
    test_publisher_sanitize()
    test_finalize_official_grade_fail_open()
    test_harness_execution()
    if SKIPPED:
        print("SKIPPED: %s" % ", ".join(SKIPPED))
    if FAILURES:
        print("RESULT=FAIL:%s" % ",".join(FAILURES))
        sys.exit(1)
    print("RESULT=PASS")
