# -*- coding: utf-8 -*-
"""v2.3.8 offline tests: generic structural task recognition (RLE masks /
bbox detection / audio) + deterministic baseline templates.

Covers the previously-unrecognized MLE-Bench families as GENERIC layout
classes - never competition-specific:
  1) RLE-mask target sniffing (empty / '-' rows are valid 'no mask');
  2) bbox coordinate-column sets + repeated-id multi-row signal;
  3) JSON box columns on ANY column (nested per-object box lists);
  4) audio file counting -> audio modality (threshold);
  5) modality override order (image_mask > image_detection > audio);
  6) label-free table synthesis: no train/test table + sample submission
     (audio/mask/detection shapes) -> synthesized train/test tables;
     dir-label train tables (train/audio/<label>/*.wav);
     the flat-candidate guard never hijacks prepared/public layouts;
  7) metrics registry: segmentation -> dice, detection -> map_at_k;
  8) capability registry: the three v2.3.8 baselines exist with the right
     contracts; compiler renders them with single_holdout + no preprocessing;
  9) executed harnesses end-to-end: oof.csv + submission.csv correct and
     metric printed for mask / detection / audio baselines.

Run: python test_v2_238.py   (from the aisci payload dir)
"""
import os
import csv
import io
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from capability_registry import BUILTIN_SPECS, CapabilityRegistry
from data_layout import resolve_dataset_layout, DatasetLayoutError
from hera.analyzer import Analyzer
from metrics_registry import infer_metric_spec
from program_compiler import ProgramCompiler
from v2_contracts import AnalysisProfile, MethodInvocationV1

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("[OK] " + name)
    else:
        print("[FAIL] " + name + (" | " + str(detail)[:300] if detail else ""))
        FAILURES.append(name)


# ---------------------------------------------------------------- fixtures
def _write_csv(path, rows):
    with io.open(str(path), "w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(rows)


def _make_mask_dataset(tmp):
    """hubmap/tgs/uw shape: train.csv with RLE target + image dirs."""
    pub = tmp / "prepared" / "public"
    priv = tmp / "prepared" / "private"
    (pub / "train").mkdir(parents=True)
    (pub / "test").mkdir(parents=True)
    priv.mkdir(parents=True)
    rows = [["id", "rle_mask", "image_height", "image_width"]]
    for i in range(1, 11):
        rows.append(["img%02d" % i, ("1 2 5 3" if i % 2 else ""), "16", "16"])
    _write_csv(pub / "train.csv", rows)
    _write_csv(pub / "sample_submission.csv",
               [["id", "rle_mask"]] + [["t%02d" % i, ""] for i in range(1, 6)])
    _write_csv(priv / "test.csv", [["id"]] + [["t%02d" % i] for i in range(1, 6)])
    return pub


def _make_detection_dataset(tmp):
    """vinbigdata shape: per-image box rows with coordinate columns."""
    pub = tmp / "prepared" / "public"
    priv = tmp / "prepared" / "private"
    (pub / "train").mkdir(parents=True)
    (pub / "test").mkdir(parents=True)
    priv.mkdir(parents=True)
    rows = [["image_id", "class_id", "x_min", "y_min", "x_max", "y_max"]]
    for i in range(1, 11):
        rows.append(["img%02d" % i, "14", "0", "0", "1", "1"])
        if i % 2 == 0:
            rows.append(["img%02d" % i, "5", "1", "2", "3", "4"])
    _write_csv(pub / "train.csv", rows)
    _write_csv(pub / "sample_submission.csv",
               [["image_id", "PredictionString"]] +
               [["t%02d" % i, "14 1 0 0 1 1"] for i in range(1, 6)])
    _write_csv(priv / "test.csv", [["image_id"]] + [["t%02d" % i] for i in range(1, 6)])
    return pub


def _make_json_box_dataset(tmp):
    """kuzushiji shape: JSON box array target column + image dirs."""
    pub = tmp / "prepared" / "public"
    priv = tmp / "prepared" / "private"
    (pub / "train").mkdir(parents=True)
    (pub / "test").mkdir(parents=True)
    priv.mkdir(parents=True)
    rows = [["image_id", "labels"]]
    for i in range(1, 8):
        box = '[{"utf8": "U+%04X", "x": 1, "y": 2, "w": 4, "h": 5}]' % (0x30 + i)
        rows.append(["img%02d" % i, box])
    _write_csv(pub / "train.csv", rows)
    _write_csv(pub / "sample_submission.csv",
               [["image_id", "labels"]] +
               [["t%02d" % i, "U+003F 1 1"] for i in range(1, 5)])
    _write_csv(priv / "test.csv", [["image_id"]] + [["t%02d" % i] for i in range(1, 5)])
    return pub


def _make_audio_dataset(tmp):
    """tensorflow-speech shape: NO train/test table, labels in dirs."""
    pub = tmp / "prepared" / "public"
    priv = tmp / "prepared" / "private"
    for label in ("yes", "no", "silence"):
        (pub / "train" / "audio" / label).mkdir(parents=True)
    (pub / "test" / "audio").mkdir(parents=True)
    priv.mkdir(parents=True)
    for label in ("yes", "no", "silence"):
        for i in range(20):
            (pub / "train" / "audio" / label / ("%s_%02d.wav" % (label, i))).write_bytes(b"RIFF" + b"\x00" * 8)
    for i in range(12):
        (pub / "test" / "audio" / ("clip_%08d.wav" % i)).write_bytes(b"RIFF" + b"\x00" * 8)
    _write_csv(pub / "sample_submission.csv",
               [["fname", "label"]] + [["clip_%08d.wav" % i, "silence"] for i in range(12)])
    _write_csv(priv / "test.csv",
               [["fname", "label"]] + [["clip_%08d.wav" % i, "yes" if i % 3 else "no"] for i in range(12)])
    return pub


def _run_harness(code, workdir, env):
    src = workdir / "main.py"
    src.write_text(code, encoding="utf-8")
    e = dict(os.environ)
    e["PYTHONIOENCODING"] = "utf-8"
    e.update(env)
    r = subprocess.run([sys.executable, str(src)], cwd=str(workdir),
                       env=e, capture_output=True, text=True, timeout=300)
    return r


# ---------------------------------------------------------------- 1-5 sniff
def test_analyzer_structural_sniffing():
    tmp = Path(tempfile.mkdtemp(prefix="v238_mask_"))
    try:
        pub = _make_mask_dataset(tmp)
        prof = Analyzer(str(tmp), "segment objects", sample_path=str(pub / "sample_submission.csv")).profile("mask_demo")
        check("rle: mask_target set", prof.mask_target == "rle_mask", prof.mask_target)
        check("rle: task_type segmentation", prof.task_type == "segmentation", prof.task_type)
        check("rle: modality image_mask", prof.modality == "image_mask", prof.modality)
        check("rle: bbox empty", prof.bbox_columns == [], str(prof.bbox_columns))

        pub2 = _make_detection_dataset(tmp / "det")
        prof2 = Analyzer(str(tmp / "det"), "find boxes", sample_path=str(pub2 / "sample_submission.csv")).profile("det_demo")
        check("bbox: columns detected", set(prof2.bbox_columns) == {"x_min", "y_min", "x_max", "y_max"}, str(prof2.bbox_columns))
        check("bbox: task_type detection", prof2.task_type == "detection", prof2.task_type)
        check("bbox: modality image_detection", prof2.modality == "image_detection", prof2.modality)
        check("bbox: multi-row target", prof2.multi_row_target is True, str(prof2.multi_row_target))

        pub3 = _make_json_box_dataset(tmp / "json")
        prof3 = Analyzer(str(tmp / "json"), "recognize characters", sample_path=str(pub3 / "sample_submission.csv")).profile("json_demo")
        check("json-box: column detected", "labels" in prof3.bbox_columns, str(prof3.bbox_columns))
        check("json-box: task_type detection", prof3.task_type == "detection", prof3.task_type)
        check("json-box: modality image_detection", prof3.modality == "image_detection", prof3.modality)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_analyzer_audio():
    tmp = Path(tempfile.mkdtemp(prefix="v238_audio_"))
    try:
        pub = _make_audio_dataset(tmp)
        prof = Analyzer(str(tmp), "classify spoken words", sample_path=str(pub / "sample_submission.csv")).profile("audio_demo")
        check("audio: file count >= threshold", prof.audio_file_count >= 50, str(prof.audio_file_count))
        check("audio: modality audio", prof.modality == "audio", prof.modality)
        check("audio: task_type classification", prof.task_type == "classification", prof.task_type)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- 6 synthesis
def test_layout_synthesis():
    tmp = Path(tempfile.mkdtemp(prefix="v238_synth_"))
    try:
        pub = _make_audio_dataset(tmp)
        # dir-label train table + id-only test table written by the resolver
        layout = resolve_dataset_layout(str(tmp), sample_path=str(pub / "sample_submission.csv"))
        check("synth: dir-label train.csv exists", (pub / "train.csv").is_file(), "")
        rows = list(csv.reader(io.open(str(pub / "train.csv"), encoding="utf-8")))
        check("synth: dir-label rows", len(rows) == 61 and rows[0] == ["fname", "label"], str(len(rows)))
        # tensorflow-speech shape: the PRIVATE test table resolves (labels
        # live there); no public test.csv synthesis is needed.
        check("synth: private test resolves", layout.test_path is not None
              and "private" in str(layout.test_path), str(layout.test_path))

        # no-private-test shape: sample-only ids -> id-only public test.csv
        tmp1 = tmp / "no_priv"
        pub1 = _make_audio_dataset(tmp1)
        (pub1.parent / "private" / "test.csv").unlink()
        layout1 = resolve_dataset_layout(str(tmp1), sample_path=str(pub1 / "sample_submission.csv"))
        check("synth: id-only test.csv written", (pub1 / "test.csv").is_file(), "")
        if (pub1 / "test.csv").is_file():
            trows = list(csv.reader(io.open(str(pub1 / "test.csv"), encoding="utf-8")))
            check("synth: test header id-only", trows[0] == ["fname"] and len(trows) == 13, str(trows[:2]))

        # mask shape: sample-copy synthesis (no dir labels)
        tmp2 = tmp / "mask_nolabels"
        pub2 = _make_mask_dataset(tmp2)
        # remove the train table to force synthesis
        (pub2 / "train.csv").unlink()
        (pub2 / "train").rmdir()
        (pub2 / "test").rmdir()
        layout2 = resolve_dataset_layout(str(tmp2), sample_path=str(pub2 / "sample_submission.csv"))
        check("synth: sample-copy train.csv", (pub2 / "train.csv").is_file(), "")
        rows2 = list(csv.reader(io.open(str(pub2 / "train.csv"), encoding="utf-8")))
        check("synth: sample-copy header", rows2[0] == ["id", "rle_mask"] and len(rows2) == 6, str(rows2[:2]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- 7 metrics
def test_metrics_inference():
    s = infer_metric_spec("segmentation")
    check("metrics: segmentation -> dice", s["metric_name"] == "dice", str(s))
    d = infer_metric_spec("detection")
    check("metrics: detection -> map_at_k", d["metric_name"] == "map_at_k", str(d))
    r = infer_metric_spec("regression")
    check("metrics: regression unchanged", r["metric_name"] == "rmse", str(r))


# ---------------------------------------------------------------- 8 registry/compiler
def test_registry_and_compiler():
    reg = CapabilityRegistry()
    for mid, renderer, mod, task in (
            ("image.mask.rle.baseline.v1", "image_mask_rle_baseline", "image_mask", "segmentation"),
            ("image.detection.bbox.baseline.v1", "image_detection_bbox_baseline", "image_detection", "detection"),
            ("audio.tabular.baseline.v1", "audio_tabular_baseline", "audio", "classification")):
        spec = reg.get(mid)
        check("registry: %s exists" % mid, spec is not None and not spec.broken, "")
        check("registry: %s renderer" % mid, spec is not None and spec.renderer == renderer, str(getattr(spec, "renderer", None)))
        check("registry: %s gpu=False" % mid, spec is not None and not spec.gpu, "")
        check("registry: %s modality" % mid, spec is not None and mod in (spec.supported_modalities or []), "")
    compiler = ProgramCompiler(reg)
    prof = AnalysisProfile(competition="demo", modality="image_mask",
                           task_type="segmentation", metric_name="dice")
    inv = MethodInvocationV1(method_id="image.mask.rle.baseline.v1", params={})
    norm = compiler.normalize(inv)
    check("compiler: mask validation single_holdout", norm.validation == "single_holdout", norm.validation)
    check("compiler: mask preprocessing empty", norm.preprocessing == [], str(norm.preprocessing))
    code, th = compiler.render(inv, profile=prof)
    check("compiler: mask render", "submission.csv" in code and "MASK_TARGET" in code, th)
    prof2 = AnalysisProfile(competition="demo", modality="image_detection",
                            task_type="detection", metric_name="map_at_k")
    code2, _ = compiler.render(MethodInvocationV1(method_id="image.detection.bbox.baseline.v1", params={}), profile=prof2)
    check("compiler: detection render", "BBOX_COLUMNS" in code2 and "map_at_k" in code2, "")
    prof3 = AnalysisProfile(competition="demo", modality="audio",
                            task_type="classification", metric_name="accuracy")
    code3, _ = compiler.render(MethodInvocationV1(method_id="audio.tabular.baseline.v1", params={}), profile=prof3)
    check("compiler: audio render", "majority" in code3 and "SAMPLE_SUB" in code3, "")
    # wrong modality must be rejected
    ok, reason = compiler.validate(MethodInvocationV1(method_id="image.mask.rle.baseline.v1", params={}), profile=prof3)
    check("compiler: mask rejected for audio", not ok, reason)


# ---------------------------------------------------------------- 9 harness execution
def test_harness_execution():
    tmp = Path(tempfile.mkdtemp(prefix="v238_exec_"))
    try:
        reg = CapabilityRegistry()
        compiler = ProgramCompiler(reg)

        # mask
        pub = _make_mask_dataset(tmp / "mask")
        prof = Analyzer(str(tmp / "mask"), "segment", sample_path=str(pub / "sample_submission.csv")).profile("m")
        code, _ = compiler.render(MethodInvocationV1(method_id="image.mask.rle.baseline.v1", params={}), profile=prof)
        wd = tmp / "mask_wd"
        wd.mkdir()
        env = {"TRAIN_CSV": str(pub / "train.csv"),
               "SAMPLE_SUBMISSION": str(pub / "sample_submission.csv"),
               "MASK_TARGET": "rle_mask", "TARGET_COLUMN": "rle_mask"}
        r = _run_harness(code, wd, env)
        check("mask: harness exit 0", r.returncode == 0, (r.stdout or "")[-400:] + (r.stderr or "")[-400:])
        check("mask: oof written", (wd / "oof.csv").is_file(), "")
        check("mask: submission written", (wd / "submission.csv").is_file(), "")
        if (wd / "submission.csv").is_file():
            srows = list(csv.reader(io.open(str(wd / "submission.csv"), encoding="utf-8")))
            check("mask: submission rows", len(srows) == 6 and srows[0][1] == "rle_mask", str(srows[:2]))
        check("mask: metric printed", "dice:" in (r.stdout or ""), "")

        # detection
        pub2 = _make_detection_dataset(tmp / "det")
        prof2 = Analyzer(str(tmp / "det"), "find boxes", sample_path=str(pub2 / "sample_submission.csv")).profile("d")
        code2, _ = compiler.render(MethodInvocationV1(method_id="image.detection.bbox.baseline.v1", params={}), profile=prof2)
        wd2 = tmp / "det_wd"
        wd2.mkdir()
        env2 = {"TRAIN_CSV": str(pub2 / "train.csv"),
                "SAMPLE_SUBMISSION": str(pub2 / "sample_submission.csv"),
                "TARGET_COLUMN": "y_max", "BBOX_COLUMNS": "[\"x_min\", \"y_min\", \"x_max\", \"y_max\"]",
                "TASK_TYPE": "detection"}
        r2 = _run_harness(code2, wd2, env2)
        check("det: harness exit 0", r2.returncode == 0, (r2.stdout or "")[-400:] + (r2.stderr or "")[-400:])
        check("det: oof written", (wd2 / "oof.csv").is_file(), "")
        check("det: submission constant placeholder", (wd2 / "submission.csv").is_file(), "")
        if (wd2 / "submission.csv").is_file():
            drow = list(csv.reader(io.open(str(wd2 / "submission.csv"), encoding="utf-8")))[1]
            check("det: submission = 14 1 0 0 1 1", drow[1] == "14 1 0 0 1 1", str(drow))
        check("det: map_at_k printed", "map_at_k:" in (r2.stdout or ""), "")

        # audio (single-label, dir-label synthesized table)
        pub3 = _make_audio_dataset(tmp / "audio")
        layout3 = resolve_dataset_layout(str(tmp / "audio"), sample_path=str(pub3 / "sample_submission.csv"))
        prof3 = Analyzer(str(tmp / "audio"), "words", sample_path=str(pub3 / "sample_submission.csv")).profile("a")
        code3, _ = compiler.render(MethodInvocationV1(method_id="audio.tabular.baseline.v1", params={}), profile=prof3)
        wd3 = tmp / "audio_wd"
        wd3.mkdir()
        env3 = {"TRAIN_CSV": str(pub3 / "train.csv"),
                "SAMPLE_SUBMISSION": str(pub3 / "sample_submission.csv"),
                "TARGET_COLUMN": "label"}
        r3 = _run_harness(code3, wd3, env3)
        check("audio: harness exit 0", r3.returncode == 0, (r3.stdout or "")[-400:] + (r3.stderr or "")[-400:])
        check("audio: oof written", (wd3 / "oof.csv").is_file(), "")
        check("audio: submission written", (wd3 / "submission.csv").is_file(), "")
        if (wd3 / "submission.csv").is_file():
            arow = list(csv.reader(io.open(str(wd3 / "submission.csv"), encoding="utf-8")))[1]
            check("audio: majority label", arow[1] in ("yes", "no", "silence"), str(arow))
        check("audio: accuracy printed", "accuracy:" in (r3.stdout or ""), "")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_analyzer_structural_sniffing()
    test_analyzer_audio()
    test_layout_synthesis()
    test_metrics_inference()
    test_registry_and_compiler()
    test_harness_execution()
    if FAILURES:
        print("RESULT=FAIL:%s" % ",".join(FAILURES))
        sys.exit(1)
    print("RESULT=PASS")