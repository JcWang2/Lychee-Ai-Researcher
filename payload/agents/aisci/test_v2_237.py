# -*- coding: utf-8 -*-
"""v2.3.7 offline tests: paired-image pixel regression (generic).

Covers the denoising-dirty-documents failure mode as a GENERIC layout
class - never competition-specific:
  1) paired-image layout detection: train input dir + sibling TARGET dir
     with matching stems + per-pixel sample submission (id=stem_r_c,
     value 0..1) -> pixel_level layout with synthesized public/train.csv
     (one row per sampled pixel, value = grayscale intensity / 255);
  2) stdlib PNG decoder (gray + RGB luma, all filter types) - no PIL
     dependency on the host;
  3) deterministic row-cap stride sampling (V2_PIXEL_ROW_CAP) covering
     every image;
  4) streaming sanitize of huge per-pixel private tables (no full-table
     RAM load), target column dropped;
  5) Analyzer: modality=image_pixel, task_type=regression (layout wins;
     plus a generic full-statistics fallback when the 30-row sample looks
     constant);
  6) capability registry: image.pixel.baseline.v1 is the ONLY compatible
     method for image_pixel/regression/rmse; timm templates are rejected;
     render produces a runnable harness with image-level holdout OOF +
     per-pixel submission (row count == sample submission);
  7) executed harness end-to-end: oof.csv + submission.csv correct,
     metric printed, per-image mean baseline matches the data.

Run: python test_v2_237.py   (from the aisci payload dir)
"""
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

from capability_registry import BUILTIN_SPECS, CapabilityRegistry
from data_layout import (DatasetLayoutError, resolve_dataset_layout,
                         sanitize_test_csv, synthesize_train_labels,
                         _decode_png_gray, _png_dims)
from hera.analyzer import Analyzer
from program_compiler import ProgramCompiler
from v2_contracts import AnalysisProfile, MethodInvocationV1

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("[OK] " + name)
    else:
        print("[FAIL] " + name + (" | " + str(detail)[:300] if detail else ""))
        FAILURES.append(name)


def _png_bytes(w, h, gray_vals, color_type=0):
    """Build a valid PNG (stdlib only): gray (0) or RGB (2), bit depth 8."""
    def chunk(typ, data):
        c = struct.pack(">I", len(data)) + typ + data
        return c + struct.pack(">I", zlib.crc32(typ + data) & 0xffffffff)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    if color_type == 0:
        bpp = 1
        pixels = bytes(gray_vals)
    else:
        bpp = 3
        pixels = b"".join(bytes((v, v, v)) for v in gray_vals)
    raw = b"".join(b"\x00" + pixels[y * w * bpp:(y + 1) * w * bpp]
                   for y in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def _write_png(path, w, h, gray_vals, color_type=0):
    Path(path).write_bytes(_png_bytes(w, h, gray_vals, color_type))


def _make_pixel_dataset(tmp: Path, n_train=12, w=8, h=6, cap_env=None):
    pub = tmp / "prepared" / "public"
    priv = tmp / "prepared" / "private"
    for d in ("train", "train_cleaned", "test"):
        (pub / d).mkdir(parents=True, exist_ok=True)
    priv.mkdir(parents=True, exist_ok=True)
    for n in range(1, n_train + 1):
        _write_png(pub / "train" / ("img%d.png" % n), w, h,
                   [20 + 17 * n] * (w * h))
        _write_png(pub / "train_cleaned" / ("img%d.png" % n), w, h,
                   [200 - 11 * n] * (w * h))
    _write_png(pub / "test" / "img101.png", w, h, [0] * (w * h))
    rows = [["id", "value"]]
    for n in list(range(1, n_train + 1)) + [101]:
        for r in range(1, h + 1):
            for c in range(1, w + 1):
                rows.append(["img%d_%d_%d" % (n, r, c), "1"])
    import csv
    with open(pub / "sampleSubmission.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    with open(priv / "answers.csv", "w", newline="") as fh:
        w_csv = csv.writer(fh)
        w_csv.writerow(["id", "value"])
        for r in range(1, h + 1):
            for c in range(1, w + 1):
                w_csv.writerow(["img101_%d_%d" % (r, c), "0.4"])
    return pub, priv


def test_pixel_layout_synthesis_and_analysis():
    """denoising shape: paired target dir + per-pixel sample -> layout,
    synthesis (gray/255, 1-based ids), streaming sanitize, analyzer."""
    import pandas as pd
    tmp = Path(tempfile.mkdtemp(prefix="v237_pix_"))
    try:
        pub, priv = _make_pixel_dataset(tmp)
        syn = synthesize_train_labels(tmp)
        check("pixel: synthesis mode paired-image",
              syn.get("mode") == "paired-image", str(syn))
        check("pixel: all images covered",
              int(syn.get("images") or 0) == 12, str(syn))
        train = pd.read_csv(pub / "train.csv")
        check("pixel: train header id,value",
              list(train.columns) == ["id", "value"], list(train.columns))
        check("pixel: row count = 12 images x 48 px",
              len(train) == 576, len(train))
        check("pixel: id is 1-based stem_r_c",
              str(train["id"].iloc[0]) == "img1_1_1", train["id"].iloc[0])
        # img1 target gray = 200-11 = 189 -> 189/255
        check("pixel: value = gray/255",
              abs(float(train["value"].iloc[0]) - 189.0 / 255.0) < 1e-6,
              train["value"].iloc[0])
        san = sanitize_test_csv(tmp)
        check("pixel: sanitize wrote label-free test",
              bool(san.get("written")), str(san))
        test = pd.read_csv(pub / "test.csv")
        check("pixel: test has id only",
              list(test.columns) == ["id"] and len(test) == 48,
              (list(test.columns), len(test)))
        layout = resolve_dataset_layout(tmp)
        check("pixel: layout pixel_level",
              layout.pixel_level is True, layout.layout_name)
        check("pixel: paired target dir found",
              layout.paired_target_dir is not None and
              layout.paired_target_dir.name == "train_cleaned",
              str(layout.paired_target_dir))
        prof = Analyzer(str(tmp), task_prompt=(
            "Denoising Dirty Documents: RMSE between cleaned pixel "
            "intensities and actual grayscale intensities (0..1).")
        ).profile("denoising-dirty-documents")
        check("pixel: modality=image_pixel", prof.modality == "image_pixel",
              prof.modality)
        check("pixel: task_type=regression",
              prof.task_type == "regression", prof.task_type)
        check("pixel: target=value", prof.target_column == "value",
              prof.target_column)
        check("pixel: no classes", prof.n_classes == 0, prof.n_classes)
        # idempotency
        syn2 = synthesize_train_labels(tmp)
        check("pixel: idempotent", syn2.get("skipped") == "has-train-table",
              str(syn2))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_png_decoder_gray_and_rgb():
    tmp = Path(tempfile.mkdtemp(prefix="v237_png_"))
    try:
        p = tmp / "gray.png"
        vals = list(range(6))
        _write_png(p, 3, 2, vals, color_type=0)
        check("png: dims", _png_dims(p) == (3, 2), _png_dims(p))
        dec = _decode_png_gray(p)
        check("png: gray decode", dec == [float(v) for v in vals], dec)
        p2 = tmp / "rgb.png"
        _write_png(p2, 3, 2, vals, color_type=2)
        dec2 = _decode_png_gray(p2)
        # luma of (v,v,v) == v
        check("png: rgb luma decode",
              dec2 is not None and abs(dec2[0] - 0.0) < 1e-9 and
              abs(dec2[5] - 5.0) < 1e-9, dec2)
        check("png: not a png -> None",
              _decode_png_gray(tmp / "missing.png") is None)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_pixel_row_cap_stride():
    import pandas as pd
    tmp = Path(tempfile.mkdtemp(prefix="v237_cap_"))
    old = os.environ.get("V2_PIXEL_ROW_CAP")
    try:
        pub, priv = _make_pixel_dataset(tmp, n_train=12, w=8, h=6)
        os.environ["V2_PIXEL_ROW_CAP"] = "60"
        syn = synthesize_train_labels(tmp)
        check("cap: stride applied", int(syn.get("stride") or 0) >= 2,
              str(syn))
        train = pd.read_csv(pub / "train.csv")
        check("cap: rows bounded", 0 < len(train) <= 60, len(train))
        stems = set(str(x).rsplit("_", 2)[0] for x in train["id"])
        check("cap: every image covered", len(stems) == 12, len(stems))
    finally:
        if old is None:
            os.environ.pop("V2_PIXEL_ROW_CAP", None)
        else:
            os.environ["V2_PIXEL_ROW_CAP"] = old
        shutil.rmtree(tmp, ignore_errors=True)


def test_pixel_capability_and_render():
    reg = CapabilityRegistry([s for s in BUILTIN_SPECS])
    comp = ProgramCompiler(reg)
    prof = AnalysisProfile(competition="denoising-dirty-documents",
                           modality="image_pixel", task_type="regression",
                           train_rows=576, test_rows=48,
                           feature_columns=["id", "value"],
                           target_column="value", feature_dim=2,
                           image_width=8, image_height=6,
                           metric_name="rmse")
    check("registry: pixel method exists",
          reg.get("image.pixel.baseline.v1") is not None)
    compat = [s.method_id for s in
              reg.compatible("image_pixel", "regression", "rmse")]
    check("registry: only pixel method compatible",
          compat == ["image.pixel.baseline.v1"], compat)
    check("registry: timm excluded for image_pixel",
          all(not m.startswith("image.embedding") and
              not m.startswith("image.finetune") for m in compat), compat)
    inv = MethodInvocationV1(method_id="image.pixel.baseline.v1",
                             params={"basis": "per_image", "val_seed": 7},
                             hypothesis="pixel mean")
    ok, reason = comp.validate(inv, prof, None)
    check("compiler: pixel invocation valid", ok, reason)
    code, th = comp.render(inv, prof, None)
    check("compiler: rendered harness",
          "submission.csv" in code and "image-level holdout" in code and
          th.startswith("sha256:"), th[:24])
    # defaults: no preprocessing, single_holdout forced
    norm = comp.normalize(MethodInvocationV1(
        method_id="image.pixel.baseline.v1", hypothesis="d"))
    check("compiler: defaults safe",
          norm.preprocessing == [] and norm.validation == "single_holdout",
          (norm.preprocessing, norm.validation))
    check("compiler: params defaulted",
          norm.params.get("basis") == "per_image", norm.params)
    # invalid basis rejected
    bad = MethodInvocationV1(method_id="image.pixel.baseline.v1",
                             params={"basis": "magic"},
                             hypothesis="bad")
    ok2, reason2 = comp.validate(bad, prof, None)
    check("compiler: bad param rejected", not ok2, reason2)


def test_pixel_harness_execution():
    """Run the rendered harness end-to-end in a subprocess."""
    import csv as _csv
    tmp = Path(tempfile.mkdtemp(prefix="v237_run_"))
    try:
        pub, priv = _make_pixel_dataset(tmp, n_train=12, w=8, h=6)
        synthesize_train_labels(tmp)
        sanitize_test_csv(tmp)
        reg = CapabilityRegistry([s for s in BUILTIN_SPECS])
        comp = ProgramCompiler(reg)
        prof = AnalysisProfile(competition="denoising-dirty-documents",
                               modality="image_pixel",
                               task_type="regression",
                               train_rows=576, test_rows=48,
                               feature_columns=["id", "value"],
                               target_column="value", feature_dim=2,
                               image_width=8, image_height=6,
                               metric_name="rmse")
        inv = MethodInvocationV1(method_id="image.pixel.baseline.v1",
                                 params={"basis": "per_image",
                                         "val_seed": 42},
                                 validation="single_holdout",
                                 hypothesis="baseline")
        code, _th = comp.render(inv, prof, None)
        work = tmp / "work"
        work.mkdir()
        env = dict(os.environ)
        env.update({
            "TRAIN_CSV": str(pub / "train.csv"),
            "TEST_CSV": str(pub / "test.csv"),
            "SAMPLE_SUBMISSION": str(pub / "sampleSubmission.csv"),
            "TARGET_COLUMN": "value",
            "TASK_TYPE": "regression",
        })
        code_path = work / "main.py"
        code_path.write_text(code, encoding="utf-8")
        proc = subprocess.run([sys.executable, str(code_path)],
                              cwd=str(work), env=env, capture_output=True,
                              text=True, timeout=180)
        check("harness: rc=0", proc.returncode == 0,
              (proc.stderr or proc.stdout)[-400:])
        out = proc.stdout or ""
        check("harness: oof written", (work / "oof.csv").is_file())
        sub_rows = sum(1 for _ in (work / "submission.csv").open(
            newline="", encoding="utf-8")) - 1
        with open(pub / "sampleSubmission.csv", newline="",
                  encoding="utf-8") as fh:
            sample_rows = len(list(_csv.reader(fh))) - 1
        check("harness: submission row count == sample",
              sub_rows == sample_rows, (sub_rows, sample_rows))
        check("harness: metric printed",
              "rmse" in out and "=" in out, out[-300:])
        # per-image mean correctness: img1 clean gray=189 -> 0.741176
        with open(work / "submission.csv", newline="",
                  encoding="utf-8") as fh:
            first = next(_csv.reader(fh))
            second = next(_csv.reader(fh))
        check("harness: submission header id,value",
              first == ["id", "value"], first)
        check("harness: per-image mean", abs(
            float(second[1]) - 189.0 / 255.0) < 1e-5, second)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_stats_based_task_type_fallback():
    """A tabular target whose first 30 rows are constant must still be
    detected as regression via the full measured statistics."""
    import pandas as pd
    tmp = Path(tempfile.mkdtemp(prefix="v237_stats_"))
    try:
        pub = tmp / "prepared" / "public"
        priv = tmp / "prepared" / "private"
        pub.mkdir(parents=True)
        priv.mkdir(parents=True)
        n = 400
        vals = [1.0] * 40 + [float(2 + (i % 12)) for i in range(40, n)]
        pd.DataFrame({"idx": range(n), "y": vals}).to_csv(
            pub / "train.csv", index=False)
        pd.DataFrame({"idx": range(20), "y": 0.0}).to_csv(
            pub / "test.csv", index=False)
        pd.DataFrame({"idx": range(20), "y": 0.0}).to_csv(
            pub / "sample_submission.csv", index=False)
        prof = Analyzer(str(tmp), task_prompt=(
            "predict a continuous target value")).profile("v237_stub")
        check("stats: regression via full stats",
              prof.task_type == "regression", prof.task_type)
        check("stats: target y", prof.target_column == "y",
              prof.target_column)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_pixel_missing_synthesis_guard():
    """Analyzing a pixel layout before synthesis raises a clear error."""
    tmp = Path(tempfile.mkdtemp(prefix="v237_guard_"))
    try:
        pub, priv = _make_pixel_dataset(tmp)
        try:
            Analyzer(str(tmp)).profile("x")
            check("guard: raises", False, "no exception")
        except DatasetLayoutError as e:
            check("guard: actionable error",
                  "synthesize_train_labels" in str(e), str(e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    test_pixel_layout_synthesis_and_analysis()
    test_png_decoder_gray_and_rgb()
    test_pixel_row_cap_stride()
    test_pixel_capability_and_render()
    test_pixel_harness_execution()
    test_stats_based_task_type_fallback()
    test_pixel_missing_synthesis_guard()
    if FAILURES:
        print("RESULT=FAIL:%s" % ",".join(FAILURES))
        return 1
    print("RESULT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
