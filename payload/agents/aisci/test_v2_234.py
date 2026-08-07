# -*- coding: utf-8 -*-
"""v2.3.4 offline tests: multi-output regression (nomad2018-style),
sample-submission-driven target correction (NYC-taxi labels.csv), and
text multi-class probability submissions (spooky-author-style).

Run: python test_v2_234.py   (from the aisci payload dir)
"""
import os, shutil, subprocess, sys, tempfile, types
from pathlib import Path

from v2_contracts import MethodInvocationV1
from program_compiler import ProgramCompiler
from capability_registry import CapabilityRegistry, load_ephemeral_path
from hera.analyzer import Analyzer

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("[OK] " + name)
    else:
        print("[FAIL] " + name + (" | " + str(detail)[:300] if detail else ""))
        FAILURES.append(name)


def _inv(method_id, params=None):
    return MethodInvocationV1.from_dict({
        "method_id": method_id,
        "params": params or {},
        "preprocessing": [],
        "validation": "single_holdout",
        "hypothesis": "v2.3.4 render test",
    })


def _render(inv, profile, manifest):
    reg = CapabilityRegistry(ephemeral_path=load_ephemeral_path(""))
    code, th = ProgramCompiler(reg).render(inv, profile, manifest)
    assert th.startswith("sha256:"), th
    return code, th


def _run(code, workdir, env):
    src = workdir / "rendered_code.py"
    src.write_text(code, encoding="utf-8")
    e = dict(os.environ)
    e.update(env)
    return subprocess.run([sys.executable, str(src)], cwd=str(workdir),
                          env=e, capture_output=True, text=True, timeout=600)


def test_multi_output_regression():
    """nomad2018 shape: sample_submission names TWO train targets."""
    import numpy as np, pandas as pd
    tmp = Path(tempfile.mkdtemp(prefix="v234_nomad_"))
    try:
        rng = np.random.RandomState(7)
        n_tr, n_te = 120, 40
        tr = pd.DataFrame({
            "id": range(1, n_tr + 1),
            "spacegroup": rng.choice([1, 2, 3, 4], n_tr),
            "label": rng.choice(["a", "b", "c"], n_tr),
            "formula": ["Al2O3"] * n_tr,
            "formation_energy_ev_natom": rng.randn(n_tr) + 1.0,
            "bandgap_energy_ev": rng.randn(n_tr) * 0.5 + 2.0,
        })
        te = tr.iloc[:n_te].drop(columns=["formation_energy_ev_natom",
                                          "bandgap_energy_ev"])
        tr.to_csv(tmp / "train.csv", index=False)
        te.to_csv(tmp / "test.csv", index=False)
        pd.DataFrame({
            "id": te["id"],
            "formation_energy_ev_natom": 0.17,
            "bandgap_energy_ev": 1.88,
        }).to_csv(tmp / "sample_submission.csv", index=False)

        profile = types.SimpleNamespace(
            modality="tabular", task_type="regression",
            target_column="bandgap_energy_ev", text_columns=[], time_column="",
            train_rows=n_tr, test_rows=n_te, n_classes=0, feature_dim=5)
        manifest = {
            "sample_submission_header": ["id", "formation_energy_ev_natom",
                                         "bandgap_energy_ev"],
            "target_column": "bandgap_energy_ev", "modality": "tabular",
            "text_columns": [], "time_column": ""}
        inv = _inv("tabular.gbdt.histgb.v1",
                   {"learning_rate": 0.1, "max_iter": 60,
                    "max_leaf_nodes": 8, "folds": 2})
        code, th = _render(inv, profile, manifest)
        check("multi-output: render", "MULTI_TARGET" in code, th)
        r = _run(code, tmp, {
            "TASK_TYPE": "regression",
            "TARGET_COLUMN": "bandgap_energy_ev",
            "SAMPLE_SUBMISSION": "sample_submission.csv"})
        check("multi-output: run rc=0", r.returncode == 0,
              (r.stdout + r.stderr)[-500:])
        if r.returncode == 0:
            out = pd.read_csv(tmp / "submission.csv")
            check("multi-output: submission cols",
                  list(out.columns) == ["id", "formation_energy_ev_natom",
                                        "bandgap_energy_ev"],
                  list(out.columns))
            check("multi-output: row count", len(out) == n_te)
            v1 = pd.to_numeric(out["formation_energy_ev_natom"],
                               errors="coerce")
            v2 = pd.to_numeric(out["bandgap_energy_ev"], errors="coerce")
            check("multi-output: numeric", v1.notna().all() and v2.notna().all())
            check("multi-output: not 0/1 labels", (v1.abs() > 1).any(),
                  v1.head(3).tolist())
            check("multi-output: distinct columns",
                  not (v1.round(3) == v2.round(3)).all())
        else:
            print(r.stdout[-1200:])
            print(r.stderr[-1200:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_sample_driven_target_analyzer():
    """NYC taxi shape: labels.csv ends with passenger_count; the sample
    submission names fare_amount. Analyzer must pick fare_amount and call
    it regression (not a 10-class classification)."""
    import numpy as np, pandas as pd
    tmp = Path(tempfile.mkdtemp(prefix="v234_taxi_"))
    try:
        rng = np.random.RandomState(11)
        n_tr, n_te = 150, 40
        tr = pd.DataFrame({
            "key": ["t%d" % i for i in range(n_tr)],
            "fare_amount": rng.uniform(3, 40, n_tr).round(2),
            "pickup_datetime": ["2015-01-01 00:%02d:00" % (i % 60)
                                for i in range(n_tr)],
            "pickup_longitude": rng.uniform(-74.2, -73.9, n_tr).round(5),
            "pickup_latitude": rng.uniform(40.5, 40.9, n_tr).round(5),
            "dropoff_longitude": rng.uniform(-74.2, -73.9, n_tr).round(5),
            "dropoff_latitude": rng.uniform(40.5, 40.9, n_tr).round(5),
            "passenger_count": rng.choice([1, 2, 3, 4], n_tr),
        })
        te = tr.iloc[:n_te].drop(columns=["fare_amount"])
        tr.to_csv(tmp / "labels.csv", index=False)
        te.to_csv(tmp / "test.csv", index=False)
        pd.DataFrame({"key": te["key"], "fare_amount": 11.35}).to_csv(
            tmp / "sample_submission.csv", index=False)

        prof = Analyzer(str(tmp), task_prompt="NYC taxi fare regression").profile("stub_taxi")
        check("taxi: analyzer target=fare_amount",
              prof.target_column == "fare_amount", prof.target_column)
        check("taxi: analyzer task_type=regression",
              prof.task_type == "regression", prof.task_type)
        check("taxi: analyzer modality=tabular",
              prof.modality == "tabular", prof.modality)

        profile = types.SimpleNamespace(
            modality=prof.modality, task_type=prof.task_type,
            target_column="passenger_count",  # deliberately stale value
            text_columns=[], time_column="",
            train_rows=n_tr, test_rows=n_te, n_classes=0, feature_dim=7)
        manifest = {
            "sample_submission_header": ["key", "fare_amount"],
            "target_column": "passenger_count", "modality": "tabular",
            "text_columns": [], "time_column": ""}
        inv = _inv("tabular.linear.logistic.v1",
                   {"C": 1.0, "max_iter": 300, "folds": 2})
        code, th = _render(inv, profile, manifest)
        r = _run(code, tmp, {
            "TASK_TYPE": "regression",
            "TARGET_COLUMN": "passenger_count",
            "TRAIN_CSV": "labels.csv",
            "SAMPLE_SUBMISSION": "sample_submission.csv"})
        check("taxi: run rc=0", r.returncode == 0, (r.stdout + r.stderr)[-500:])
        if r.returncode == 0:
            out = pd.read_csv(tmp / "submission.csv")
            check("taxi: submission cols",
                  list(out.columns) == ["key", "fare_amount"],
                  list(out.columns))
            v = pd.to_numeric(out["fare_amount"], errors="coerce")
            check("taxi: numeric fare", v.notna().all())
            check("taxi: not passenger_count 0/1", (v > 1).all(),
                  v.head(5).tolist())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_text_multi_class_proba():
    """spooky shape: text column + 3-class probability submission."""
    import numpy as np, pandas as pd
    tmp = Path(tempfile.mkdtemp(prefix="v234_spooky_"))
    try:
        rng = np.random.RandomState(3)
        authors = ["EAP", "HPL", "MWS"]
        words = {"EAP": ["raven", "nevermore", "chamber"],
                 "HPL": ["cthulhu", "eldritch", "nyarlathotep"],
                 "MWS": ["frankenstein", "creature", "geneva"]}
        n_tr, n_te = 90, 30
        rows = []
        for i in range(n_tr):
            a = authors[i % 3]
            txt = " ".join(list(rng.choice(words[a], 6))
                           + ["the", "of", "and"])
            rows.append((i + 1, txt, a))
        tr = pd.DataFrame(rows, columns=["id", "text", "author"])
        te = tr.iloc[:n_te].drop(columns=["author"])
        tr.to_csv(tmp / "train.csv", index=False)
        te.to_csv(tmp / "test.csv", index=False)
        pd.DataFrame({"id": te["id"], "EAP": 0.4, "HPL": 0.3,
                      "MWS": 0.3}).to_csv(tmp / "sample_submission.csv",
                                          index=False)

        prof = Analyzer(str(tmp), task_prompt="spooky author classification").profile("stub_spooky")
        check("spooky: analyzer modality=text",
              prof.modality == "text", prof.modality)
        check("spooky: analyzer target=author",
              prof.target_column == "author", prof.target_column)

        profile = types.SimpleNamespace(
            modality="text", task_type="classification",
            target_column="author", text_columns=["text"], time_column="",
            train_rows=n_tr, test_rows=n_te, n_classes=3, feature_dim=1)
        manifest = {
            "sample_submission_header": ["id", "EAP", "HPL", "MWS"],
            "target_column": "author", "modality": "text",
            "text_columns": ["text"], "time_column": ""}
        inv = _inv("text.embedding.tfidf.v1",
                   {"C": 1.0, "max_iter": 300, "folds": 2})
        code, th = _render(inv, profile, manifest)
        check("spooky: render ok", th.startswith("sha256:"), th)
        r = _run(code, tmp, {
            "TASK_TYPE": "classification",
            "TARGET_COLUMN": "author",
            "SAMPLE_SUBMISSION": "sample_submission.csv"})
        check("spooky: run rc=0", r.returncode == 0, (r.stdout + r.stderr)[-500:])
        if r.returncode == 0:
            out = pd.read_csv(tmp / "submission.csv")
            check("spooky: submission cols",
                  list(out.columns) == ["id", "EAP", "HPL", "MWS"],
                  list(out.columns))
            probs = out[["EAP", "HPL", "MWS"]].astype(float)
            check("spooky: probs in [0,1]",
                  ((probs >= 0) & (probs <= 1)).all().all())
            check("spooky: rows sum ~1",
                  (probs.sum(axis=1) - 1).abs().max() < 0.05)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_multi_output_regression()
    test_sample_driven_target_analyzer()
    test_text_multi_class_proba()
    if FAILURES:
        print("FAILURES=%d: %s" % (len(FAILURES), ", ".join(FAILURES)))
        sys.exit(1)
    print("ALL_V234_TESTS=PASS")