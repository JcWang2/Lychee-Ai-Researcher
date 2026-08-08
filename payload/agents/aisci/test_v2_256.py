# -*- coding: utf-8 -*-
"""v2.5.6 offline tests: row-cap alignment + string-target lookup chain.

1) the compiled tabular harness caps training rows WITHOUT corrupting the
   label vector (sklearn train_test_split returns (train, test); the
   row-cap block must keep the TRAIN side and must re-index y_fit
   unconditionally - not only for multi-target tasks). Regression labels
   were previously misaligned (taxi) and classification crashed (TPS-Dec);
2) the string-target capability is declared, renders a deterministic
   most-frequent source->target lookup with generic compound-id join
   synthesis (id = sentence_id_token_id style), and the analyzer detects
   high-cardinality string targets with a copy-source column from data
   alone - no competition names anywhere;
3) invocation runnability clamps fold counts to the platform budget;
4) the PACT host fallback writes per-row copy-source artifacts for string
   targets instead of a single majority string (text-norm EN baseline);
5) no competition-name hardcoding in the modified modules.
"""
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from capability_registry import CapabilityRegistry  # noqa: E402
from data_layout import resolve_dataset_layout  # noqa: E402
from hera.analyzer import Analyzer  # noqa: E402
from pact.deterministic import write_deterministic_artifacts  # noqa: E402
from program_compiler import ProgramCompiler  # noqa: E402
from v2_closed_loop import clamp_invocation_runnability  # noqa: E402
from v2_contracts import AnalysisProfile, MethodInvocationV1  # noqa: E402

PASS = 0
FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print("[OK] %s" % name)
    else:
        FAIL += 1
        print("[FAIL] %s %s" % (name, detail))


def _write(path: Path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _scrub_comments(src):
    return "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith("#"))


# ------------------------------------------------------------- row-cap fix
def test_row_cap_template_fix():
    src = (HERE / "program_compiler.py").read_text(encoding="utf-8")
    check("row-cap keeps TRAIN side (_sel, _)",
          "_sel, _ = train_test_split(np.arange(n), train_size="
          "int(MAX_TRAIN_ROWS)" in src, "")
    check("row-cap swap bug absent (_, _sel)",
          "_, _sel = train_test_split(np.arange(n), train_size="
          "int(MAX_TRAIN_ROWS)" not in src, "")
    n_y = src.count("y_fit = y_fit[sel_idx]")
    check("y_fit re-indexed after row cap (single occurrence)",
          n_y == 1, "count=%d" % n_y)
    i_cap = src.find("if MAX_TRAIN_ROWS > 0 and n > MAX_TRAIN_ROWS")
    i_y = src.find("y_fit = y_fit[sel_idx]")
    check("y_fit subsample lives inside the row-cap block",
          -1 < i_cap < i_y, "cap=%d y=%d" % (i_cap, i_y))


def _run_harness(code, tmp, target_column, task_type, timeout=None):
    if timeout is None:
        timeout = int(os.environ.get("V2_TEST_HARNESS_TIMEOUT", "1800"))
    code_path = tmp / "run.py"
    code_path.write_text(code, encoding="utf-8")
    env = dict(os.environ)
    env.update({
        "TRAIN_CSV": str(tmp / "train.csv"),
        "TEST_CSV": str(tmp / "test.csv"),
        "SAMPLE_SUBMISSION": str(tmp / "sample_submission.csv"),
        "TARGET_COLUMN": target_column,
        "TASK_TYPE": task_type,
    })
    proc = subprocess.run([sys.executable, str(code_path)], cwd=str(tmp),
                          env=env, capture_output=True, text=True,
                          timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError("harness failed rc=%s\n%s" % (
            proc.returncode, (proc.stdout + proc.stderr)[-3000:]))
    return proc


def test_row_cap_end_to_end():
    # Heavy subprocess gate mirrors test_v2_251/255: skipped under
    # V2_TEST_HARNESS_SKIP=1 (ops default under fleet load); validated in
    # full installs without the flag.
    if os.environ.get("V2_TEST_HARNESS_SKIP", "0") == "1":
        print("[SKIP] row-cap e2e subprocess (V2_TEST_HARNESS_SKIP=1)")
        return
    import numpy as np
    import pandas as pd
    tmp = Path(tempfile.mkdtemp(prefix="v256_rowcap_"))
    try:
        rng = np.random.RandomState(3)
        n, cap = 60000, 10000
        y = rng.choice(np.arange(1, 8), size=n,
                       p=[0.40, 0.15, 0.12, 0.10, 0.08, 0.08, 0.07])
        cols = {"f%02d" % i: rng.randn(n) + y * 0.05 for i in range(8)}
        df = pd.DataFrame(cols)
        df.insert(0, "Id", np.arange(1, n + 1))
        df["Cover_Type"] = y
        df.to_csv(str(tmp / "train.csv"), index=False)
        n_test = 3000
        xt = {"f%02d" % i: rng.randn(n_test) for i in range(8)}
        tdf = pd.DataFrame(xt)
        tdf.insert(0, "Id", np.arange(n + 1, n + 1 + n_test))
        tdf.to_csv(str(tmp / "test.csv"), index=False)
        pd.DataFrame({"Id": tdf["Id"].values,
                      "Cover_Type": 2}).to_csv(
            str(tmp / "sample_submission.csv"), index=False)
        reg = CapabilityRegistry()
        comp = ProgramCompiler(reg)
        prof = AnalysisProfile(competition="row-cap-stub",
                               task_type="classification",
                               modality="tabular",
                               target_column="Cover_Type",
                               metric_name="accuracy",
                               train_rows=n, n_classes=7, feature_dim=9)
        manifest = {"metric_name": "accuracy",
                    "metric_direction": "higher_is_better",
                    "metric_alignment": "exact",
                    "metric_label": "accuracy",
                    "metric_params": {},
                    "modality": "tabular",
                    "text_columns": [], "datetime_columns": [],
                    "time_column": "",
                    "sample_submission_header": ["Id", "Cover_Type"]}
        inv = MethodInvocationV1(
            method_id="tabular.gbdt.histgb.v1",
            params={"max_iter": 50, "learning_rate": 0.1},
            validation="single_holdout",
            resource_request={"max_train_rows": cap},
            hypothesis="row-cap")
        code, th = comp.render(inv, prof, manifest)
        check("row-cap e2e: render ok",
              bool(code) and th.startswith("sha256:"), th[:24])
        _run_harness(code, tmp, target_column="Cover_Type",
                     task_type="classification")
        with open(str(tmp / "oof.csv"), newline="") as fh:
            oof = list(csv.reader(fh))
        check("row-cap e2e: oof header true,pred",
              bool(oof) and oof[0] == ["true", "pred"],
              str(oof[0] if oof else None))
        check("row-cap e2e: oof rows capped (<= %d)" % cap,
              0 < len(oof) - 1 <= cap, "rows=%d" % (len(oof) - 1))
        with open(str(tmp / "submission.csv"), newline="") as fh:
            sub = list(csv.reader(fh))
        check("row-cap e2e: submission rows",
              len(sub) - 1 == n_test, str(len(sub) - 1))
        check("row-cap e2e: submission header",
              bool(sub) and sub[0] == ["Id", "Cover_Type"],
              str(sub[0] if sub else None))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------- string lookup chain
def _string_prof_manifest():
    prof = AnalysisProfile(competition="string-lookup-stub",
                           task_type="classification",
                           modality="tabular",
                           target_column="after",
                           metric_name="accuracy",
                           string_target=True,
                           string_source_column="before")
    manifest = {"metric_name": "accuracy",
                "metric_direction": "higher_is_better",
                "metric_alignment": "exact",
                "metric_label": "accuracy",
                "metric_params": {},
                "modality": "tabular",
                "text_columns": [], "datetime_columns": [],
                "time_column": "",
                "string_target": True,
                "string_source_column": "before",
                "sample_submission_header": ["id", "after"]}
    return prof, manifest


def test_string_lookup_declared():
    reg = CapabilityRegistry()
    spec = reg.get("text.string_lookup.v1")
    check("string lookup spec present", spec is not None, "")
    check("string lookup renderer wired",
          spec is not None and spec.renderer == "text_string_lookup", "")
    check("string lookup supports accuracy",
          spec is not None and "accuracy" in (spec.metric_outputs or {}), "")
    check("string lookup fixed holdout",
          spec is not None and spec.default_validation == "single_holdout", "")
    reg_src = _scrub_comments(
        (HERE / "capability_registry.py").read_text(encoding="utf-8"))
    check("registry code declares string lookup",
          "text.string_lookup.v1" in reg_src
          and "text_string_lookup" in reg_src, "")
    pc_src = _scrub_comments(
        (HERE / "program_compiler.py").read_text(encoding="utf-8"))
    check("compiler code has string lookup template",
          "_STRING_LOOKUP_TEMPLATE" in pc_src
          and '"text_string_lookup"' in pc_src, "")


def test_string_lookup_renders():
    reg = CapabilityRegistry()
    comp = ProgramCompiler(reg)
    prof, manifest = _string_prof_manifest()
    inv = MethodInvocationV1(method_id="text.string_lookup.v1",
                             params={"min_count": 1},
                             validation="single_holdout",
                             hypothesis="lookup")
    code, th = comp.render(inv, prof, manifest)
    check("string lookup render succeeds",
          code.startswith("# -*- coding"), th[:24])
    check("string lookup bakes source column",
          "SOURCE_COL = 'before'" in code, "")
    check("string lookup has compound-id synthesis",
          "SYN_COLS" in code and "sentence_id" in code
          and "token_id" in code, "")
    check("string lookup writes submission+oof",
          "submission.csv" in code and "oof.csv" in code, "")
    inv2 = MethodInvocationV1(method_id="tabular.gbdt.histgb.v1",
                              params={"max_iter": 50},
                              validation="single_holdout",
                              hypothesis="tabular")
    code2, th2 = comp.render(inv2, prof, manifest)
    check("tabular harness still renders for string profile",
          code2.startswith("# -*- coding"), th2[:24])


def test_string_lookup_end_to_end():
    # Heavy subprocess gate mirrors test_v2_251/255: skipped under
    # V2_TEST_HARNESS_SKIP=1 (ops default under fleet load).
    if os.environ.get("V2_TEST_HARNESS_SKIP", "0") == "1":
        print("[SKIP] string-lookup e2e subprocess (V2_TEST_HARNESS_SKIP=1)")
        return
    import random
    tmp = Path(tempfile.mkdtemp(prefix="v256_sl_"))
    try:
        random.seed(7)
        words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy"]
        train_rows, test_rows, sample_ids = [], [], []
        sid = 0
        for _s in range(40):
            for t in range(6):
                w = random.choice(words)
                cls = "PLAIN"
                after = w
                if random.random() < 0.15:
                    cls = "NUM"
                    w = str(random.randint(10, 99))
                    after = w
                train_rows.append((sid, t, cls, w, after))
                sample_ids.append("%d_%d" % (sid, t))
                test_rows.append((sid, t, w))
            sid += 1
        with open(str(tmp / "train.csv"), "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["sentence_id", "token_id", "class", "before",
                        "after"])
            w.writerows(train_rows)
        with open(str(tmp / "test.csv"), "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["sentence_id", "token_id", "before"])
            w.writerows(test_rows)
        with open(str(tmp / "sample_submission.csv"), "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "after"])
            for i, (_s, _t, b) in enumerate(test_rows):
                w.writerow([sample_ids[i], b])
        reg = CapabilityRegistry()
        comp = ProgramCompiler(reg)
        prof, manifest = _string_prof_manifest()
        inv = MethodInvocationV1(method_id="text.string_lookup.v1",
                                 params={"min_count": 1},
                                 validation="single_holdout",
                                 hypothesis="lookup")
        code, th = comp.render(inv, prof, manifest)
        check("string e2e: render ok", bool(code), th[:24])
        _run_harness(code, tmp, target_column="after",
                     task_type="classification")
        with open(str(tmp / "submission.csv"), newline="") as fh:
            sub = list(csv.reader(fh))
        check("string e2e: submission header",
              bool(sub) and sub[0] == ["id", "after"],
              str(sub[0] if sub else None))
        check("string e2e: submission row count",
              len(sub) - 1 == len(sample_ids), str(len(sub) - 1))
        check("string e2e: ids match sample",
              all(sub[i + 1][0] == sample_ids[i]
                  for i in range(len(sample_ids))), "")
        with open(str(tmp / "oof.csv"), newline="") as fh:
            oof = list(csv.reader(fh))
        acc = (sum(1 for r in oof[1:] if len(r) == 2 and r[0] == r[1])
               / float(max(1, len(oof) - 1)))
        check("string e2e: oof accuracy high",
              bool(oof) and oof[0] == ["true", "pred"] and acc > 0.9,
              "acc=%.3f" % acc)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _write_text_norm_like(tmp: Path, n_train=600, n_test=200):
    words = ["w%04d" % i for i in range(max(n_train, n_test))]
    rows = []
    for i in range(n_train):
        cls = "PLAIN" if i % 10 else "NUM"
        rows.append("%d,%d,%s,%s,%s" % (i % 20, i % 7, cls,
                                        words[i], words[i]))
    _write(tmp / "train.csv",
           ["sentence_id,token_id,class,before,after"] + rows)
    test_rows = ["%d,%d,%s" % (i % 20, i % 7, words[i])
                 for i in range(n_test)]
    _write(tmp / "test.csv",
           ["sentence_id,token_id,before"] + test_rows)
    sample_rows = ["%d_%d,%s" % (i % 20, i % 7, words[i])
                   for i in range(n_test)]
    _write(tmp / "sample_submission.csv", ["id,after"] + sample_rows)


def test_analyzer_string_target():
    tmp = Path(tempfile.mkdtemp(prefix="v256_strtgt_"))
    try:
        _write_text_norm_like(tmp)
        an = Analyzer(str(tmp), task_prompt=(
            "Text normalization: semiotic class + normalized token "
            "prediction (classification accuracy)"))
        prof = an.profile("string-target-stub")
        check("analyzer: target is after",
              prof.target_column == "after", prof.target_column)
        check("analyzer: task classification",
              prof.task_type == "classification", prof.task_type)
        check("analyzer: string target detected",
              bool(prof.string_target), str(prof.string_target))
        check("analyzer: source column before",
              prof.string_source_column == "before",
              prof.string_source_column)
        check("analyzer: notes mention string target",
              "string target" in prof.data_notes, "")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_clamp_invocation_runnability():
    reg = CapabilityRegistry()
    inv = MethodInvocationV1(method_id="tabular.gbdt.histgb.v1",
                             params={"folds": 5},
                             validation="single_holdout")
    out = clamp_invocation_runnability(inv, {"max_folds": 2}, reg)
    check("clamp folds 5->2", out.params.get("folds") == 2,
          str(out.params.get("folds")))
    inv2 = MethodInvocationV1(method_id="tabular.gbdt.histgb.v1",
                              params={"folds": 1},
                              validation="single_holdout")
    out2 = clamp_invocation_runnability(inv2, {"max_folds": 2}, reg)
    check("clamp keeps folds within budget",
          out2.params.get("folds") == 1, str(out2.params.get("folds")))
    inv3 = MethodInvocationV1(method_id="text.string_lookup.v1",
                              params={"min_count": 1},
                              validation="single_holdout")
    out3 = clamp_invocation_runnability(inv3, {"max_folds": 2}, reg)
    check("clamp no-op without folds schema",
          out3.params.get("folds") is None, str(out3.params))
    src = _scrub_comments(
        (HERE / "v2_closed_loop.py").read_text(encoding="utf-8"))
    check("closed loop wires runnability clamp",
          "clamp_invocation_runnability" in src
          and "text.string_lookup.v1" in src, "")


def test_deterministic_string_copy():
    tmp = Path(tempfile.mkdtemp(prefix="v256_detcopy_"))
    try:
        _write_text_norm_like(tmp)
        layout = resolve_dataset_layout(str(tmp))
        prof = AnalysisProfile(competition="string-target-stub",
                               task_type="classification",
                               target_column="after",
                               string_target=True,
                               string_source_column="before")
        res = write_deterministic_artifacts(layout, tmp, prof,
                                            metric_name="accuracy")
        check("deterministic: submission written",
              bool(res.get("submission")), str(res))
        check("deterministic: oof written",
              bool(res.get("oof")), str(res))
        with open(str(tmp / "submission.csv"), newline="") as fh:
            sub = list(csv.reader(fh))
        check("deterministic: header id,after",
              bool(sub) and sub[0] == ["id", "after"],
              str(sub[0] if sub else None))
        with open(str(tmp / "test.csv"), newline="") as fh:
            trows = list(csv.DictReader(fh))
        ok_rows = all(
            sub[i + 1][1] == (trows[i].get("before") or "").strip()
            for i in range(min(len(trows), len(sub) - 1)))
        check("deterministic: copy-source per row", ok_rows, "")
        with open(str(tmp / "oof.csv"), newline="") as fh:
            oof = list(csv.reader(fh))
        check("deterministic: oof copy-source",
              bool(oof) and oof[0] == ["true", "pred"]
              and all(len(r) == 2 and r[0] == r[1]
                      for r in oof[1:]),
              "")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_no_competition_hardcoding():
    for path in ("program_compiler.py", "hera/analyzer.py",
                 "v2_closed_loop.py", "capability_registry.py",
                 "pact/deterministic.py", "v2_contracts.py"):
        src = (HERE / path).read_text(encoding="utf-8")
        code = _scrub_comments(src)
        for token in ("new-york-city-taxi-fare-prediction",
                      "tabular-playground-series-dec-2021",
                      "text-normalization-challenge-english-language"):
            check("%s code free of %r" % (path, token),
                  token not in code, "")


def main():
    test_row_cap_template_fix()
    test_row_cap_end_to_end()
    test_string_lookup_declared()
    test_string_lookup_renders()
    test_string_lookup_end_to_end()
    test_analyzer_string_target()
    test_clamp_invocation_runnability()
    test_deterministic_string_copy()
    test_no_competition_hardcoding()
    print("RESULT=%s ok=%d fail=%d" % ("PASS" if FAIL == 0 else "FAIL",
                                       PASS, FAIL))
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())