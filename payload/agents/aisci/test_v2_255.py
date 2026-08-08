# -*- coding: utf-8 -*-
"""v2.5.5 offline tests: generic MLE-Bench data-layout + sample-column
robustness.

1) localized-prefix zip tables (en_/ru_ style) resolve via prefix-agnostic
   materialization - no competition names involved;
2) unzipped localized tables resolve too (copy path);
3) materialization is idempotent;
4) broken layouts still raise DatasetLayoutError (launch preflight fails
   loudly instead of a silent closed-loop crash);
5) filename-prefix image labels (cat.0.jpg / dog.1.jpg style) synthesize
   a REAL train.csv through the resolver - never the all-zero sample copy;
6) no-id sample submissions (target FIRST, e.g. Insult,Date,Comment):
   analyzer picks the target from the sample columns absent in test.csv,
   the join id is the first sample column present in test.csv, and the
   compiled harness writes passthrough columns verbatim in sample order;
7) data_layout.py / program_compiler.py / hera/analyzer.py contain no
   competition-name hardcoding in code (full-line comments are doc).
"""
import base64
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from capability_registry import CapabilityRegistry  # noqa: E402
from data_layout import DatasetLayoutError, resolve_dataset_layout  # noqa: E402
from hera.analyzer import Analyzer  # noqa: E402
from program_compiler import ProgramCompiler  # noqa: E402
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


def _zip_csv(src: Path, dst: Path):
    with zipfile.ZipFile(str(dst), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(str(src), arcname=src.name)


def test_localized_zip_tables_resolve():
    tmp = Path(tempfile.mkdtemp(prefix="v255_localized_zip_"))
    try:
        pub = tmp / "prepared" / "public"
        priv = tmp / "prepared" / "private"
        pub.mkdir(parents=True)
        priv.mkdir(parents=True)
        train = tmp / "zz_lang_train.csv"
        test = tmp / "zz_lang_test_2.csv"
        sample = tmp / "zz_lang_sample_submission_2.csv"
        _write(train, ["sentence_id,token_id,class,before,after",
                       "0,0,PLAIN,hello,hello"])
        _write(test, ["sentence_id,token_id,before", "0,0,world"])
        _write(sample, ["id,after", "0_0,world"])
        _write(priv / "answers.csv", ["id,after", "0_0,world"])
        _zip_csv(train, pub / "zz_lang_train.csv.zip")
        _zip_csv(test, pub / "zz_lang_test_2.csv.zip")
        _zip_csv(sample, pub / "zz_lang_sample_submission_2.csv.zip")
        train.unlink()
        test.unlink()
        sample.unlink()
        d = resolve_dataset_layout(str(tmp))
        check("localized zip -> mlebench_prepared layout",
              d.layout_name == "mlebench_prepared", d.layout_name)
        check("localized zip -> canonical train.csv",
              d.train_path.name == "train.csv" and d.train_path.is_file(),
              str(d.train_path))
        check("localized zip -> canonical test.csv",
              d.test_path.name == "test.csv" and d.test_path.is_file(),
              str(d.test_path))
        check("localized zip -> canonical sample_submission.csv",
              d.sample_submission_path is not None
              and d.sample_submission_path.name == "sample_submission.csv",
              str(d.sample_submission_path))
        check("localized zip -> label-free public test",
              d.test_has_labels is False, str(d.test_has_labels))
        d2 = resolve_dataset_layout(str(tmp))
        check("materialization idempotent",
              d2.train_path == d.train_path and d2.layout_name == d.layout_name,
              "")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_localized_plain_tables_resolve():
    tmp = Path(tempfile.mkdtemp(prefix="v255_localized_plain_"))
    try:
        pub = tmp / "prepared" / "public"
        priv = tmp / "prepared" / "private"
        pub.mkdir(parents=True)
        priv.mkdir(parents=True)
        _write(pub / "zz_lang_train.csv", ["sentence_id,token_id,class,before,after",
                                           "0,0,PLAIN,hello,hello"])
        _write(pub / "zz_lang_test_1.csv", ["sentence_id,token_id,before",
                                            "0,0,world"])
        _write(pub / "zz_lang_sample_submission_1.csv", ["id,after", "0_0,world"])
        _write(priv / "answers.csv", ["id,after", "0_0,world"])
        d = resolve_dataset_layout(str(tmp))
        check("localized plain -> resolves", d.layout_name == "mlebench_prepared",
              d.layout_name)
        check("localized plain -> canonical train.csv",
              d.train_path.name == "train.csv" and (pub / "train.csv").is_file(),
              str(d.train_path))
        check("localized plain -> canonical test.csv",
              d.test_path.name == "test.csv" and (pub / "test.csv").is_file(),
              str(d.test_path))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_broken_layout_raises():
    tmp = Path(tempfile.mkdtemp(prefix="v255_broken_"))
    try:
        pub = tmp / "prepared" / "public"
        priv = tmp / "prepared" / "private"
        pub.mkdir(parents=True)
        priv.mkdir(parents=True)
        _write(pub / "notes.txt", ["hello"])
        _write(priv / "answers.csv", ["id,after", "0_0,world"])
        try:
            resolve_dataset_layout(str(tmp))
            check("broken layout raises DatasetLayoutError", False,
                  "resolve unexpectedly succeeded")
        except DatasetLayoutError:
            check("broken layout raises DatasetLayoutError", True, "")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _scrub_comments(src):
    return "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith("#"))


def test_no_competition_hardcoding():
    # Scan CODE only (comments/documentation may name the quirk; logic may
    # not). Strips full-line comments - the routing logic must stay generic.
    for path in ("data_layout.py", "program_compiler.py",
                 "hera/analyzer.py"):
        src = (HERE / path).read_text(encoding="utf-8")
        code = _scrub_comments(src)
        for token in ("text-normalization", "en_train", "ru_train",
                      "en_test_2", "ru_test_2", "en_sample", "ru_sample",
                      "dogs-vs-cats", "detecting-insults",
                      "random-acts-of-pizza", "mlsp-2013",
                      "right-whale-redux"):
            check("%s code free of %r" % (path, token), token not in code, "")
    dl = _scrub_comments((HERE / "data_layout.py").read_text(encoding="utf-8"))
    check("data_layout has generic materializer",
          "def _materialize_localized_tables" in dl, "")
    check("data_layout has prefix-label synthesis",
          "def _synthesize_prefix_label_table" in dl, "")
    pc = _scrub_comments((HERE / "program_compiler.py").read_text(encoding="utf-8"))
    check("compiler templates use passthrough rule",
          "SAMPLE_PASSTHROUGH = [str(c) for c in sample_header"
          " if str(c) in df_test.columns]" in pc, "")


# ------------------------------------------------- prefix-image label synthesis
_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


def _write_png(path: Path):
    path.write_bytes(_PNG_1X1)


def test_prefix_image_labels_resolve():
    tmp = Path(tempfile.mkdtemp(prefix="v255_prefix_img_"))
    try:
        pub = tmp / "prepared" / "public"
        priv = tmp / "prepared" / "private"
        (pub / "train").mkdir(parents=True)
        (pub / "test").mkdir()
        priv.mkdir()
        for i in range(60):
            _write_png(pub / "train" / ("cat.%d.jpg" % i))
            _write_png(pub / "train" / ("dog.%d.jpg" % i))
        for i in range(5):
            _write_png(pub / "test" / ("%d.jpg" % i))
        _write(pub / "sample_submission.csv",
               ["id,label", "0,cat", "1,dog", "2,cat", "3,dog", "4,cat"])
        _write(pub / "test.csv", ["id", "0", "1", "2", "3", "4"])
        _write(priv / "answers.csv",
               ["id,label", "0,cat", "1,dog", "2,cat", "3,dog", "4,cat"])
        d = resolve_dataset_layout(str(tmp))
        check("prefix images -> mlebench_prepared layout",
              d.layout_name == "mlebench_prepared", d.layout_name)
        check("prefix images -> synthesized train.csv",
              d.train_path.name == "train.csv" and d.train_path.is_file(),
              str(d.train_path))
        with open(str(d.train_path), newline="") as fh:
            rows = list(csv.reader(fh))
        check("prefix images -> header id,label",
              bool(rows) and rows[0] == ["id", "label"],
              str(rows[0] if rows else None))
        labels = sorted({r[1] for r in rows[1:] if len(r) > 1})
        check("prefix images -> real cat/dog labels",
              labels == ["cat", "dog"], str(labels))
        check("prefix images -> every image row present",
              len(rows) - 1 == 120, str(len(rows) - 1))
        check("prefix images -> test label-free",
              d.test_has_labels is False, str(d.test_has_labels))
        check("prefix images -> image dirs detected",
              d.train_image_dir is not None and d.test_image_dir is not None,
              "")
        d2 = resolve_dataset_layout(str(tmp))
        check("prefix images -> idempotent",
              d2.train_path == d.train_path, "")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_plain_numbered_images_do_not_synthesize():
    # A flat image dir with plain numeric names must NOT synthesize a
    # bogus label table (all-numeric prefixes are rejected).
    tmp = Path(tempfile.mkdtemp(prefix="v255_num_img_"))
    try:
        pub = tmp / "prepared" / "public"
        priv = tmp / "prepared" / "private"
        (pub / "train").mkdir(parents=True)
        (pub / "test").mkdir()
        priv.mkdir()
        for i in range(60):
            _write_png(pub / "train" / ("%d.jpg" % i))
        for i in range(5):
            _write_png(pub / "test" / ("%d.jpg" % i))
        _write(pub / "sample_submission.csv",
               ["id,label", "0,cat", "1,dog", "2,cat", "3,dog", "4,cat"])
        _write(pub / "test.csv", ["id", "0", "1", "2", "3", "4"])
        _write(priv / "answers.csv",
               ["id,label", "0,cat", "1,dog", "2,cat", "3,dog", "4,cat"])
        d = resolve_dataset_layout(str(tmp))
        check("numeric images -> resolves",
              d.layout_name == "mlebench_prepared", d.layout_name)
        with open(str(d.train_path), newline="") as fh:
            rows = list(csv.reader(fh))
        vals = sorted({r[1] for r in rows[1:] if len(r) > 1})
        check("numeric images -> no prefix tokens",
              not vals or vals == ["cat", "dog"], str(vals))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------- no-id sample (target-first) rules
def _write_insults_like(tmp: Path, n_train=60, n_test=20):
    dates = ["2026-01-%02d" % (i % 28 + 1) for i in range(n_train + n_test)]
    comments = [
        "this comment is completely fine and harmless",
        "you are a terrible person and everyone hates you",
        "great point thank you for sharing this insight",
        "go away nobody wants to read your nonsense here",
        "i completely agree with everything you just said",
        "your opinion is stupid and your mother is ugly",
    ]
    rows = []
    for i in range(n_train):
        insult = 1 if i % 2 == 0 else 0
        rows.append("%d,%s,%s" % (insult, dates[i], comments[i % len(comments)]))
    _write(tmp / "train.csv", ["Insult,Date,Comment"] + rows)
    test_rows = ["%s,%s" % (dates[i], comments[i % len(comments)])
                 for i in range(n_train, n_train + n_test)]
    _write(tmp / "test.csv", ["Date,Comment"] + test_rows)
    sample_rows = ["0,%s,%s" % (dates[i], comments[i % len(comments)])
                   for i in range(n_train, n_train + n_test)]
    _write(tmp / "sample_submission.csv",
           ["Insult,Date,Comment"] + sample_rows)


def test_no_id_sample_analyzer():
    tmp = Path(tempfile.mkdtemp(prefix="v255_insults_ana_"))
    try:
        _write_insults_like(tmp)
        an = Analyzer(str(tmp),
                      task_prompt="Insults: binary text classification (insult 0/1)")
        prof = an.profile("no-id-sample-stub")
        check("no-id sample: target is first column",
              prof.target_column == "Insult", prof.target_column)
        check("no-id sample: task classification",
              prof.task_type == "classification", prof.task_type)
        check("no-id sample: modality text",
              prof.modality == "text", prof.modality)
        check("no-id sample: join id is test column",
              an._id_column(prof.feature_columns) == "Date",
              an._id_column(prof.feature_columns))
        check("no-id sample: sample targets exclude passthrough",
              an._sample_target_columns(prof.feature_columns) == ["Insult"],
              str(an._sample_target_columns(prof.feature_columns)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# -------------------------------------------------- harness end-to-end (no-id)
def _run_harness(code, tmp, target_column, task_type, timeout=None):
    if timeout is None:
        timeout = int(os.environ.get("V2_TEST_HARNESS_TIMEOUT", "600"))
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


def test_no_id_sample_harness_end_to_end():
    # Heavy subprocess gate mirrors test_v2_251: skipped under
    # V2_TEST_HARNESS_SKIP=1 (ops default under fleet load); validated in
    # full installs without the flag.
    if os.environ.get("V2_TEST_HARNESS_SKIP", "0") == "1":
        print("[SKIP] end-to-end harness subprocess (V2_TEST_HARNESS_SKIP=1)")
        return
    tmp = Path(tempfile.mkdtemp(prefix="v255_insults_run_"))
    try:
        _write_insults_like(tmp)
        reg = CapabilityRegistry()
        pc = ProgramCompiler(reg)
        prof = AnalysisProfile(
            competition="no-id-sample-stub",
            modality="text",
            task_type="classification",
            metric_name="logloss",
            metric_direction="min",
            train_rows=60,
            test_rows=20,
            feature_columns=["Insult", "Date", "Comment"],
            target_column="Insult",
            text_columns=["Comment"],
        )
        manifest = {"metric_name": "logloss", "metric_direction": "min"}
        inv = MethodInvocationV1(
            method_id="text.embedding.tfidf.v1",
            hypothesis="no-id-sample",
            params={"max_iter": 100, "C": 1.0},
            preprocessing=["tfidf_vectorization"],
            validation="single_holdout")
        code, th = pc.render(inv, profile=prof, manifest=manifest)
        check("no-id sample: render succeeds",
              code.startswith("# -*- coding"), th[:24])
        _run_harness(code, tmp, target_column="Insult", task_type="classification")
        with open(str(tmp / "submission.csv"), newline="") as fh:
            sub = list(csv.reader(fh))
        check("no-id sample: submission header",
              bool(sub) and sub[0] == ["Insult", "Date", "Comment"],
              str(sub[0] if sub else None))
        check("no-id sample: submission rows",
              len(sub) - 1 == 20, str(len(sub) - 1))
        with open(str(tmp / "sample_submission.csv"), newline="") as fh:
            sample_rows = list(csv.reader(fh))[1:]
        passthrough_ok = all(sub[i + 1][1:] == sample_rows[i][1:]
                             for i in range(20))
        check("no-id sample: passthrough verbatim", passthrough_ok, "")
        bad = 0
        vals = []
        for r in sub[1:]:
            try:
                vals.append(float(r[0]))
            except (ValueError, IndexError):
                bad += 1
        check("no-id sample: insult preds numeric",
              bad == 0 and len(vals) == 20, "bad=%d" % bad)
        check("no-id sample: not placeholder zeros",
              not (len(set(vals)) == 1 and set(vals) == {0.0}),
              str(sorted(set(vals))[:5]))
        with open(str(tmp / "oof.csv"), newline="") as fh:
            oof = list(csv.reader(fh))
        truths = {r[0] for r in oof[1:] if r}
        check("no-id sample: oof targets are insult labels",
              truths <= {"0", "1"}, str(truths))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    test_localized_zip_tables_resolve()
    test_localized_plain_tables_resolve()
    test_broken_layout_raises()
    test_prefix_image_labels_resolve()
    test_plain_numbered_images_do_not_synthesize()
    test_no_id_sample_analyzer()
    test_no_id_sample_harness_end_to_end()
    test_no_competition_hardcoding()
    print("RESULT=%s ok=%d fail=%d" % ("PASS" if FAIL == 0 else "FAIL",
                                       PASS, FAIL))
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
