# -*- coding: utf-8 -*-
"""v2.5.5 offline tests: generic MLE-Bench data-layout robustness.

1) localized-prefix zip tables (text-normalization en_/ru_ style) resolve
   via prefix-agnostic materialization - no competition names involved;
2) unzipped localized tables resolve too (copy path);
3) materialization is idempotent;
4) broken layouts still raise DatasetLayoutError (launch preflight fails
   loudly instead of a silent closed-loop crash);
5) data_layout.py contains no competition-name hardcoding.
"""
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data_layout import DatasetLayoutError, resolve_dataset_layout  # noqa: E402

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


def test_no_competition_hardcoding():
    # Scan CODE only (comments/documentation may name the quirk; logic may
    # not). Strips full-line comments - the routing logic must stay generic.
    src = (HERE / "data_layout.py").read_text(encoding="utf-8")
    code = "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith("#"))
    for token in ("text-normalization", "en_train", "ru_train",
                  "en_test_2", "ru_test_2", "en_sample", "ru_sample"):
        check("data_layout code free of %r" % token, token not in code, "")
    check("data_layout has generic materializer",
          "def _materialize_localized_tables" in code, "")


def main():
    test_localized_zip_tables_resolve()
    test_localized_plain_tables_resolve()
    test_broken_layout_raises()
    test_no_competition_hardcoding()
    print("RESULT=%s ok=%d fail=%d" % ("PASS" if FAIL == 0 else "FAIL",
                                       PASS, FAIL))
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())