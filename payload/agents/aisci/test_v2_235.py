# -*- coding: utf-8 -*-
"""v2.3.5 offline tests: MLE-Bench full-layout generalization.

Covers the four blind spots found by the 82-competition static scan:
  1) train_labels.csv train tables (histopathologic/rsna-miccai/seti);
  2) .tsv train/test tables (movie-review) incl. sampleSubmission.csv;
  3) image competitions with NO public test.csv (private answers.csv +
     sanitize) and NO train labels CSV (class-dir / flat-prefix synthesis,
     plant-seedlings / dogs-vs-cats);
  4) mixed-data tables (stray prose Name column) staying tabular, while
     multi-label text tasks (all labels in the sample file) stay text.

Run: python test_v2_235.py   (from the aisci payload dir)
"""
import os, shutil, sys, tempfile
from pathlib import Path

from data_layout import (resolve_dataset_layout, sanitize_test_csv,
                         synthesize_train_labels)
from hera.analyzer import Analyzer

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("[OK] " + name)
    else:
        print("[FAIL] " + name + (" | " + str(detail)[:300] if detail else ""))
        FAILURES.append(name)


def _mkimg(d, name, ext=".jpg"):
    (Path(d) / (name + ext)).write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 60)


def _profile(data, prompt):
    return Analyzer(str(data), task_prompt=prompt).profile("v235_stub")


def test_train_labels_layout():
    """histopathologic shape: train_labels.csv + train/*.tif + private/
    answers.csv (no public test.csv)."""
    import numpy as np, pandas as pd
    tmp = Path(tempfile.mkdtemp(prefix="v235_tl_"))
    try:
        pub = tmp / "prepared" / "public"
        priv = tmp / "prepared" / "private"
        pub.mkdir(parents=True)
        priv.mkdir(parents=True)
        (pub / "train").mkdir()
        (pub / "test").mkdir()
        for i in range(60):
            _mkimg(pub / "train", "id%d" % i, ".tif")
        for i in range(3):
            _mkimg(pub / "test", "id%d" % (60 + i), ".tif")
        pd.DataFrame({"id": ["id%d" % i for i in range(60)],
                      "label": [0, 1] * 30}).to_csv(pub / "train_labels.csv",
                                                    index=False)
        pd.DataFrame({"id": ["id%d" % i for i in range(60, 63)],
                      "label": 0}).to_csv(priv / "answers.csv", index=False)
        pd.DataFrame({"id": ["id%d" % i for i in range(60, 63)],
                      "label": 0}).to_csv(pub / "sample_submission.csv",
                                          index=False)
        san = sanitize_test_csv(tmp)
        check("train_labels: sanitize wrote public test",
              bool(san.get("written")), str(san))
        layout = resolve_dataset_layout(tmp)
        check("train_labels: layout resolves",
              layout.train_path.name == "train_labels.csv",
              str(layout.train_path))
        check("train_labels: test label-free",
              layout.test_has_labels is False and
              layout.test_path.name == "test.csv", str(layout.test_path))
        prof = _profile(tmp, "cancer detection binary classification")
        check("train_labels: image modality", prof.modality == "image",
              prof.modality)
        check("train_labels: target label",
              prof.target_column == "label", prof.target_column)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_tsv_layout():
    """movie-review shape: train.tsv/test.tsv + sampleSubmission.csv."""
    import numpy as np, pandas as pd
    tmp = Path(tempfile.mkdtemp(prefix="v235_tsv_"))
    try:
        pub = tmp / "prepared" / "public"
        priv = tmp / "prepared" / "private"
        pub.mkdir(parents=True)
        priv.mkdir(parents=True)
        pd.DataFrame({"PhraseId": range(20), "SentenceId": range(20),
                      "Phrase": ["great film"] * 20,
                      "Sentiment": [0] * 20}).to_csv(pub / "train.tsv",
                                                     sep="\t", index=False)
        pd.DataFrame({"PhraseId": range(20, 25), "SentenceId": range(20, 25),
                      "Phrase": ["bad film"] * 5}).to_csv(pub / "test.tsv",
                                                          sep="\t",
                                                          index=False)
        pd.DataFrame({"PhraseId": range(20, 25), "Sentiment": 2}).to_csv(
            pub / "sampleSubmission.csv", index=False)
        layout = resolve_dataset_layout(tmp)
        check("tsv: train resolves", layout.train_path.suffix == ".tsv",
              str(layout.train_path))
        check("tsv: test resolves", layout.test_path.suffix == ".tsv",
              str(layout.test_path))
        check("tsv: sample resolves",
              layout.sample_submission_path is not None and
              layout.sample_submission_path.name == "sampleSubmission.csv",
              str(layout.sample_submission_path))
        prof = _profile(tmp, "sentiment classification")
        check("tsv: text modality", prof.modality == "text", prof.modality)
        check("tsv: target Sentiment",
              prof.target_column == "Sentiment", prof.target_column)
        check("tsv: Phrase is text col",
              "Phrase" in prof.text_columns, str(prof.text_columns))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_mixed_stays_tabular():
    """spaceship/titanic + adversarial 5-word Name: stays tabular."""
    import numpy as np, pandas as pd
    tmp = Path(tempfile.mkdtemp(prefix="v235_mix_"))
    try:
        pub = tmp / "prepared" / "public"
        priv = tmp / "prepared" / "private"
        pub.mkdir(parents=True)
        priv.mkdir(parents=True)
        pd.DataFrame({"id": range(30),
                      "Name": ["John Smith from Boston, MA"] * 30,
                      "a": np.random.randn(30), "b": np.random.randn(30),
                      "target": [0, 1] * 15}).to_csv(pub / "train.csv",
                                                     index=False)
        pd.DataFrame({"id": range(30, 40),
                      "Name": ["Jane Doe from NYC, NY"] * 10,
                      "a": np.random.randn(10),
                      "b": np.random.randn(10)}).to_csv(pub / "test.csv",
                                                        index=False)
        pd.DataFrame({"id": range(30, 40), "target": 0}).to_csv(
            pub / "sample_submission.csv", index=False)
        prof = _profile(tmp, "binary classification")
        check("mixed: tabular modality", prof.modality == "tabular",
              prof.modality)
        check("mixed: Name recorded as text",
              "Name" in prof.text_columns, str(prof.text_columns))
        check("mixed: target correct",
              prof.target_column == "target", prof.target_column)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_multilabel_text_stays_text():
    """jigsaw-toxic shape: all labels in the sample file, prose column."""
    import numpy as np, pandas as pd
    tmp = Path(tempfile.mkdtemp(prefix="v235_mlt_"))
    try:
        pub = tmp / "prepared" / "public"
        priv = tmp / "prepared" / "private"
        pub.mkdir(parents=True)
        priv.mkdir(parents=True)
        pd.DataFrame({"id": range(20),
                      "comment_text": ["this is a long abusive sentence " * 4] * 20,
                      "toxic": [0, 1] * 10, "severe_toxic": [0] * 20,
                      "obscene": [0] * 20}).to_csv(pub / "train.csv",
                                                   index=False)
        pd.DataFrame({"id": range(20, 25),
                      "comment_text": ["totally fine normal sentence"] * 5}
                     ).to_csv(pub / "test.csv", index=False)
        pd.DataFrame({"id": range(20, 25), "toxic": .5, "severe_toxic": .1,
                      "obscene": .2}).to_csv(pub / "sample_submission.csv",
                                             index=False)
        prof = _profile(tmp, "toxic comment multi-label classification")
        check("multilabel: text modality", prof.modality == "text",
              prof.modality)
        check("multilabel: comment_text detected",
              "comment_text" in prof.text_columns, str(prof.text_columns))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_classdir_synthesis():
    """plant-seedlings shape: no train CSV, train/<species>/*.png."""
    import numpy as np, pandas as pd
    tmp = Path(tempfile.mkdtemp(prefix="v235_synth_"))
    try:
        pub = tmp / "prepared" / "public"
        priv = tmp / "prepared" / "private"
        pub.mkdir(parents=True)
        priv.mkdir(parents=True)
        for sp in ("Black-grass", "Sugar beet"):
            (pub / "train" / sp).mkdir(parents=True)
            for i in range(30):
                _mkimg(pub / "train" / sp, "%s_%d" % (sp.replace(" ", "_"), i),
                       ".png")
        (pub / "test").mkdir()
        for i in range(3):
            _mkimg(pub / "test", "t%d" % i, ".png")
        pd.DataFrame({"file": ["t0", "t1", "t2"],
                      "species": "Sugar beet"}).to_csv(
            pub / "sample_submission.csv", index=False)
        pd.DataFrame({"file": ["t0", "t1", "t2"],
                      "species": ["Black-grass", "Sugar beet", "Black-grass"]}
                     ).to_csv(priv / "answers.csv", index=False)
        rep = synthesize_train_labels(tmp)
        check("synth: wrote train.csv",
              rep.get("written") and rep.get("mode") == "class-dirs", str(rep))
        train_csv = pub / "train.csv"
        check("synth: columns from sample",
              train_csv.is_file() and
              train_csv.read_text(encoding="utf-8").splitlines()[0] ==
              "file,species",
              train_csv.read_text(encoding="utf-8").splitlines()[:1])
        prof = _profile(tmp, "plant seedling species classification")
        check("synth: image modality", prof.modality == "image", prof.modality)
        check("synth: target species",
              prof.target_column == "species", prof.target_column)
        check("synth: target stats measured", prof.train_rows == 60,
              prof.train_rows)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_flat_prefix_synthesis():
    """dogs-vs-cats shape: flat train dir with cat.0.jpg / dog.1.jpg."""
    import numpy as np, pandas as pd
    tmp = Path(tempfile.mkdtemp(prefix="v235_flat_"))
    try:
        pub = tmp / "prepared" / "public"
        priv = tmp / "prepared" / "private"
        pub.mkdir(parents=True)
        priv.mkdir(parents=True)
        (pub / "train").mkdir()
        (pub / "test").mkdir()
        for i in range(30):
            _mkimg(pub / "train", "cat.%d.jpg" % i)
            _mkimg(pub / "train", "dog.%d.jpg" % i)
        for i in range(3):
            _mkimg(pub / "test", "%d.jpg" % (i + 1))
        pd.DataFrame({"id": [1, 2, 3], "label": 0.5}).to_csv(
            pub / "sample_submission.csv", index=False)
        pd.DataFrame({"id": [1, 2, 3], "label": [0, 1, 0]}).to_csv(
            priv / "answers.csv", index=False)
        rep = synthesize_train_labels(tmp)
        check("flat: synth mode",
              rep.get("mode") == "flat-prefix", str(rep))
        prof = _profile(tmp, "cats vs dogs binary classification")
        check("flat: image modality", prof.modality == "image", prof.modality)
        check("flat: target label",
              prof.target_column == "label", prof.target_column)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_sanitize_sample_driven_target():
    """taxi shape: labels.csv ends with passenger_count, target is the
    mid-table fare_amount; sanitize must drop ONLY fare_amount."""
    import numpy as np, pandas as pd
    tmp = Path(tempfile.mkdtemp(prefix="v235_taxi_"))
    try:
        pub = tmp / "prepared" / "public"
        priv = tmp / "prepared" / "private"
        pub.mkdir(parents=True)
        priv.mkdir(parents=True)
        pd.DataFrame({"key": range(30),
                      "pickup_datetime": ["2020-01-01 00:00:00 UTC"] * 30,
                      "pickup_longitude": np.random.randn(30),
                      "dropoff_longitude": np.random.randn(30),
                      "passenger_count": np.random.randint(1, 5, 30),
                      "fare_amount": np.random.randn(30) * 10}).to_csv(
            pub / "labels.csv", index=False)
        pd.DataFrame({"key": range(30, 40),
                      "pickup_datetime": ["2020-01-02 00:00:00 UTC"] * 10,
                      "pickup_longitude": np.random.randn(10),
                      "dropoff_longitude": np.random.randn(10),
                      "passenger_count": np.random.randint(1, 5, 10)}).to_csv(
            priv / "test.csv", index=False)
        pd.DataFrame({"key": range(30, 40), "fare_amount": 0.0}).to_csv(
            pub / "sample_submission.csv", index=False)
        rep = sanitize_test_csv(tmp)
        check("taxi: sanitize wrote", bool(rep.get("written")), str(rep))
        out = (pub / "test.csv").read_text(encoding="utf-8").splitlines()
        check("taxi: fare_amount dropped",
              "fare_amount" not in out[0] and "key" in out[0], out[0])
        check("taxi: passenger_count kept",
              "passenger_count" in out[0], out[0])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    print("=== V2.3.5 layout generalization tests ===")
    test_train_labels_layout()
    test_tsv_layout()
    test_mixed_stays_tabular()
    test_multilabel_text_stays_text()
    test_classdir_synthesis()
    test_flat_prefix_synthesis()
    test_sanitize_sample_driven_target()
    if FAILURES:
        print("FAILURES=%d: %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("ALL_V235_TESTS=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
