# -*- coding: utf-8 -*-
"""pact/deterministic.py - host-side deterministic artifact fallback.

When candidate code fails (crash / timeout / missing artifacts), PACT writes
the deterministic majority/mean baseline artifacts itself so the trial still
produces a verifiable, independently recomputable metric instead of burning
the trial budget on nothing. Mirrors implementer._build_baseline_code
semantics but runs in the host process (no subprocess, no container).
"""
import csv
import os
from pathlib import Path
from typing import Optional

from data_layout import DatasetLayout
from v2_contracts import AnalysisProfile


def _read_rows(path: Path) -> list:
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        return list(csv.DictReader(fh))


def _sniff_newline(path) -> str:
    """Match the reference file's line terminator so re-serialized CSV is
    byte-identical to the sample on any OS (Linux LF vs Windows CRLF)."""
    try:
        if path is not None and path.is_file():
            with open(path, "rb") as fh:
                chunk = fh.read(8192)
            if b"\r\n" in chunk:
                return "\r\n"
            if b"\n" in chunk:
                return "\n"
            if b"\r" in chunk:
                return "\r"
    except OSError:
        pass
    return "\r\n"  # csv module default (RFC 4180)

def _is_float(value) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def majority_prediction(values: list, task_type: str = "classification") -> str:
    """Deterministic majority class id / mean (same space as labels)."""
    if not values:
        return ""
    if task_type == "regression":
        numeric = [v for v in values if _is_float(v)]
        if not numeric:
            return "0.0"
        return repr(round(sum(float(v) for v in numeric) / len(numeric), 6))
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=counts.get)


def _class_freqs(values: list) -> dict:
    n = float(len(values) or 1)
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return {k: counts[k] / n for k in sorted(counts)}


# Metrics whose deterministic OOF artifact needs the positive-class
# frequency as pred (binary_logloss recomputes P(y=1) per row).
_BINARY_POS_FREQ_METRICS = frozenset({"binary_logloss"})


def _prob_family(metric_name: Optional[str]) -> bool:
    """Metrics whose OOF v2 contract needs true_<class>/pred_<class> columns."""
    return (metric_name or "") in ("logloss", "weighted_logloss", "kl_div",
                                   "mean_auc_multilabel")


def write_deterministic_artifacts(layout: DatasetLayout, work_dir,
                                  profile: Optional[AnalysisProfile] = None,
                                  metric_name: Optional[str] = None) -> dict:
    """Write submission.csv + oof.csv into work_dir from majority/mean.

    OOF v2 contract: probability-family metrics get one-hot true_<class>
    columns plus class-frequency pred_<class> columns so the trusted
    evaluator can recompute logloss/kl-div/auc-family; binary_logloss gets
    the positive-class frequency as pred; everything else stays true,pred
    with the majority class id / mean.

    Returns {'submission': bool, 'oof': bool, 'pred': str, 'rows': int}.
    """
    work_dir = Path(work_dir)
    train_path = Path(layout.train_path) if layout.train_path else None
    test_path = Path(layout.test_path) if layout.test_path else None
    sample_path = (Path(layout.sample_submission_path)
                   if layout.sample_submission_path else None)

    target = getattr(profile, "target_column", "") or ""
    task_type = getattr(profile, "task_type", "classification") or "classification"

    values = []
    if train_path is not None and train_path.is_file():
        rows = _read_rows(train_path)
        if rows:
            if target and target in rows[0]:
                values = [(r.get(target) or "").strip() for r in rows]
            elif len(rows[0]) > 1:
                values = [list(r.values())[-1].strip() for r in rows]
    pred = majority_prediction(values, task_type)

    submission_written = False
    src_rows = None
    if sample_path is not None and sample_path.is_file():
        src_rows = _read_rows(sample_path)
    elif test_path is not None and test_path.is_file():
        src_rows = _read_rows(test_path)
    lt = _sniff_newline(sample_path if (sample_path is not None and sample_path.is_file()) else test_path)
    if src_rows:
        header = list(src_rows[0].keys())
        id_col = header[0]
        pred_cols = header[1:]
        _tmp_sub = work_dir / "submission.csv.tmp"
        with open(_tmp_sub, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh, lineterminator=lt)
            writer.writerow(header)
            for row in src_rows:
                out = []
                for col in header:
                    if col == id_col:
                        out.append(row.get(col, ""))
                    elif len(pred_cols) > 1:
                        out.append(1.0 if col == str(pred) else 0.0)
                    else:
                        out.append(pred)
                writer.writerow(out)
        # Atomic replace: the candidate container may have left root-owned
        # files that the host cannot open("w"); rename needs only dir write.
        os.replace(_tmp_sub, work_dir / "submission.csv")
        submission_written = True

    oof_written = False
    if values:
        _tmp_oof = work_dir / "oof.csv.tmp"
        with open(_tmp_oof, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh, lineterminator=lt)
            if _prob_family(metric_name):
                classes = sorted(set(values))
                freqs = _class_freqs(values)
                writer.writerow(["true"] + ["true_%s" % k for k in classes]
                                + ["pred_%s" % k for k in classes])
                for v in values:
                    writer.writerow([v]
                                    + ["1" if v == k else "0" for k in classes]
                                    + ["%.6f" % freqs[k] for k in classes])
            elif metric_name in _BINARY_POS_FREQ_METRICS:
                freqs = _class_freqs(values)
                pos = ("1" if "1" in freqs
                       else ("True" if "True" in freqs
                             else max(freqs, key=freqs.get)))
                writer.writerow(["true", "pred"])
                for v in values:
                    writer.writerow([v, "%.6f" % freqs.get(pos, 0.5)])
            else:
                writer.writerow(["true", "pred"])
                for v in values:
                    writer.writerow([v, pred])
        os.replace(_tmp_oof, work_dir / "oof.csv")
        oof_written = True

    return {
        "submission": submission_written,
        "oof": oof_written,
        "pred": pred,
        "rows": len(values),
    }
