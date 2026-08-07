# -*- coding: utf-8 -*-
"""pact/verifier.py - Deterministic verification of an executed trial.

No LLM. Checks the submission artifact, parses the metric from the execution
log, and produces a fail-closed verdict against the best verified metric.
"""
import hashlib
import re
from pathlib import Path
from typing import List, Optional

from v2_contracts import TrialReceipt, now_iso, new_id
from pact.executor import ExecOutcome

_METRIC_PATTERNS = [
    r"{metric}[\s:]*([0-9]*\.?[0-9]+)",
    r"val_{metric}[\s:]*([0-9]*\.?[0-9]+)",
    r"test_{metric}[\s:]*([0-9]*\.?[0-9]+)",
    r"best_{metric}[\s:]*([0-9]*\.?[0-9]+)",
    r"([0-9]*\.?[0-9]+)\s*(?:accuracy|auc|f1|logloss|rmse)",
]


class Verifier:
    """Verifies submissions, parses metrics and computes verdicts."""

    def __init__(self, submission_dir, work_dir=None,
                 sample_path: Optional[str] = None, min_delta: float = 0.01):
        self.submission_dir = Path(submission_dir)
        self.work_dir = Path(work_dir) if work_dir else None
        self.sample_path = Path(sample_path) if sample_path else None
        # v2.3.6: per-metric improvement threshold (default keeps legacy
        # behavior for tests/legacy callers).
        self.min_delta = float(min_delta)
        self.submission_dir.mkdir(parents=True, exist_ok=True)

    def parse_metric(self, stdout: str, stderr: str, returncode: int,
                     metric_name: str = "accuracy") -> Optional[float]:
        if returncode != 0:
            return None
        token = re.escape(metric_name)
        for text in (stdout or "", stderr or ""):
            for pat in _METRIC_PATTERNS:
                match = re.search(pat.format(metric=token), text, re.IGNORECASE)
                if match:
                    try:
                        return float(match.group(1))
                    except ValueError:
                        continue
        return None

    def _submission_rows(self, path: Path) -> int:
        try:
            with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
                return sum(1 for _ in fh)
        except OSError:
            return 0

    def _locate_submission(self) -> Optional[Path]:
        primary = self.submission_dir / "submission.csv"
        if primary.is_file() and primary.stat().st_size > 0:
            return primary
        if self.work_dir is not None:
            candidate = self.work_dir / "submission.csv"
            if candidate.is_file() and candidate.stat().st_size > 0:
                import shutil
                shutil.copy2(candidate, primary)
                return primary
        return None

    def verify(self, spec, outcome: ExecOutcome, best_metric: Optional[float],
               metric_name: str = "accuracy") -> TrialReceipt:
        submission_path = self._locate_submission()
        submission_exists = False
        submission_hash = ""
        if submission_path is not None:
            submission_exists = True
            submission_hash = "sha256:" + hashlib.sha256(submission_path.read_bytes()).hexdigest()
            if self.sample_path is not None:
                expected = max(1, self._submission_rows(self.sample_path) - 1)
                actual = max(0, self._submission_rows(submission_path) - 1)
                if expected and abs(actual - expected) > max(2, expected * 0.01):
                    submission_exists = False  # row-count mismatch -> not verified
                    submission_hash = ""

        metric = self.parse_metric(outcome.stdout, outcome.stderr,
                                   outcome.returncode, metric_name)
        if outcome.returncode != 0:
            verdict, evidence = "failure", (
                "Execution failed with returncode=%s" % outcome.returncode)
        elif metric is None:
            verdict, evidence = "failure", (
                "No metric found in execution log and no verified submission")
        elif best_metric is None:
            verdict, evidence = "success", "First verified trial: metric=%.4f" % metric
        elif metric > best_metric + self.min_delta:
            verdict, evidence = "success", (
                "Improved from %.4f to %.4f (+%.4f)" % (best_metric, metric, metric - best_metric))
        elif metric >= best_metric - self.min_delta:
            verdict, evidence = "stagnant", (
                "Stagnant at %.4f (delta=%.4f)" % (metric, metric - best_metric))
        else:
            verdict, evidence = "regression", (
                "Regressed from %.4f to %.4f (%.4f)" % (best_metric, metric, metric - best_metric))
        failure_reason = ""
        if verdict == "failure":
            if outcome.timed_out:
                failure_reason = "timeout after %ss" % outcome.wall_clock_seconds
            elif outcome.returncode != 0:
                tail = [line.strip() for line in
                        ((outcome.stderr or "") + "\n" + (outcome.stdout or ""))
                        .strip().splitlines() if line.strip()][-3:]
                failure_reason = " ".join(tail) or "exit code %s" % outcome.returncode
                failure_reason = failure_reason[:300]
            else:
                failure_reason = "no metric or submission artifact found"

        return TrialReceipt(
            receipt_id=new_id("receipt"),
            spec_id=spec.spec_id,
            competition=spec.competition,
            round_num=spec.round_num,
            returncode=outcome.returncode,
            stdout=(outcome.stdout or "")[-4000:],
            stderr=(outcome.stderr or "")[-2000:],
            metric=metric,
            metric_name=metric_name,
            verdict=verdict,
            evidence=evidence,
            submission_exists=submission_exists,
            submission_path=str(submission_path) if submission_path else "",
            submission_hash=submission_hash,
            wall_clock_seconds=outcome.wall_clock_seconds,
            code_hash=spec.code_hash,
            verified=True,
            failure_reason=failure_reason,
        )
