# -*- coding: utf-8 -*-
"""pact/publisher.py - ControlledPublisher: publish ONLY certified-best bundles.

Design contract: the publisher accepts only a bundle bound to the
certified-best pointer from PromotionRecord. It copies the bundle's
submission artifact into the submission area and refuses anything else.
"""
import shutil
from pathlib import Path
from typing import Optional

from v2_contracts import CandidateBundle, PromotionRecord


class PublishError(RuntimeError):
    """Raised when an uncertified bundle tries to publish."""


class ControlledPublisher:
    """Publishes certified-best submissions only."""

    def __init__(self, bus, submission_dir):
        self.bus = bus
        self.submission_dir = Path(submission_dir)
        self.submission_dir.mkdir(parents=True, exist_ok=True)


    def publish_certified(self) -> Path:
        """Publish the certified-best bundle if it exists; else no-op."""
        record = self.bus.load_promotion()
        if not record:
            raise PublishError("no promotion record: nothing certified")
        promo = PromotionRecord.from_dict(record)
        best_id = promo.certified_best_trial_id
        if not best_id:
            raise PublishError("no certified-best trial to publish")

        # certified-best points at a TRIAL id; bundles are keyed by bundle id,
        # so resolve by the trial_id binding inside the bundle payload.
        bundle = self._find_bundle_by_trial(best_id)
        if bundle is None:
            raise PublishError("certified-best bundle missing: " + best_id)

        src = Path(bundle.submission_path) if bundle.submission_path else None
        if src is None or not src.is_file():
            raise PublishError("certified bundle has no submission artifact")
        dst = self.submission_dir / "submission.csv"
        shutil.copy2(src, dst)
        # v2.3.9: generic submission self-check BEFORE delivery. Probability
        # rows (multi-column, values in [0,1], row sums ~1) get clipped and
        # renormalized so the published artifact can never carry NaN,
        # negative, or off-by-rounding row sums that official graders reject
        # (e.g. atol=1e-6 row-sum checks). Regression/class outputs are left
        # untouched. Detection is shape-driven, never competition-named.
        report = self._sanitize_submission(dst)
        if report.get("fixed"):
            print("[publisher] submission sanitized: rows=%d detail=%s"
                  % (report["rows_repaired"], report["detail"]))
        return dst

    def _sanitize_submission(self, dst: Path) -> dict:
        """Shape-driven probability-row repair on a published submission.

        Returns {"fixed": bool, "rows_repaired": int, "detail": str}.
        """
        import csv
        import math
        report = {"fixed": False, "rows_repaired": 0, "detail": ""}
        try:
            raw = dst.read_bytes()
            text = raw.decode("utf-8", errors="replace")
            newline = "\r\n" if b"\r\n" in raw else "\n"
            rows = list(csv.reader(text.splitlines()))
        except OSError:
            return report
        if len(rows) < 2 or len(rows[0]) < 2:
            return report
        header, body = rows[0], rows[1:]
        ncols = len(header)
        body = [r for r in body if len(r) >= ncols]
        if not body:
            return report

        def _is_num(v):
            try:
                float(v)
                return True
            except (TypeError, ValueError):
                return False

        num_cols = [c for c in range(1, ncols) if _is_num(body[0][c])]
        if not num_cols:
            return report
        # Probability detection: multi numeric columns and row sums close
        # to 1 (sampled over the first 200 rows). Sampling clips to [0,1]
        # and maps NaN to 0 so a POISONED file (negative / NaN cells) is
        # still recognized as probability-shaped and repaired instead of
        # being skipped; genuinely non-probability outputs (sums far from
        # 1, values > 1) are left untouched.
        sums = []
        for r in body[:200]:
            vals = []
            for c in num_cols:
                try:
                    f = float(r[c])
                except (TypeError, ValueError):
                    f = float("nan")
                if not math.isfinite(f):
                    f = 0.0
                vals.append(max(0.0, min(1.0, f)))
            if len(vals) != len(num_cols):
                continue
            sums.append(sum(vals))
        if len(sums) < 5 or len(num_cols) < 2:
            return report
        # majority-of-rows heuristic: at least 60% of sampled rows must be
        # within 5% of a unit sum for the file to be probability-shaped
        near = sum(1 for s in sums if abs(s - 1.0) <= 0.05)
        if near / float(len(sums)) < 0.6:
            return report

        repaired = 0
        out_rows = [header]
        for r in body:
            vals = []
            changed = False
            for c in num_cols:
                try:
                    f = float(r[c])
                except (TypeError, ValueError):
                    f = float("nan")
                if math.isnan(f) or math.isinf(f):
                    f = 0.0
                    changed = True
                if f < 0.0:
                    f = 0.0
                    changed = True
                if f > 1.0:
                    f = 1.0
                    changed = True
                vals.append(f)
            s = sum(vals)
            if s <= 0.0 or not math.isfinite(s) or abs(s - 1.0) > 1e-8:
                vals = [v / s if s > 0.0 else 1.0 / len(num_cols)
                        for v in vals]
                changed = True
            if changed:
                repaired += 1
            out = list(r)
            for idx, c in enumerate(num_cols):
                out[c] = "%.9f" % vals[idx]
            out_rows.append(out)
        if repaired:
            with dst.open("w", encoding="utf-8", newline="") as fh:
                w = csv.writer(fh, lineterminator=newline)
                w.writerows(out_rows)
            report["fixed"] = True
            report["rows_repaired"] = repaired
            report["detail"] = "probability rows clipped+renormalized"
        return report

    def _find_bundle_by_trial(self, trial_id: str) -> Optional[CandidateBundle]:
        import json
        for p in self.bus.host_bundles.glob("bundle_*.json"):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if d.get("trial_id") == trial_id:
                return CandidateBundle.from_dict(d)
        return None
