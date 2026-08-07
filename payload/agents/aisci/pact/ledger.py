# -*- coding: utf-8 -*-
"""pact/ledger.py - Evidence ledger for verified TrialReceipts.

Reuses the V1 experiment ledger; only PACT-verified receipts are recorded.
"""
from pathlib import Path
from typing import List

from v2_contracts import ResearchPlan, TrialReceipt
from evidence_store import ExperimentLedger, ExperimentRecord


class PactLedger:
    """Records verified receipts and answers evidence queries for the loop."""

    def __init__(self, state_dir):
        self._ledger = ExperimentLedger(Path(state_dir))

    def append(self, receipt: TrialReceipt, plan: ResearchPlan, code: str = "") -> None:
        record = ExperimentRecord(
            competition=receipt.competition,
            round_num=receipt.round_num,
            trial_id=receipt.receipt_id,
            hypothesis=plan.hypothesis,
            approach_type=plan.approach_type,
            method_detail=plan.method_detail,
            code_path=receipt.submission_path,
            code_hash=receipt.code_hash,
            code_snippet=(code or "")[:1500],
            returncode=receipt.returncode,
            metric=receipt.metric,
            metric_name=receipt.metric_name,
            verdict=receipt.verdict,
            evidence=receipt.evidence,
            blind_spots="",
            causal_attribution="",
            next_suggestion="",
            stderr_snippet=(receipt.stderr or "")[:2000],
            wall_clock_seconds=receipt.wall_clock_seconds,
            parent_trial_id="",
            submission_exists=receipt.submission_exists,
        )
        self._ledger.append(record)

    def best_metric(self, competition: str):
        return self._ledger.get_best_metric(competition)

    def recent_rounds(self, competition: str, n: int = 3) -> List[dict]:
        return self._ledger.get_recent_rounds(competition, n=n)

    def stagnation(self, competition: str, window: int = 3) -> int:
        return self._ledger.count_stagnation(competition, window=window)

    def trials(self, competition: str) -> List[dict]:
        return self._ledger.get_trials_for_competition(competition)
