# -*- coding: utf-8 -*-
"""hera/memory.py - Scientific memory: strategy pool + causal evidence graph.

HERA maintains knowledge built from PACT-verified receipts:
  - strategy pool: reusable successful strategies
  - evidence graph: method x task x outcome causal accumulation
"""
from pathlib import Path
from typing import List, Optional

from evidence_graph import EvidenceGraph
from strategy_pool import StrategyPool
from v2_contracts import ResearchPlan, TrialReceipt


class ScientificMemory:
    """Wraps the V1 strategy pool and evidence graph for HERA."""

    def __init__(self, state_dir):
        base = Path(state_dir)
        base.mkdir(parents=True, exist_ok=True)
        self.strategy_pool = StrategyPool(base)
        self.evidence_graph = EvidenceGraph(base)

    def relevant_strategies(self, task_description: str, top_k: int = 3) -> List[dict]:
        return [s.to_dict() for s in
                self.strategy_pool.get_relevant(task_description, top_k=top_k)]

    def cross_task_knowledge(self, competition: str, top_k: int = 5) -> str:
        return self.evidence_graph.get_cross_task_knowledge(competition, top_k=top_k)

    def update(self, plan: ResearchPlan, receipt: TrialReceipt,
               task_description: str, best_before: Optional[float]) -> None:
        """Update knowledge from a verified receipt (facts only)."""
        delta = None
        if receipt.metric is not None and best_before is not None:
            delta = receipt.metric - best_before

        if receipt.verdict == "success" and delta is not None and delta > 0:
            self.strategy_pool.extract_strategy(
                hypothesis=plan.hypothesis,
                approach_type=plan.approach_type,
                verdict=receipt.verdict,
                metric_delta=delta,
                task_description=task_description[:100],
                round_num=receipt.round_num,
            )

        method_name = str(plan.method_detail.get("model", "unknown"))
        self.evidence_graph.record_trial_outcome(
            method_name=method_name,
            method_type=plan.approach_type,
            delta=delta,
            success=(receipt.verdict == "success"),
            task=receipt.competition,
            round_num=receipt.round_num,
            trial_id=receipt.receipt_id,
        )
