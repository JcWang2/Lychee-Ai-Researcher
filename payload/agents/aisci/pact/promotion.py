# -*- coding: utf-8 -*-
"""pact/promotion.py - PromotionManager: compare incumbent, promote/reject.

Host-owned. Keeps a certified-best pointer (PromotionRecord) in
pact_control_host/promotion_records/. Only verified, evaluator-receipt-backed
metrics participate in promotion.
"""
from typing import Optional

from v2_contracts import PromotionRecord, now_iso


class PromotionManager:
    """Promotes trials against the certified incumbent."""

    def __init__(self, bus, metric_direction: str = "higher_is_better",
                 min_delta: float = 0.01):
        self.bus = bus
        self.metric_direction = metric_direction
        self.min_delta = float(min_delta)

    def _better(self, candidate: float, incumbent: Optional[float]) -> bool:
        if incumbent is None:
            return True
        if self.metric_direction == "lower_is_better":
            return candidate < incumbent - self.min_delta
        return candidate > incumbent + self.min_delta

    def promote(self, trial_id: str, metric: Optional[float],
                evidence: str = "", verified: bool = True) -> PromotionRecord:
        current = self.bus.load_promotion() or {}
        incumbent_id = current.get("certified_best_trial_id", "")
        incumbent_metric = current.get("certified_best_metric")

        if metric is None or not verified:
            record = PromotionRecord(
                competition=(current.get("competition") or ""),
                certified_best_trial_id=incumbent_id,
                certified_best_metric=incumbent_metric,
                incumbent_trial_id=trial_id,
                incumbent_metric=(metric if not verified else None),
                decision="reject",
                reason=("no verifiable metric" if metric is None
                        else "unverified trial (rc!=0): not promoted"),
                updated_at=now_iso(),
            )
        elif self._better(metric, incumbent_metric):
            record = PromotionRecord(
                competition=(current.get("competition") or ""),
                certified_best_trial_id=trial_id,
                certified_best_metric=metric,
                incumbent_trial_id=incumbent_id,
                incumbent_metric=incumbent_metric,
                decision="promote",
                reason=evidence or "new certified best",
                updated_at=now_iso(),
            )
        else:
            record = PromotionRecord(
                competition=(current.get("competition") or ""),
                certified_best_trial_id=incumbent_id,
                certified_best_metric=incumbent_metric,
                incumbent_trial_id=trial_id,
                incumbent_metric=metric,
                decision="reject",
                reason="did not beat certified incumbent",
                updated_at=now_iso(),
            )
        self.bus.save_promotion(record.to_dict())
        return record

    def certified_best(self) -> PromotionRecord:
        current = self.bus.load_promotion() or {}
        return PromotionRecord.from_dict(current)
