# -*- coding: utf-8 -*-
"""method_selector.py - v2.5.0 declarative method selection.

Architecture rule (frozen v2.5 contract):
  Method selection is DECLARATIVE. A dataset contract is matched against
  capability metadata (modalities/tasks/metric outputs/GPU/resource model)
  plus an empirical experience table. There is NO if/else chain that names a
  method, a modality, a metric, or a competition anywhere in this module (or
  in any routing layer). Adding a method = adding a registry entry; adding
  knowledge = adding a data row to the experience table.

This module answers:
  - which capabilities are usable for this contract?   (metadata filter)
  - how should they be ranked for this budget?         (cost + prior score)
It does NOT decide the final pick: the planner/HERA layer may choose any
candidate (or none) and is responsible for research choices.
"""
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from capability_registry import CapabilityRegistry, MethodSpec


# ---------------------------------------------------------------- contract
@dataclass(frozen=True)
class DatasetContract:
    """Data-shaped contract derived from analysis (no task identity)."""
    modality: str = "tabular"
    task_type: str = "classification"
    metric_family: str = "accuracy"
    n_rows: int = 0
    n_classes: int = 0
    gpu_available: bool = False
    budget_seconds: float = 300.0
    has_pretrained: bool = False
    image_cache: bool = False
    text_columns: int = 0
    multi_target: bool = False


_SCALE_BUCKETS = (
    (1_000_000, "huge"),
    (100_000, "large"),
    (10_000, "medium"),
    (1_000, "small"),
    (0, "tiny"),
)


def scale_bucket(n_rows: int) -> str:
    """Data-driven size bucket (declarative threshold table)."""
    for threshold, name in _SCALE_BUCKETS:
        if n_rows >= threshold:
            return name
    return "tiny"


# ------------------------------------------------------------ experience
@dataclass
class ExperienceTable:
    """Cross-task empirical prior. Pure data rows:

    key  = (modality, metric_family, scale_bucket, method_family)
    row  = {"n": observations, "mean_lift": mean log-score lift,
            "mean_cost_ratio": mean measured-cost/budget ratio}
    """
    rows: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str = "") -> "ExperienceTable":
        if not path:
            path = os.environ.get("V2_EXPERIENCE_JSON", "")
        if not path or not os.path.isfile(path):
            return cls()
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return cls(rows=data if isinstance(data, dict) else {})
        except (OSError, ValueError):
            return cls()

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.rows, fh, indent=1, sort_keys=True)

    def _key(self, contract: DatasetContract, method_id: str) -> str:
        family = method_id.split(".")[1] if method_id.count(".") >= 1 \
            else method_id
        return "|".join((contract.modality, contract.metric_family,
                         scale_bucket(contract.n_rows), family))

    def record(self, contract: DatasetContract, method_id: str,
               lift: float, cost_ratio: float) -> None:
        k = self._key(contract, method_id)
        row = self.rows.setdefault(k, {"n": 0, "mean_lift": 0.0,
                                       "mean_cost_ratio": 0.0})
        n = float(row["n"])
        row["mean_lift"] = (row["mean_lift"] * n + float(lift)) / (n + 1.0)
        row["mean_cost_ratio"] = (
            row["mean_cost_ratio"] * n + float(cost_ratio)) / (n + 1.0)
        row["n"] = int(n) + 1

    def prior(self, contract: DatasetContract,
              method_id: str) -> Tuple[float, float, int]:
        k = self._key(contract, method_id)
        row = self.rows.get(k)
        if not row:
            return 0.0, 0.0, 0
        return (float(row.get("mean_lift", 0.0)),
                float(row.get("mean_cost_ratio", 0.0)),
                int(row.get("n", 0)))


# ------------------------------------------------------------- scoring
@dataclass(frozen=True)
class ScoredMethod:
    method_id: str
    renderer: str
    family: str
    score: float
    cost_estimate_seconds: float
    prior_lift: float
    prior_n: int
    reasons: Tuple[str, ...] = ()


class MethodSelector:
    """Declarative candidate filter + ranker.

    Compatibility is evaluated ONLY by comparing contract fields against
    declared capability metadata. Ranking combines a resource-cost estimate
    (from the capability resource model) with the experience-table prior.
    No value-based branching: every rule is a metadata comparison, so a new
    modality/metric/scale needs zero code changes.
    """

    def __init__(self, registry: Optional[CapabilityRegistry] = None,
                 experience: Optional[ExperienceTable] = None):
        self.registry = registry or CapabilityRegistry()
        self.experience = experience or ExperienceTable()

    # --------------------------------------------------------- filter
    def compatible(self, spec: MethodSpec, contract: DatasetContract) -> bool:
        if spec.broken:
            return False
        if spec.supported_modalities and \
                contract.modality not in spec.supported_modalities:
            return False
        if spec.supported_tasks and \
                contract.task_type not in spec.supported_tasks:
            return False
        if spec.metric_outputs and \
                contract.metric_family not in spec.metric_outputs:
            return False
        if spec.gpu and not contract.gpu_available:
            return False
        return True

    # --------------------------------------------------------- cost
# resource_model -> (base, per_row, pretrained_discount, cache_discount)
# Fully declarative: adding a resource model = adding one row; discounts are
# metadata, not branches.
_COST_MODELS = {
    "timm_finetune_cost_v1": (120.0, 0.0, 0.7, 0.8),
    "tabular_gbdt_cost_v1": (45.0, 0.0004, 1.0, 1.0),
    "tabular_linear_cost_v1": (15.0, 0.0001, 1.0, 1.0),
    "text_tfidf_cost_v1": (30.0, 0.0002, 1.0, 1.0),
    "audio_baseline_cost_v1": (60.0, 0.0003, 1.0, 1.0),
    "timeseries_lag_cost_v1": (40.0, 0.0003, 1.0, 1.0),
    "image_embed_cost_v1": (60.0, 0.0002, 1.0, 1.0),
}


class MethodSelector:
    """Declarative candidate filter + ranker.

    Compatibility is evaluated ONLY by comparing contract fields against
    declared capability metadata. Ranking combines a resource-cost estimate
    (declarative cost table) with the experience-table prior. No value-based
    branching: every rule is a metadata comparison, so a new
    modality/metric/scale needs zero code changes.
    """

    def __init__(self, registry: Optional[CapabilityRegistry] = None,
                 experience: Optional[ExperienceTable] = None):
        self.registry = registry or CapabilityRegistry()
        self.experience = experience or ExperienceTable()

    # --------------------------------------------------------- filter
    def compatible(self, spec: MethodSpec, contract: DatasetContract) -> bool:
        if spec.broken:
            return False
        if spec.supported_modalities and \
                contract.modality not in spec.supported_modalities:
            return False
        if spec.supported_tasks and \
                contract.task_type not in spec.supported_tasks:
            return False
        if spec.metric_outputs and \
                contract.metric_family not in spec.metric_outputs:
            return False
        if spec.gpu and not contract.gpu_available:
            return False
        return True

    # --------------------------------------------------------- cost
    def cost_estimate(self, spec: MethodSpec,
                      contract: DatasetContract) -> float:
        base, per_row, pre_disc, cache_disc = _COST_MODELS.get(
            getattr(spec, "resource_model", "") or "",
            (60.0, 0.0, 1.0, 1.0))
        cost = base + per_row * max(0, contract.n_rows)
        if contract.has_pretrained:
            cost *= pre_disc
        if contract.image_cache:
            cost *= cache_disc
        return cost

    # --------------------------------------------------------- rank
    def score(self, spec: MethodSpec, contract: DatasetContract,
              cost: float, prior_lift: float, prior_n: int) -> float:
        # budget fit: prefer methods whose estimated cost fits the budget
        budget = max(1.0, float(contract.budget_seconds))
        fit = max(0.0, 1.0 - abs(cost - budget) / max(budget, 1.0))
        # experience prior: lift in log-score terms (mean over observations)
        prior = 0.0
        if prior_n > 0:
            prior = max(-1.0, min(1.0, prior_lift))
        # cheap-first bias shrinks as budget grows
        cheap_bias = 0.25 * max(0.0, 1.0 - budget / 3600.0)
        return 1.0 + 2.0 * fit + 3.0 * prior + cheap_bias

    # --------------------------------------------------------- entry
    def candidates(self, contract: DatasetContract,
                   k: Optional[int] = None) -> List[ScoredMethod]:
        out: List[ScoredMethod] = []
        for spec in self.registry.all():
            if not self.compatible(spec, contract):
                continue
            cost = self.cost_estimate(spec, contract)
            plift, pratio, pn = self.experience.prior(contract, spec.method_id)
            s = self.score(spec, contract, cost, plift, pn)
            out.append(ScoredMethod(
                method_id=spec.method_id,
                renderer=spec.renderer,
                family=spec.family,
                score=round(s, 4),
                cost_estimate_seconds=round(cost, 1),
                prior_lift=round(plift, 4),
                prior_n=pn,
                reasons=("metadata-compatible", "cost-fit",
                         "experience-prior" if pn else ""),
            ))
        out.sort(key=lambda m: (-m.score, m.method_id))
        return out[:k] if k else out
