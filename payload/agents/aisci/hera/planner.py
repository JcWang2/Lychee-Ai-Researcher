# -*- coding: utf-8 -*-
"""hera/planner.py - Plan making (analysis result + evidence -> hypothesis).

HERA produces a ResearchPlan (hypothesis + method + budget) using the LLM,
with deterministic fallbacks. Candidate code is NOT produced here - writing
the implementation belongs to PACT's implementer step.

v2.2.1: the planner carries the same code-master belief system as PACT -
world-class scientist persona, evidence discipline, and a fast-cadence
contract (2-6 min trials) so a 24h run can iterate hundreds of rounds.
"""
import json
import re
from typing import Callable, Optional

from hera.portfolio import VALID_INTENTS
from method_selector import DatasetContract, MethodSelector
from v2_contracts import AnalysisProfile, ResearchPlan
from v2_llm import default_llm_call

_FALLBACK_PLAN = {
    "hypothesis": "Fallback: standard ML pipeline",
    "approach_type": "explore",
    "expected_improvement": "baseline",
    "risk": "Low",
    "research_intent": "",
    "method_detail": {"model": "random_forest", "features": "all"},
    "max_budget_seconds": 1200,
}


def _deep_prompt_line(profile) -> str:
    """v2.4 M1: one compact evidence line with deep diagnostics + ladder."""
    dd = getattr(profile, "deep_diagnostics", None) or {}
    dl = getattr(profile, "difficulty_ladder", None) or {}
    parts = []
    t = dd.get("target_diag") or {}
    f = dd.get("feature_diag") or {}
    o = dd.get("order_diag") or {}
    if t:
        parts.append("classes=%s" % t.get("n_classes"))
        if t.get("top1_share") is not None:
            parts.append("top1=%.2f" % t["top1_share"])
        if t.get("skew") is not None:
            parts.append("skew=%.2f" % t["skew"])
    if f:
        if f.get("numeric_share") is not None:
            parts.append("num_share=%.2f" % f["numeric_share"])
        if f.get("constant_cols"):
            parts.append("const=%s"
                         % ",".join(str(c) for c in f["constant_cols"][:3]))
    if o:
        if o.get("id_monotonic") is not None:
            parts.append("id_mono=%d" % (1 if o["id_monotonic"] else 0))
        if o.get("time_present"):
            parts.append("time=%s..%s" % (o.get("time_min"), o.get("time_max")))
    if dl and dl.get("constant") is not None:
        parts.append("ladder_c=%.4f/l=%.4f/g=%.4f"
                     % (dl.get("constant"), dl.get("linear"), dl.get("gbdt")))
    if not parts:
        return ""
    return "DEEP DIAGNOSTICS: " + "; ".join(parts) + "\n\n"


def _extract_json(response: str) -> Optional[dict]:
    match = re.search(r"\{.*\}", response or "", re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
        return data if isinstance(data, dict) else None
    except (ValueError, TypeError):
        return None


def _has_pretrained(resource: dict) -> bool:
    """cached_weights arrives as a count (ResourceProfiler) or a name list
    (direct callers); both are accepted without branching on call sites."""
    cw = resource.get("cached_weights", 0) or 0
    if isinstance(cw, (list, tuple, set)):
        cw = len(cw)
    try:
        return int(cw) > 0
    except (TypeError, ValueError):
        return False


class Planner:
    """Turns analysis + evidence into a ResearchPlan (no code)."""

    def __init__(self, llm_call_fn: Optional[Callable[[str], str]] = None,
                 selector: Optional[MethodSelector] = None):
        self.llm_call = llm_call_fn or default_llm_call
        self.selector = selector  # lazy default on first use

    def _contract(self, profile: AnalysisProfile,
                  resource: Optional[dict]) -> DatasetContract:
        resource = dict(resource or {})
        return DatasetContract(
            modality=str(getattr(profile, "modality", "") or "tabular"),
            task_type=str(getattr(profile, "task_type", "") or "other"),
            metric_family=str(getattr(profile, "metric_name", "") or "accuracy"),
            n_rows=int(getattr(profile, "train_rows", 0) or 0),
            n_classes=int(getattr(profile, "n_classes", 0) or 0),
            gpu_available=bool(int(resource.get("gpu_memory_mb", 0) or 0) > 0),
            budget_seconds=float(resource.get("max_budget_seconds") or 1800),
            has_pretrained=_has_pretrained(resource),
            image_cache=bool(resource.get("cache_profile") or False),
            text_columns=len(getattr(profile, "text_columns", []) or []),
        )

    def prior_block(self, profile: AnalysisProfile,
                    resource: Optional[dict], k: int = 6) -> str:
        """Empirical PRIOR candidates from the registry + experience table.

        Reference ONLY: the final research decision (choose / combine /
        create) always stays with the planner/Analyzer. This block exists to
        save rounds, never to constrain creativity.
        """
        if self.selector is None:
            self.selector = MethodSelector()
        contract = self._contract(profile, resource)
        cands = self.selector.candidates(contract, k=k)
        if not cands:
            return ""
        lines = []
        for c in cands:
            prior_txt = ("+%.3f(n=%d)" % (c.prior_lift, c.prior_n)
                         if c.prior_n else "no-data")
            lines.append("  - %s  score=%.2f est_cost=%.0fs prior=%s"
                         % (c.method_id, c.score, c.cost_estimate_seconds,
                            prior_txt))
        return ("PRIOR KNOWLEDGE (empirical reference ONLY - you may follow, "
                "combine, or reject these candidates; the final research "
                "decision is always yours):\n"
                + "\n".join(lines)
                + "\n(These come from the capability registry + cross-task "
                  "experience table; they never override your judgment.)\n\n")

    def build_plan_prompt(self, profile: AnalysisProfile, evidence: str,
                          round_num: int, elapsed: int, total_budget: int,
                          resource: Optional[dict] = None,
                          stage_block: str = "",
                          intent_hints: Optional[dict] = None) -> str:
        resource = dict(resource or {})
        budget_max = int(resource.get("max_budget_seconds") or 1800)
        folds_max = int(resource.get("max_folds") or 2)
        epochs_txt = ("%s-%s" % (resource.get("epochs_min", 3),
                                   resource.get("epochs_max", 8))
                      if resource.get("epochs_min") else "as needed")
        image_txt = ("<=%spx" % resource["image_size_max"]
                     if resource.get("image_size_max") else "as needed")
        rows_txt = ("<=%s rows" % resource["train_rows_cap"]
                    if resource.get("train_rows_cap") else "all rows")
        resource_txt = (
            "RESOURCE PROFILE (hard constraints for THIS competition):\n"
            "- max trial seconds: %d\n- max CV folds: %d\n"
            "- image size cap: %s\n- epochs: %s\n- train rows cap: %s\n"
            "Fast partial-data training that FINISHES beats a full-data "
            "model that times out (rc=-9). No full-data retraining after "
            "validation.\n\n"
            % (budget_max, folds_max, image_txt, epochs_txt, rows_txt))
        stage_txt = ""
        if stage_block:
            stage_txt = "STAGE GUIDANCE (platform soft guide - you may still "                        "choose any method, but stay inside the intent whitelist):\n"                        + stage_block.strip() + "\n\n"
        intent_txt = ""
        if intent_hints and intent_hints.get("allowed"):
            ranges = intent_hints.get("child_ranges") or {}
            intent_txt = (
                "ALLOWED RESEARCH INTENTS for this stage (choose ONE in the "
                "JSON below): " + ", ".join(sorted(intent_hints["allowed"]))
                + "\nIntent -> child-trial whitelist: "
                + json.dumps(ranges, sort_keys=True) + "\n\n")
        prompt = (
            "ROLE: YOU ARE A WORLD-CLASS AI RESEARCH SCIENTIST.\n"
            "You are a Kaggle Grandmaster-level experimentalist with a "
            "master's discipline: every experiment you plan must be "
            "falsifiable, minimally invasive, and cheap enough to run in "
            "minutes. The research loop runs many rounds, so "
            "your job is to plan ONE focused, safe experiment per round - "
            "verified steps beat big rewrites. You respect the "
            "evidence above: you never repeat a failed direction, you never "
            "overfit the leaderboard, and you always prefer the simplest "
            "change that could plausibly improve the metric. You design "
            "experiments that can actually execute inside the trial budget "
            "(single stratified validation split with early stopping, at most "
            + str(folds_max)
            + " folds, pretrained weights only from the verified cache - "
              "never downloads, runtime inside the RESOURCE PROFILE budget).\n\n"
            + "OFFICIAL METRIC: " + str(getattr(profile, "metric_label", "") or "accuracy")
              + " (" + ("maximize" if getattr(profile, "metric_direction", "higher_is_better") == "higher_is_better" else "minimize") + ")\n"
            + "TASK: " + (profile.task_prompt or profile.data_notes)[:400] + "\n\n"
            + "DATA PROFILE (deterministic analysis - treat as ground truth):\n"
            + profile.data_notes + "\n\n"
            + _deep_prompt_line(profile)
            + resource_txt
            + stage_txt
            + intent_txt
            + self.prior_block(profile, resource)
            + "EVIDENCE:\n" + evidence + "\n\n"
            + "Round " + str(round_num)
            + ". Time elapsed: " + str(int(elapsed)) + "s / " + str(total_budget) + "s.\n\n"
            + """Return a JSON plan:
{
  "hypothesis": "falsifiable hypothesis tied to the evidence",
  "approach_type": "explore | exploit | ablation | transfer",
  "expected_improvement": "description",
  "risk": "Low | Medium | High",
  "research_intent": "one allowed intent (see ALLOWED RESEARCH INTENTS)",
  "children": 3,
  "method_detail": {"model": "concrete fast model name", "features": "feature_set", "params": {}},
  "max_budget_seconds": 600
}
"""
            + "Guidance: choose max_budget_seconds between 300 and "
              + str(budget_max)
              + " (per the RESOURCE PROFILE above; long enough to converge "
                "a small fast model); research_intent MUST be one of the "
                "ALLOWED RESEARCH INTENTS and children MUST be an integer "
                "inside the intent's whitelist range (the platform clamps "
                "it); method_detail must name a concrete model available in "
                "the verified capability registry "
                "(sklearn/xgboost/lightgbm/torch/torchvision/timm are "
                "confirmed installed); never plan downloads (pretrained "
                "weights must come from the preflight-verified cache) and "
                "keep folds <= "
              + str(folds_max)
              + ".\n"
        )
        return prompt

    def plan(self, profile: AnalysisProfile, evidence: str, round_num: int,
             elapsed: int, total_budget: int,
             resource: Optional[dict] = None,
             stage_block: str = "",
             intent_hints: Optional[dict] = None) -> ResearchPlan:
        prompt = self.build_plan_prompt(profile, evidence, round_num,
                                        elapsed, total_budget,
                                        resource=resource,
                                        stage_block=stage_block,
                                        intent_hints=intent_hints)
        response = self.llm_call(prompt)
        data = _extract_json(response)
        if not isinstance(data, dict):
            data = {}
        resource = dict(resource or {})
        budget_max = int(resource.get("max_budget_seconds") or 1800)
        plan_data = dict(_FALLBACK_PLAN)
        for key in ("hypothesis", "approach_type", "expected_improvement",
                    "risk", "method_detail", "research_intent",
                    "max_budget_seconds"):
            if key in data and data[key] not in (None, ""):
                plan_data[key] = data[key]
        if "children" in data:
            try:
                plan_data["children"] = int(data["children"])
            except (TypeError, ValueError):
                plan_data["children"] = 0
        method_detail = plan_data.get("method_detail", {})
        if not isinstance(method_detail, dict):
            method_detail = {}
        method_detail["resource_profile"] = dict(resource)
        # HERA-chosen research intent (platform validates against the stage
        # whitelist in the closed loop; children are clamped per intent).
        intent = str(plan_data.get("research_intent") or "").strip()
        if intent not in VALID_INTENTS:
            intent = ""
        try:
            children = int(plan_data.get("children"))
        except (TypeError, ValueError):
            children = 0
        if children >= 1:
            method_detail["children"] = children
        plan_data["method_detail"] = method_detail
        plan_budget = int(plan_data.get("max_budget_seconds") or budget_max)
        plan_budget = max(300, min(budget_max, plan_budget))
        return ResearchPlan(
            round_num=round_num,
            hypothesis=str(plan_data.get("hypothesis", _FALLBACK_PLAN["hypothesis"])),
            approach_type=str(plan_data.get("approach_type", _FALLBACK_PLAN["approach_type"])),
            method_detail=method_detail,
            expected_improvement=str(plan_data.get("expected_improvement", "")),
            risk=str(plan_data.get("risk", "Low")),
            research_intent=intent,
            max_budget_seconds=plan_budget,
        )
