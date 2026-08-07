# -*- coding: utf-8 -*-
"""hera/prioritization.py - Research decision authority: PrioritizationTicket.

HERA selects one branch + one mutation axis + trial budget, freezes a
ResearchProgramGrant and marks it SnapshotReadyV3 on the File-as-Bus.
PACT does not choose branches: it only executes what this ticket grants.

v2.2.1: the prioritizer carries the same belief system as the rest of HERA -
world-class research authority, evidence-over-intuition discipline, and a
fast-cadence contract (small safe steps, hundreds of rounds per run).
"""
import json
import re
import time
from typing import Callable, Dict, Optional

from hera.portfolio import VALID_INTENTS
from v2_contracts import (
    PrioritizationTicket, ResearchPlan, ResearchProgramGrant, SnapshotReadyV3,
    canonical_hash, new_id, now_iso,
)
from v2_llm import default_llm_call


def _extract_json(response: str) -> Optional[dict]:
    match = re.search(r"\{.*\}", response or "", re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
        return data if isinstance(data, dict) else None
    except (ValueError, TypeError):
        return None


class Prioritizer:
    """HERA prioritization: branch + mutation axis + budget -> frozen grant."""

    def __init__(self, llm_call_fn: Optional[Callable[[str], str]] = None):
        self.llm_call = llm_call_fn or default_llm_call

    def build_ticket_prompt(self, profile, portfolio, plan: ResearchPlan,
                            trial_budget: int, research_intent: str = "",
                            stage: str = "", platform_facts: str = "") -> str:
        branches_txt = []
        for b in portfolio.branches:
            branches_txt.append("  - %s | model=%s | axes=%s | %s" % (
                b.branch_id, b.model_family,
                ",".join(b.allowed_mutation_axes), (b.description or "")[:120]))
        return (
            "ROLE: YOU ARE THE HERA RESEARCH DECISION AUTHORITY - the most "
            "rigorous experiment-prioritization expert in ML. You OWN the "
            "method space: you can EXPLOIT an existing branch, or - when the "
            "evidence shows the current space is exhausted - WRITE new "
            "branches into the portfolio (genuinely new directions: a "
            "different model family, preprocessing, architecture, or "
            "ensemble). You choose the ONE candidate branch and ONE mutation "
            "axis that maximize expected information gain per minute of "
            "compute. Evidence over intuition: exploit what is working, "
            "explore cheaply what is unknown, never burn budget on a branch "
            "that just failed, and never propose a new branch unless the "
            "existing space really cannot express the best next experiment. "
            "Reason first, then choose. The research loop iterates hundreds "
            "of rounds - prefer small, safe, high-information steps over big "
            "gambles.\n"
            "Competition: " + getattr(profile, "competition", "unknown") + "\n"
            "Task type: " + getattr(profile, "task_type", "other") + "\n"
            "Current method portfolio (branch | model | axes | description):\n"
            + "\n".join(branches_txt) + "\n"
            "Plan hypothesis: " + plan.hypothesis + "\n"
            "Plan approach: " + plan.approach_type + "\n"
            "Research intent: " + (research_intent or "(platform default)")
            + "\n"
            "Stage: " + (stage or "S1_baseline") + "\n"
            "Provisional trial budget (children, from the plan intent): "
            + str(trial_budget)
            + " - you are the FINAL research-intent authority: choose "
              "research_intent carefully, because the platform re-derives "
              "the child-trial count from YOUR final intent.\n"
            "Resource profile (trial seconds cap / CV folds cap): "
            + str(portfolio.resource_profile.get("max_budget_seconds", 1800))
            + "s / "
            + str(portfolio.resource_profile.get("max_folds", 2))
            + " - prefer experiments that finish inside these limits.\n"
            + (("Measured platform facts (trust them over guesses):\n"
                 + str(platform_facts) + "\n") if platform_facts else "")
            + self._deep_block(profile)
            + "\n"
            'Return JSON: {"selected_branch_id":"branch_id from the portfolio",'
            ' "mutation_axis":"one allowed axis of the selected branch",'
            ' "research_intent":"one allowed intent (match the plan unless '
            'the evidence demands otherwise)",'
            ' "reason":"concise, evidence-based justification",'
            ' "new_branches":[]}'
            "new_branches (0-2 entries, ONLY when the existing space is "
            "exhausted): each entry is {\"branch_id\": unique alphanumeric/"
            "underscore name, \"model_family\": e.g. lightgbm/xgboost/torch_cnn/"
            "ensemble, \"description\": one sentence, \"allowed_mutation_axes\": "
            "subset of [hyperparameter, feature, model, preprocessing, "
            "ensemble, data, architecture], \"defaults\": {\"model\": ..., "
            "\"features\": ..., \"params\": {...}}} - a branch you create can "
            "be selected immediately."
        )
    def _deep_block(self, profile) -> str:
        """v2.4 M1: measured deep diagnostics + difficulty ladder as facts."""
        lines = ["Measured deep diagnostics (from the actual train sample):"]
        dd = getattr(profile, "deep_diagnostics", None) or {}
        if not dd or dd.get("error"):
            lines.append("  (unavailable)")
        else:
            t = dd.get("target_diag") or {}
            f = dd.get("feature_diag") or {}
            o = dd.get("order_diag") or {}
            lines.append("  target: classes=%s top1_share=%s entropy_bits=%s "
                         "skew=%s unique_ratio=%s"
                         % (t.get("n_classes"), t.get("top1_share"),
                            t.get("entropy_bits"), t.get("skew"),
                            t.get("unique_ratio")))
            lines.append("  features: n=%s numeric_share=%s constant=%s "
                         "duplicate_groups=%s high_card=%s"
                         % (f.get("n_columns"), f.get("numeric_share"),
                            f.get("constant_cols"),
                            len(f.get("duplicate_cols") or []),
                            f.get("high_card_cols")))
            lines.append("  order: id_monotonic=%s id_target_corr=%s "
                         "time=%s..%s"
                         % (o.get("id_monotonic"), o.get("id_target_corr"),
                            o.get("time_min"), o.get("time_max")))
        dl = getattr(profile, "difficulty_ladder", None) or {}
        if dl and dl.get("constant") is not None:
            lines.append("Measured difficulty ladder (baseline scores, "
                         "direction=%s):"
                         % getattr(profile, "metric_direction", "?"))
            lines.append("  constant=%s linear=%s gbdt=%s headroom=%s"
                         % (dl.get("constant"), dl.get("linear"),
                            dl.get("gbdt"), dl.get("headroom")))
        else:
            lines.append("Difficulty ladder: (not yet measured - the first "
                         "grant builds it)")
        return "\n".join(lines) + "\n"

    def prioritize(self, profile, portfolio, plan: ResearchPlan,
                   trial_budget: int = 3, research_intent: str = "",
                   stage: str = "", platform_facts: str = "") -> PrioritizationTicket:
        # NOTE (v2.2.1): the returned intent is the FINAL intent authority;
        # the closed loop re-validates it against the stage whitelist and
        # re-derives children BEFORE freezing the grant.
        import sys as _sys
        prompt = self.build_ticket_prompt(profile, portfolio, plan,
                                          trial_budget,
                                          research_intent=research_intent,
                                          stage=stage,
                                          platform_facts=platform_facts)
        intent = str(research_intent or "").strip()
        branch_id = "baseline"
        mutation_axis = "hyperparameter"
        try:
            data = _extract_json(self.llm_call(prompt))
            if isinstance(data, dict):
                # HERA may WRITE new branches into the method space.
                new_branches = data.get("new_branches") or []
                if isinstance(new_branches, list):
                    for nb in new_branches[:2]:
                        if not isinstance(nb, dict):
                            continue
                        err = portfolio.add_branch(nb)
                        if err:
                            print("[prioritizer] new branch rejected: %s"
                                  % err, file=_sys.stderr, flush=True)
                        else:
                            print("[prioritizer] HERA wrote new branch: %s (%s)"
                                  % (nb.get("branch_id"),
                                     (nb.get("description") or "")[:80]),
                                  file=_sys.stderr, flush=True)
                candidate = str(data.get("selected_branch_id") or "").strip()
                if candidate in portfolio.branch_ids():
                    branch_id = candidate
                else:
                    print("[prioritizer] LLM chose unknown branch %r -> fallback %s"
                          % (candidate, branch_id), file=_sys.stderr, flush=True)
                axis = str(data.get("mutation_axis") or "").strip()
                if axis in portfolio.allowed_axes(branch_id):
                    mutation_axis = axis
                else:
                    print("[prioritizer] LLM chose invalid axis %r for %s -> fallback %s"
                          % (axis, branch_id, mutation_axis),
                          file=_sys.stderr, flush=True)
                llm_intent = str(data.get("research_intent") or "").strip()
                if llm_intent in VALID_INTENTS:
                    intent = llm_intent
                else:
                    print("[prioritizer] LLM chose invalid intent %r -> keep %r"
                          % (llm_intent, intent), file=_sys.stderr, flush=True)
        except Exception as _exc:  # noqa: BLE001 - deterministic fallback
            print("[prioritizer] LLM failed, fallback baseline/hyperparameter: %s"
                  % str(_exc)[:200], file=_sys.stderr, flush=True)

        # Merge the selected branch direction into the plan so PACT's
        # implementer actually receives the HERA-chosen method space.
        branch = portfolio.get_branch(branch_id)
        merged = dict(plan.method_detail or {})
        merged.update(dict(branch.defaults or {}))
        merged["branch_id"] = branch_id
        if branch.description:
            merged["branch_description"] = branch.description
        plan.method_detail = merged

        ticket = PrioritizationTicket(
            ticket_id=new_id("ticket"),
            selected_branch_id=branch_id,
            mutation_axis=mutation_axis,
            research_intent=intent,
            trial_budget=max(1, int(trial_budget)),
            method_portfolio_hash=portfolio.portfolio_hash,
            plan_hash=canonical_hash(plan.to_dict()),
            created_at=now_iso(),
        )
        ticket.ticket_hash = ticket.compute_hash()
        return ticket

    def freeze_grant(self, competition: str, task_prompt: str, plan: ResearchPlan,
                     ticket: PrioritizationTicket,
                     stage: str = "") -> ResearchProgramGrant:
        grant = ResearchProgramGrant(
            grant_id=new_id("grant"),
            competition=competition,
            task_prompt=task_prompt[:500],
            directive_hash=canonical_hash({
                "competition": competition,
                "plan": plan.to_dict(),
            }),
            plan=plan.to_dict(),
            ticket=ticket.to_dict(),
            selected_branch_id=ticket.selected_branch_id,
            mutation_axis=ticket.mutation_axis,
            research_intent=ticket.research_intent,
            stage=stage or "",
            trial_budget=ticket.trial_budget,
            max_budget_seconds=int(plan.max_budget_seconds or 3600),
            created_at=now_iso(),
            status="frozen",
        )
        grant.grant_hash = grant.compute_hash()
        return grant

    def snapshot_ready(self, grant: ResearchProgramGrant) -> SnapshotReadyV3:
        return SnapshotReadyV3(
            grant_id=grant.grant_id,
            snapshot_hash=grant.grant_hash,
            status="ready",
            created_at=now_iso(),
        )
