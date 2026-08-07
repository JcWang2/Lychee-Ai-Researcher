# -*- coding: utf-8 -*-
"""pact/agent_client.py - ProgramAgentClient: agent-side proposer.

Design contract: the agent side ONLY proposes. It writes TrialProposal
messages to protocol/pending_agent/ and advances exclusively from host-owned
outcome envelopes in protocol/outcomes_visible/. It never executes, never
evaluates, and never promotes.
"""
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from v2_contracts import TrialProposal, new_id, now_iso
from pact.file_bus import FileBus


class ProgramAgentClient:
    """Agent-side proposer bound to one frozen ResearchProgramGrant."""

    def __init__(self, bus: FileBus, grant: dict, host_id: str = "agent",
                 proposer: Optional[Callable[[int, dict], dict]] = None):
        self.bus = bus
        self.grant = grant
        self.host_id = host_id
        self.proposer = proposer
        self._outcomes_seen: set = set()

    # ---- validation: bindings must match the frozen grant ----
    def validate_bindings(self) -> None:
        grant = self.grant
        missing = []
        for key in ("grant_id", "directive_hash", "selected_branch_id",
                    "mutation_axis", "trial_budget"):
            if not str(grant.get(key) or ""):
                missing.append(key)
        if missing:
            raise RuntimeError("program_agent_grant_missing:" + ",".join(missing))
        if grant.get("status") not in ("frozen", "active"):
            raise RuntimeError("program_agent_grant_not_frozen")

    # ---- proposal generation ----
    def _propose_next(self, child_index: int, evidence: str) -> dict:
        """Deterministic proposal: single mutation axis, param override.

        The external proposer (HERA FeedbackView) receives the prior-child
        evidence so child N+1 is adapted to child N's verified outcome.
        """
        proposal = {
            "proposal_id": new_id("proposal"),
            "grant_id": self.grant["grant_id"],
            "child_index": int(child_index),
            "hypothesis": "Child %d: mutate %s on branch %s"
                          % (child_index, self.grant["mutation_axis"],
                             self.grant["selected_branch_id"]),
            "mutation_axis": self.grant["mutation_axis"],
            "param_overrides": {"child_index": int(child_index)},
            "evidence": (evidence or "")[:1600],
            "created_at": now_iso(),
        }
        if self.proposer is not None:
            try:
                external = self.proposer(child_index, self.grant, evidence)
            except TypeError:
                try:
                    external = self.proposer(child_index, self.grant)
                except Exception:  # noqa: BLE001 - fall back to deterministic
                    external = None
            except Exception:  # noqa: BLE001 - fall back to deterministic
                external = None
            if isinstance(external, dict) and external.get("hypothesis"):
                proposal.update(external)
        return proposal

    def propose_next(self, child_index: int, evidence: str = "") -> TrialProposal:
        """Write ONE proposal for the next child (FeedbackView-driven).

        Called once per child, after the previous child's outcome has been
        collected, so the host service can consume proposals incrementally
        (single-process inline or as an independent resident daemon).
        """
        self.validate_bindings()
        payload = self._propose_next(int(child_index),
                                     evidence or self.feedback_view())
        self.bus.propose(payload)
        return TrialProposal.from_dict(payload)

    def propose_all(self) -> List[TrialProposal]:
        """Compatibility: propose the whole grant budget at once.

        Kept for callers that intentionally pre-propose (tests, batch mode).
        The feedback-driven loop prefers propose_next() + collect_outcomes().
        """
        self.validate_bindings()
        budget = int(self.grant["trial_budget"])
        return [self.propose_next(i) for i in range(1, budget + 1)]

    def _evidence_summary(self) -> str:
        outcomes = self.bus.list_outcomes()
        if not outcomes:
            return "No previous outcomes"
        lines = []
        for o in outcomes[-3:]:
            lines.append("R%s verdict=%s metric=%s"
                         % (o.get("child_index"), o.get("verdict"),
                            o.get("metric")))
        return "; ".join(lines)

    def feedback_view(self, max_children: int = 8) -> str:
        """Grant-internal FeedbackView: every prior child outcome in order.

        This is what makes the inner loop adaptive: the evidence is handed
        to the next propose_next() call, and from there into the implementer
        prompt, so child N+1 never repeats a dead end of child N.
        """
        outcomes = self.bus.list_outcomes()
        if not outcomes:
            return "No prior children completed in this grant yet"
        lines = []
        for o in sorted(outcomes,
                        key=lambda x: int(x.get("child_index") or 0)):
            lines.append(
                "child %s verdict=%s metric=%s rc=%s evidence=%s"
                % (o.get("child_index"), o.get("verdict"), o.get("metric"),
                   o.get("returncode"), (o.get("evidence") or "")[:160]))
        return "\n".join(lines[-max_children:])

    # ---- consume host outcomes ----
    def collect_outcomes(self) -> List[dict]:
        fresh = []
        for o in self.bus.list_outcomes():
            pid = o.get("proposal_id") or o.get("proposalId")
            if pid and pid not in self._outcomes_seen:
                self._outcomes_seen.add(pid)
                fresh.append(o)
        return fresh
