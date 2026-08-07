# -*- coding: utf-8 -*-
"""hera/interpreter.py - Scientific interpretation of verified receipts.

Turns a PACT TrialReceipt into a scientific conclusion (blind spots, causal
attribution, next direction) and a stop/switch decision. LLM-assisted with
deterministic fallbacks; the verdict itself comes from PACT, not the LLM.

v2.2.1: the interpreter carries the same belief system as the rest of HERA -
world-class scientist persona, root-cause-first attribution, honesty about
uncertainty, and ONE actionable next experiment per interpretation.
"""
import json
import re
from dataclasses import dataclass
from typing import Callable, Optional

from v2_llm import default_llm_call
from v2_contracts import TrialReceipt


@dataclass
class Interpretation:
    verdict: str = "unknown"
    delta: Optional[float] = None
    blind_spots: str = ""
    causal_attribution: str = ""
    next_direction: str = ""
    stop_decision: str = "continue"   # continue | stop | switch_approach


def _extract_json(response: str) -> Optional[dict]:
    match = re.search(r"\{.*\}", response or "", re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
        return data if isinstance(data, dict) else None
    except (ValueError, TypeError):
        return None


class Interpreter:
    """Interprets verified receipts into scientific conclusions."""

    def __init__(self, llm_call_fn: Optional[Callable[[str], str]] = None,
                 stagnation_limit: int = 6):
        self.llm_call = llm_call_fn or default_llm_call
        self.stagnation_limit = max(1, int(stagnation_limit))

    def interpret(self, receipt: TrialReceipt, best_before: Optional[float],
                  stagnation_count: int, max_rounds: int, elapsed: int,
                  total_budget: int) -> Interpretation:
        delta = None
        if receipt.metric is not None and best_before is not None:
            delta = receipt.metric - best_before

        blind_spots, causal, next_dir = self._llm_analysis(receipt)
        decision = self._stop_decision(
            receipt, stagnation_count, max_rounds, elapsed, total_budget)

        return Interpretation(
            verdict=receipt.verdict,
            delta=delta,
            blind_spots=blind_spots,
            causal_attribution=causal,
            next_direction=next_dir,
            stop_decision=decision,
        )

    def _llm_analysis(self, receipt: TrialReceipt) -> tuple:
        prompt = (
            "ROLE: YOU ARE A WORLD-CLASS ML SCIENTIST INTERPRETING A "
            "VERIFIED EXPERIMENT OUTCOME.\n"
            "Be rigorous and honest. Attribute the result to its ROOT "
            "CAUSE (data quality, features, model choice, training budget, "
            "or execution failure) using only the evidence below - never "
            "invent facts. A metric marked IN-SAMPLE in the evidence "
            "is a training-fit signal, not a generalization estimate: use "
            "it only for relative comparisons (model capacity, features, "
            "optimization) and prefer directions that also hold on "
            "held-out evidence when available. MLE-Bench scores the hidden "
            "test set, so the submission is the final ground truth. "
            "Hundreds of fast rounds are the intended mode; if the run "
            "failed or timed out, say exactly what to fix. Name the blind "
            "spots of this experiment and give "
            "exactly ONE actionable next experiment that would teach us the "
            "most per minute of compute. If the run failed, say what must "
            "be fixed before the next round.\n"
            "Hypothesis: (see evidence)\n"
            "Verdict: " + receipt.verdict + "\n"
            "Metric: " + (str(receipt.metric) if receipt.metric is not None else "None") + "\n"
            "Return code: " + str(receipt.returncode) + "\n"
            "Evidence: " + receipt.evidence + "\n"
            "Stderr (last 400 chars): " + (receipt.stderr or "")[-400:] + "\n\n"
            'Return JSON: {"blind_spots":"...","causal_attribution":"...","next_suggestion":"..."}'
        )
        try:
            data = _extract_json(self.llm_call(prompt))
            if isinstance(data, dict):
                return (str(data.get("blind_spots", "")),
                        str(data.get("causal_attribution", "")),
                        str(data.get("next_suggestion", "")))
        except Exception:  # noqa: BLE001
            pass
        return "", "Unknown", "Continue refining current approach"

    def _stop_decision(self, receipt: TrialReceipt, stagnation_count: int,
                       max_rounds: int, elapsed: int, total_budget: int) -> str:
        if receipt.round_num >= max_rounds:
            return "stop"
        if elapsed >= total_budget:
            return "stop"
        if stagnation_count >= self.stagnation_limit:
            return "stop"
        if stagnation_count >= max(4, self.stagnation_limit - 2):
            prompt = (
                "ROLE: YOU ARE THE HERA RESEARCH DIRECTOR deciding whether "
                "this research cycle is still productive.\n"
                "A master scientist stops when the marginal value of another "
                "round is negative, switches approach when the current "
                "branch is exhausted, and continues when there is clear, "
                "cheap room to improve. Judge using the evidence: stagnation "
                "trend, remaining time budget, and the best verified metric. "
                "Prefer 'continue' whenever a cheap improvement is plausible "
                "and budget remains - hundreds of fast rounds are the "
                "intended mode of this system.\n"
                "Round: " + str(receipt.round_num) + "/" + str(max_rounds) + "\n"
                "Stagnation: " + str(stagnation_count) + " rounds without improvement.\n"
                "Best metric: " + str(receipt.metric) + "\n"
                "Elapsed: " + str(int(elapsed)) + "s / " + str(total_budget) + "s\n\n"
                'Return JSON: {"decision": "continue | stop | switch_approach", "reason": "..."}'
            )
            try:
                data = _extract_json(self.llm_call(prompt))
                if isinstance(data, dict):
                    decision = str(data.get("decision", "continue"))
                    if decision in ("stop", "switch_approach"):
                        return decision
            except Exception:  # noqa: BLE001
                pass
            return "continue"
        return "continue"
