# -*- coding: utf-8 -*-
"""v2_contracts.py - Shared data contracts between HERA and PACT.

V2.1 adds the L1 transactional loop contracts required by the four-layer
design (outer research loop / inner PACT transactional loop / certified
publish layer / File-as-Bus):

  Outer loop:   AnalysisProfile -> ResearchPlan -> PrioritizationTicket
                -> ResearchProgramGrant (frozen) -> SnapshotReadyV3
  Inner loop:   TrialProposal (agent-writable) -> TrialSpec (host-frozen)
                -> CandidateBundle -> EvaluatorReceipt -> TrialReceipt
                -> PromotionRecord (certified-best pointer) -> ActionOutcome
  Publish:      ControlledPublisher accepts only certified-best bundles.
"""
import hashlib
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def new_id(prefix: str) -> str:
    return prefix + "_" + uuid.uuid4().hex[:12]


def canonical_hash(d: dict) -> str:
    blob = hashlib.sha256(
        repr(sorted(d.items(), key=lambda kv: str(kv[0]))).encode("utf-8"))
    return "sha256:" + blob.hexdigest()


def _filter_kwargs(d: dict, cls) -> dict:
    fields = set(getattr(cls, "__dataclass_fields__", {}).keys())
    return {k: v for k, v in d.items() if k in fields}


# A resolved image directory must contain at least this many magic-verified
# image files before the task is treated as image modality. Guards against
# icon/attachment dirs inside otherwise tabular tasks; real MLE-Bench image
# tasks always carry hundreds/thousands of files. Generic - never keyed on
# competition names.
IMAGE_FILE_MODALITY_THRESHOLD = 50
AUDIO_FILE_MODALITY_THRESHOLD = 50


@dataclass
class AnalysisProfile:
    """HERA Analyzer output: data + task profile."""
    competition: str = "unknown"
    task_prompt: str = ""
    task_type: str = "other"          # classification | regression | timeseries | segmentation | detection | other
    feature_columns: List[str] = field(default_factory=list)
    target_column: str = ""
    train_rows: int = 0
    test_rows: int = 0
    missing_columns: List[str] = field(default_factory=list)
    target_stats: Dict[str, Any] = field(default_factory=dict)
    numeric_columns: List[str] = field(default_factory=list)
    sample_values: Dict[str, List[str]] = field(default_factory=dict)
    data_notes: str = ""
    metric_name: str = "accuracy"
    metric_direction: str = "higher_is_better"
    metric_alignment: str = "exact"
    metric_label: str = "accuracy"
    metric_params: Dict[str, Any] = field(default_factory=dict)
    metric_min_delta: float = 0.01   # v2.3.6 per-metric improvement threshold
    # v2.2 generic resource signals (measured, never guessed)
    modality: str = "tabular"          # image | image_pixel | image_mask | image_detection | audio | tabular | text | mixed | unknown
    image_width: int = 0               # 0 when the task has no image dir
    image_height: int = 0
    image_channels: int = 0
    image_file_count: int = 0          # magic-verified image files (recursive scan)
    audio_file_count: int = 0          # audio files (wav/flac/ogg/mp3...) recursive scan
    mask_target: str = ""              # target column holding RLE-encoded masks (image_mask)
    bbox_columns: List[str] = field(default_factory=list)  # detected box coordinate columns
    multi_row_target: bool = False     # ids repeat in the train table (per-image boxes/masks)
    # v2.4 M1 deep diagnostics (measured by deep_profile.py; empty until the
    # analyzer fills them - prompts read them as evidence, never as guesses)
    deep_diagnostics: Dict[str, Any] = field(default_factory=dict)
    difficulty_ladder: Dict[str, Any] = field(default_factory=dict)
    # v2.3.2 generic content evidence (measured, never guessed)
    text_columns: List[str] = field(default_factory=list)  # content-verified free-text feature columns
    time_column: str = ""               # date/time column evidence for timeseries tasks
    feature_dim: int = 0               # numeric feature count (tabular) else column count
    n_classes: int = 0                 # distinct target values (classification)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AnalysisProfile":
        return cls(**_filter_kwargs(d, cls))


@dataclass
class ResearchPlan:
    """HERA Planner output: hypothesis and method (no code)."""
    round_num: int = 1
    hypothesis: str = "Fallback: standard ML pipeline"
    approach_type: str = "explore"    # explore | exploit | ablation | transfer
    method_detail: Dict[str, Any] = field(default_factory=dict)
    expected_improvement: str = "baseline"
    risk: str = "Low"
    research_intent: str = ""          # feasibility|repair|cheap_probe|local_exploitation|expensive_structural|confirmation|final_training
    max_budget_seconds: int = 3600

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ResearchPlan":
        return cls(**_filter_kwargs(d, cls))


@dataclass
class PrioritizationTicket:
    """HERA prioritization output: one branch + mutation axis + budget.

    PACT never selects the branch itself; it only executes what this ticket
    (and its frozen grant) authorizes.
    """
    ticket_id: str = ""
    ticket_hash: str = ""
    selected_branch_id: str = "baseline"
    mutation_axis: str = "hyperparameter"   # hyperparameter | feature | model | preprocessing
    research_intent: str = ""          # HERA-chosen intent (platform whitelists child count)
    trial_budget: int = 3
    method_portfolio_hash: str = ""
    plan_hash: str = ""
    created_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PrioritizationTicket":
        return cls(**_filter_kwargs(d, cls))

    def compute_hash(self) -> str:
        payload = dict(self.to_dict())
        payload.pop("ticket_hash", None)
        return canonical_hash(payload)


@dataclass
class ResearchProgramGrant:
    """Frozen research contract written to protocol/frozen_visible/.

    Host binds execution authority to this grant. Agent children must match
    directive / branch / mutation axis exactly.
    """
    grant_id: str = ""
    grant_hash: str = ""
    competition: str = "unknown"
    task_prompt: str = ""
    directive_hash: str = ""
    plan: Dict[str, Any] = field(default_factory=dict)
    ticket: Dict[str, Any] = field(default_factory=dict)
    selected_branch_id: str = "baseline"
    mutation_axis: str = "hyperparameter"
    research_intent: str = ""
    stage: str = ""
    trial_budget: int = 3
    max_budget_seconds: int = 3600
    created_at: str = ""
    status: str = "frozen"            # frozen | active | terminal

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ResearchProgramGrant":
        return cls(**_filter_kwargs(d, cls))

    def compute_hash(self) -> str:
        payload = dict(self.to_dict())
        payload.pop("grant_hash", None)
        return canonical_hash(payload)

    def ticket_obj(self) -> PrioritizationTicket:
        return PrioritizationTicket.from_dict(self.ticket or {})

    def plan_obj(self) -> ResearchPlan:
        return ResearchPlan.from_dict(self.plan or {})


@dataclass
class SnapshotReadyV3:
    """Frozen-grant readiness marker (protocol/frozen_visible/*.ready)."""
    grant_id: str = ""
    snapshot_hash: str = ""
    status: str = "ready"
    created_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SnapshotReadyV3":
        return cls(**_filter_kwargs(d, cls))


@dataclass
class TrialProposal:
    """Agent-side proposer output (protocol/pending_agent/).

    Contains one hypothesis + a SINGLE mutation + parameter overrides.
    The agent proposes; the host claims, validates and executes.
    """
    proposal_id: str = ""
    grant_id: str = ""
    child_index: int = 1
    hypothesis: str = ""
    mutation_axis: str = "hyperparameter"
    param_overrides: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrialProposal":
        return cls(**_filter_kwargs(d, cls))



@dataclass
class MethodInvocationV1:
    """v2.3 structured method invocation (HERA chooses, compiler renders).

    Analysis/HERA own the method + parameter choice. This declaration is
    NOT code: the Program Compiler validates it against the Capability
    Registry and dataset contract, then deterministically renders the
    executable script from a frozen template. PACT freezes the resulting
    template_hash + invocation_hash + code_hash so a child can be replayed
    bit-for-bit.
    """
    method_id: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    preprocessing: List[str] = field(default_factory=list)
    validation: str = "stratified_kfold"   # stratified_kfold | single_holdout
    hypothesis: str = ""
    resource_request: Dict[str, Any] = field(default_factory=dict)
    invocation_hash: str = ""

    def compute_hash(self) -> str:
        """Canonical hash over the RESEARCH CHOICE only (method + params).

        hypothesis/created metadata are excluded: two invocations with the
        same method/params/preprocessing/validation are the same experiment
        regardless of prose.
        """
        payload = {
            "schema": "method_invocation_v1",
            "method_id": self.method_id,
            "params": self.params,
            "preprocessing": sorted(list(self.preprocessing or [])),
            "validation": self.validation,
        }
        return canonical_hash(payload)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MethodInvocationV1":
        return cls(**_filter_kwargs(d, cls))


@dataclass
class TrialSpec:
    """PACT output: a frozen plan + code snapshot with a code hash."""
    spec_id: str = ""
    competition: str = "unknown"
    round_num: int = 1
    plan: Dict[str, Any] = field(default_factory=dict)
    code: str = ""                    # frozen candidate code
    code_hash: str = ""
    proposal_id: str = ""
    grant_id: str = ""
    created_at: str = ""
    # v2.3 template-compiled provenance (empty for legacy LLM-written code)
    invocation: Dict[str, Any] = field(default_factory=dict)
    invocation_hash: str = ""
    template_hash: str = ""

    @classmethod
    def seal(cls, competition: str, plan: ResearchPlan, code: str,
             proposal_id: str = "", grant_id: str = "",
             invocation: Optional[dict] = None,
             template_hash: str = "") -> "TrialSpec":
        code_hash = "sha256:" + hashlib.sha256(code.encode("utf-8")).hexdigest()
        inv_hash = ""
        inv_dict = dict(invocation or {})
        if inv_dict:
            try:
                inv = MethodInvocationV1.from_dict(inv_dict)
                inv_hash = inv.compute_hash()
            except Exception:  # noqa: BLE001 - hash is best-effort metadata
                inv_hash = ""
        return cls(
            spec_id=new_id("spec"),
            competition=competition,
            round_num=plan.round_num,
            plan=plan.to_dict(),
            code=code,
            code_hash=code_hash,
            proposal_id=proposal_id,
            grant_id=grant_id,
            created_at=now_iso(),
            invocation=inv_dict,
            invocation_hash=inv_dict.get("invocation_hash") or inv_hash,
            template_hash=template_hash or "",
        )

    def plan_obj(self) -> ResearchPlan:
        return ResearchPlan.from_dict(self.plan)

    def seal_record(self) -> dict:
        """Immutable seal record: pins code_hash + grant/proposal binding.

        Written host-side before execution and re-verified by the executor,
        so a TrialSpec can never be executed under a hash it was not sealed
        with (frozen-skill-path immutability).
        """
        return {
            "schema_version": "pact_seal_v1",
            "spec_id": self.spec_id,
            "competition": self.competition,
            "round_num": self.round_num,
            "code_hash": self.code_hash,
            "proposal_id": self.proposal_id,
            "grant_id": self.grant_id,
            "created_at": self.created_at,
            "invocation_hash": self.invocation_hash or "",
            "template_hash": self.template_hash or "",
            "immutable": True,
        }

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrialSpec":
        return cls(**_filter_kwargs(d, cls))


@dataclass
class CandidateBundle:
    """Host materialized artifact set for one executed trial.

    OOF predictions + submission + run log, all hash-pinned so the trusted
    evaluator can independently recompute the metric.
    """
    bundle_id: str = ""
    trial_id: str = ""
    code_path: str = ""
    oof_path: str = ""                # OOF predictions CSV (true,pred)
    submission_path: str = ""
    run_log_path: str = ""
    bundle_hash: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CandidateBundle":
        return cls(**_filter_kwargs(d, cls))

    def compute_hash(self) -> str:
        payload = dict(self.to_dict())
        payload.pop("bundle_hash", None)
        return canonical_hash(payload)


@dataclass
class EvaluatorReceipt:
    """TrustedEvaluator output: independently recomputed metric + evidence."""
    receipt_id: str = ""
    trial_id: str = ""
    metric: Optional[float] = None
    metric_name: str = "accuracy"
    evaluator: str = "trusted_recompute"   # trusted_recompute | log_parse
    evidence: str = ""
    artifact_hash: str = ""
    metric_direction: str = "higher_is_better"
    metric_alignment: str = "exact"
    metric_label: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EvaluatorReceipt":
        return cls(**_filter_kwargs(d, cls))


@dataclass
class TrialReceipt:
    """PACT output: verified evidence for one executed trial."""
    receipt_id: str = ""
    spec_id: str = ""
    competition: str = "unknown"
    round_num: int = 1
    returncode: int = -1
    stdout: str = ""
    stderr: str = ""
    metric: Optional[float] = None
    metric_name: str = "accuracy"
    verdict: str = "unknown"          # success | stagnant | regression | failure
    evidence: str = ""
    submission_exists: bool = False
    submission_path: str = ""
    submission_hash: str = ""
    wall_clock_seconds: float = 0.0
    code_hash: str = ""
    verified: bool = True
    evaluator_receipt: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    failure_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrialReceipt":
        return cls(**_filter_kwargs(d, cls))


@dataclass
class PromotionRecord:
    """Host-owned certified-best pointer (pact_control_host/)."""
    competition: str = "unknown"
    certified_best_trial_id: str = ""
    certified_best_metric: Optional[float] = None
    incumbent_trial_id: str = ""
    incumbent_metric: Optional[float] = None
    decision: str = "noop"            # promote | reject | noop
    reason: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PromotionRecord":
        return cls(**_filter_kwargs(d, cls))


@dataclass
class ActionOutcome:
    """Outer-loop feedback: one frozen grant's terminal summary to HERA."""
    grant_id: str = ""
    competition: str = "unknown"
    trials_completed: int = 0
    best_metric: Optional[float] = None
    best_trial_id: str = ""
    certified_best_trial_id: str = ""
    certified_best_metric: Optional[float] = None
    program_summary: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ActionOutcome":
        return cls(**_filter_kwargs(d, cls))
