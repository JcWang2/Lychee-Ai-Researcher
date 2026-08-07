# -*- coding: utf-8 -*-
"""PACT subsystem: implementation + execution + verification (L1 transactional).

V2.1 adds the File-as-Bus L1 transactional loop:
  ProgramAgentClient (proposes) -> HostSupervisorService (claims/validates/
  executes/evaluates/promotes) -> ControlledPublisher (certified-best only).
Execution and verification stay 100% deterministic; LLM-written code is only a
candidate until it passes PACT verification.
"""
from pact.agent_client import ProgramAgentClient
from pact.candidate import CandidateBundler
from pact.evaluator import TrustedEvaluator
from pact.executor import Executor, ExecOutcome
from pact.file_bus import FileBus, FileBusError
from pact.guards import BudgetGuard, GuardError, assert_legacy_l1_mode
from pact.host_supervisor import HostSupervisorService
from pact.implementer import Implementer
from pact.ledger import PactLedger
from pact.promotion import PromotionManager
from pact.publisher import ControlledPublisher, PublishError
from pact.quality_gate import CodeQualityGate, check_code
from pact.receipt import dump_receipt, load_receipt
from pact.verifier import Verifier
from metrics_registry import get_metric_spec, infer_metric_spec

__all__ = [
    "ProgramAgentClient", "CandidateBundler", "TrustedEvaluator",
    "Executor", "ExecOutcome", "FileBus", "FileBusError",
    "BudgetGuard", "GuardError", "assert_legacy_l1_mode",
    "HostSupervisorService", "Implementer", "PactLedger",
    "PromotionManager", "ControlledPublisher", "PublishError",
    "CodeQualityGate", "check_code",
    "dump_receipt", "load_receipt", "Verifier",
    "get_metric_spec", "infer_metric_spec",
]
