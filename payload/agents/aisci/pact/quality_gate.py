# -*- coding: utf-8 -*-
"""pact/quality_gate.py - Deterministic pre-execution code quality gate.

PACT never trusts LLM code. Before spending a container run on a candidate,
a cheap static gate rejects code that cannot work or must not run:

  - the code must parse (ast) and be reasonably sized
  - no pip install / subprocess / os.system / network calls (offline container)
  - no reading of the gold/private labelled test CSV (label leakage)

Rejection returns rc=-3 with a clear reason that lands in the trial receipt,
so failures are diagnosable without burning a trial.
"""
import ast
import re

MAX_CODE_BYTES = 200_000

# GPU mode (V2_CPU_ONLY != 1): code that hardcodes cpu wastes the trial.
# map_location="cpu" is intentionally NOT rejected (legal weight loading).
# The idiomatic `"cuda" if torch.cuda.is_available() else "cpu"` dispatch is
# intentionally ALLOWED: it uses CUDA whenever present and only falls back to
# CPU when the machine has no GPU - the compiled image harness relies on it.
GPU_MANDATORY_PATTERNS = [
    (r"torch\.device\(['\"]cpu['\"]\)",
     "hardcoded torch.device('cpu') (CUDA available; use cuda)"),
    (r"device\s*=\s*['\"]cpu['\"]",
     "hardcoded device='cpu' (CUDA available; use cuda)"),
    (r"\.to\(['\"]cpu['\"]\)",
     ".to('cpu') (CUDA available; use cuda)"),
]

FORBIDDEN_PATTERNS = [
    (r"\bpip\s+install\b", "pip install (container is offline)"),
    (r"\bsubprocess\b", "subprocess usage"),
    (r"\bos\.system\b", "os.system shell escape"),
    (r"\brequests\.", "network request (requests)"),
    (r"\burllib\b", "network request (urllib)"),
    (r"\bsocket\b", "socket/network usage"),
    (r"https?://", "network URL"),
    (r"\bgetpass\b", "interactive password prompt"),
]


class CodeQualityGate:
    """Static checks run before a candidate script is executed."""

    def check(self, code: str, gold_path: str = "",
              test_path: str = "", gpu_mandatory: bool = False) -> tuple:
        """Return (ok: bool, reason: str)."""
        if not code or not code.strip():
            return False, "empty candidate code"
        size = len(code.encode("utf-8"))
        if size > MAX_CODE_BYTES:
            return False, "code too large (%d bytes > %d)" % (size, MAX_CODE_BYTES)
        try:
            ast.parse(code)
        except SyntaxError as exc:
            return False, "syntax error: %s" % exc
        lowered = code.lower()
        for pattern, label in FORBIDDEN_PATTERNS:
            if re.search(pattern, lowered):
                return False, "forbidden: %s" % label
        if gpu_mandatory:
            for pattern, label in GPU_MANDATORY_PATTERNS:
                if re.search(pattern, code):
                    return False, "forbidden: %s" % label
        if gold_path and gold_path != test_path:
            if gold_path in code:
                return False, "forbidden: reads gold/private labelled test CSV"
            private_hint = str(gold_path).rsplit("private", 1)[0] + "private"
            if private_hint in code:
                return False, "forbidden: references private/ directory"
        return True, ""


# Convenience single-call API used by the host supervisor.
def check_code(code: str, gold_path: str = "", test_path: str = "",
               gpu_mandatory: bool = False) -> tuple:
    return CodeQualityGate().check(code, gold_path, test_path,
                                   gpu_mandatory=gpu_mandatory)