# -*- coding: utf-8 -*-
"""pact/candidate.py - CandidateBundle: OOF + submission + run log packaging.

The bundle is the artifact set the trusted evaluator recomputes from.
Everything is hash-pinned so verification is independent of LLM claims.
Trial artifacts are staged under workspace/submission/candidates/<trial>/,
then copied into the host-only bundle store.
"""
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, Optional

from v2_contracts import CandidateBundle, new_id, now_iso


class CandidateBundler:
    """Packages executed-trial artifacts into one verifiable bundle."""

    def __init__(self, bus, work_dir):
        self.bus = bus
        self.work_dir = Path(work_dir)

    def _copy_artifact(self, trial_id: str, rel_name: str,
                       bundle_dir: Path) -> str:
        """Copy one artifact from the trial stage dir into the bundle dir."""
        src = self.bus.ws_candidates / trial_id / rel_name
        if not src.is_file():
            return ""
        dst = bundle_dir / rel_name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return str(dst)

    def build(self, trial_id: str, proposal_id: str) -> Optional[CandidateBundle]:
        bundle_dir = self.bus.host_bundles / ("bundle_" + trial_id)
        bundle_dir.mkdir(parents=True, exist_ok=True)

        oof_path = self._copy_artifact(trial_id, "oof.csv", bundle_dir)
        submission_path = self._copy_artifact(trial_id, "submission.csv",
                                              bundle_dir)
        run_log_path = ""
        logs = self.bus.ws_logs / ("run_" + trial_id + ".log")
        if logs.is_file():
            dst = bundle_dir / ("run_" + trial_id + ".log")
            shutil.copy2(logs, dst)
            run_log_path = str(dst)

        if not submission_path and not oof_path:
            return None  # nothing verifiable -> no bundle

        bundle = CandidateBundle(
            bundle_id=new_id("bundle"),
            trial_id=trial_id,
            code_path="",
            oof_path=oof_path,
            submission_path=submission_path,
            run_log_path=run_log_path,
        )
        bundle.bundle_hash = bundle.compute_hash()
        self.bus.save_bundle(bundle.to_dict())
        return bundle

    def artifact_hashes(self, bundle: CandidateBundle) -> Dict[str, str]:
        out = {}
        for field_name in ("oof_path", "submission_path", "run_log_path"):
            p = getattr(bundle, field_name)
            if p and Path(p).is_file():
                digest = hashlib.sha256(Path(p).read_bytes()).hexdigest()
                out[field_name] = "sha256:" + digest
        return out
