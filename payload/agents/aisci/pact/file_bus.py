# -*- coding: utf-8 -*-
"""pact/file_bus.py - File-as-Bus (shared persistent, role-separated, system fact source).

This is the LONG-STANDING base (same layout as v6.41 HostSupervisorStore /
ProgramMailbox). Three zones:

  workspace/            Agent-visible: paper/data, code, submission/candidates,
                        plan.md, prioritized_tasks.md, impl_log.md, exp_log.md
  protocol/             Role-separated:
                        frozen_visible/ 璺?outbox/pending_agent/ 璺?                        outbox/claimed_host/ 璺?outbox/leases_host/ 璺?                        outbox/acknowledgements_host/ 璺?outcomes_visible/
  pact_control_host/    Host-only: state/pact/{supervisor,specs,bundles,
                        receipts,promotions,publications,pointers,frozen},
                        workspaces/pact/, submission/, terminal_records/

All writes are atomic (same-directory tmp -> fsync -> os.replace -> parent
fsync, best-effort). Claims use leases so a crashed host cannot deadlock.
"""
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class FileBusError(RuntimeError):
    """Raised when a File-as-Bus operation violates role/lease rules."""


def safe_artifact_name(value: str) -> str:
    """Filesystem-safe representation for sha256:... identity values."""
    return str(value).replace(":", "_").replace("/", "_").replace("\\", "_")


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _atomic_write_bytes(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(
        path.name + ".tmp." + str(os.getpid()) + "." +
        hashlib.sha1(os.urandom(8)).hexdigest()[:8])
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass
    os.replace(str(tmp), str(path))
    _fsync_directory(path.parent)
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _atomic_write_json(path: Path, payload: dict) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")
    return _atomic_write_bytes(path, data)


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class FileBus:
    """File-as-Bus over one run root (layout aligned with v6.41 base)."""

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root)
        # -- workspace (agent-visible) --
        self.workspace = self.root / "workspace"
        self.ws_data = self.workspace / "data"
        self.ws_paper = self.workspace / "paper"
        self.ws_code = self.workspace / "code"
        self.ws_submission = self.workspace / "submission"
        self.ws_candidates = self.workspace / "submission" / "candidates"
        self.ws_logs = self.workspace / "logs"
        # -- protocol (role-separated) --
        self.protocol = self.root / "protocol"
        self.frozen_visible = self.protocol / "frozen_visible"
        self.outcomes_visible = self.protocol / "outcomes_visible"
        self.outbox = self.protocol / "outbox"
        self.pending_agent = self.outbox / "pending_agent"
        self.claimed_host = self.outbox / "claimed_host"
        self.leases_host = self.outbox / "leases_host"
        self.acknowledgements_host = self.outbox / "acknowledgements_host"
        self.stale_agent = self.outbox / "stale_agent"
        self.ledger_path = self.outbox / "ledger_host.jsonl"
        # -- pact_control_host (host-only) --
        self.host_control = self.root / "pact_control_host"
        self.pact_root = self.host_control / "state" / "pact"
        self.host_supervisor = self.pact_root / "supervisor"
        self.host_frozen = self.pact_root / "frozen"
        self.host_specs = self.pact_root / "specs"
        self.host_bundles = self.pact_root / "bundles"
        self.host_receipts = self.pact_root / "receipts"
        self.host_promotions = self.pact_root / "promotions"
        self.host_publications = self.pact_root / "publications"
        self.host_pointers = self.pact_root / "pointers"
        self.host_control_state = self.pact_root / "control"
        self.host_workspaces = self.host_control / "workspaces" / "pact"
        self.host_submission = self.host_control / "submission"
        self.host_terminal = self.host_control / "terminal_records"
        self._ensure_dirs()
        self._harden()

    def _ensure_dirs(self) -> None:
        for d in (self.workspace, self.ws_data, self.ws_paper, self.ws_code,
                  self.ws_submission, self.ws_candidates, self.ws_logs,
                  self.protocol, self.frozen_visible, self.outcomes_visible,
                  self.outbox, self.pending_agent, self.claimed_host,
                  self.leases_host, self.acknowledgements_host, self.stale_agent,
                  self.host_control, self.host_supervisor, self.host_frozen,
                  self.host_specs, self.host_bundles, self.host_receipts,
                  self.host_promotions, self.host_publications,
                  self.host_pointers, self.host_control_state,
                  self.host_workspaces, self.host_submission,
                  self.host_terminal):
            d.mkdir(parents=True, exist_ok=True)

    # ---- host control plane isolation (best-effort OS-level) ----
    def _harden(self) -> None:
        """Restrict host-only zones to the owning user where the OS allows.

        On POSIX: pact_control_host/ and the host protocol outbox dirs are
        chmod 0700 (owner-only), so a same-host, different-account process
        cannot read/write the control plane through the filesystem. On
        Windows these chmod calls are best-effort no-ops; role separation is
        still enforced by the API boundary (FileBus methods) and physical
        mounts. Never raises.
        """
        self._harden_ok = False
        for d in (self.host_control, self.pact_root, self.host_supervisor,
                  self.host_frozen, self.host_specs, self.host_bundles,
                  self.host_receipts, self.host_promotions,
                  self.host_publications, self.host_pointers,
                  self.host_control_state, self.host_workspaces,
                  self.host_submission, self.host_terminal,
                  self.claimed_host, self.leases_host,
                  self.acknowledgements_host):
            try:
                os.chmod(str(d), 0o700)
                self._harden_ok = True
            except OSError:
                pass

    @staticmethod
    def _chmod_private(path: Path) -> None:
        """Best-effort 0600 on host-only payload files."""
        try:
            os.chmod(str(path), 0o600)
        except OSError:
            pass

    # ---- workspace (agent-visible) ----
    def write_workspace_md(self, rel: str, text: str) -> Path:
        path = self.workspace / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def write_plan_md(self, text: str) -> Path:
        return self.write_workspace_md("plan.md", text)

    def write_prioritized_tasks_md(self, text: str) -> Path:
        return self.write_workspace_md("prioritized_tasks.md", text)

    def write_impl_log(self, text: str) -> Path:
        return self.write_workspace_md("impl_log.md", text)

    def write_exp_log(self, text: str) -> Path:
        return self.write_workspace_md("exp_log.md", text)

    # ---- frozen grant (protocol/frozen_visible) ----
    def freeze_grant(self, grant: dict, ready: dict) -> Path:
        safe = safe_artifact_name(grant["grant_id"])
        grant_path = self.frozen_visible / ("grant_" + safe + ".json")
        ready_path = self.frozen_visible / ("grant_" + safe + ".ready")
        _atomic_write_json(grant_path, grant)
        _atomic_write_json(ready_path, ready)
        return grant_path

    def read_frozen_grant(self, grant_id: str) -> Optional[dict]:
        safe = safe_artifact_name(grant_id)
        return _read_json(self.frozen_visible / ("grant_" + safe + ".json"))

    def list_frozen(self) -> List[dict]:
        out = []
        for p in sorted(self.frozen_visible.glob("grant_*.json")):
            d = _read_json(p)
            if d:
                out.append(d)
        return out

    # ---- agent proposals (protocol/outbox/pending_agent) ----
    def propose(self, proposal: dict) -> Path:
        safe = safe_artifact_name(proposal["proposal_id"])
        path = self.pending_agent / ("proposal_" + safe + ".json")
        _atomic_write_json(path, proposal)
        return path

    def list_pending(self) -> List[dict]:
        out = []
        for p in sorted(self.pending_agent.glob("proposal_*.json")):
            d = _read_json(p)
            if d:
                out.append(d)
        return out

    # ---- stale quarantine (protocol/outbox/stale_agent) ----
    def quarantine_pending(self, proposal_id: str,
                           reason: str = "stale") -> Optional[Path]:
        """Move an orphaned/mismatched proposal out of pending_agent.

        Called when a proposal can never be served (e.g. it belongs to a
        grant whose daemon already crashed and the director has moved on).
        Moving it (not deleting) keeps the forensic trail while healing the
        bus so a fresh daemon can serve the current grant.
        """
        safe = safe_artifact_name(proposal_id)
        src = self.pending_agent / ("proposal_" + safe + ".json")
        dst = self.stale_agent / ("proposal_" + safe + ".json")
        if not src.is_file():
            return None
        try:
            os.replace(str(src), str(dst))
        except OSError:
            return None
        _atomic_write_json(dst.with_name(dst.stem + ".meta.json"), {
            "schema_version": "pact_bus_stale_v1",
            "proposal_id": proposal_id,
            "reason": reason or "stale",
            "quarantined_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                            time.gmtime()),
        })
        return dst

    # ---- host claim + lease (protocol/outbox/claimed_host + leases_host) ----
    def claim(self, proposal_id: str, host_id: str = "host",
              ttl_seconds: int = 3600) -> bool:
        safe = safe_artifact_name(proposal_id)
        src = self.pending_agent / ("proposal_" + safe + ".json")
        dst = self.claimed_host / ("proposal_" + safe + ".json")
        if not src.is_file():
            return False
        try:
            os.replace(str(src), str(dst))
        except OSError:
            return False
        lease = {
            "schema_version": "pact_bus_lease_v1",
            "proposal_id": proposal_id,
            "host_id": host_id,
            "claimed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "ttl_seconds": int(ttl_seconds),
        }
        _atomic_write_json(self.leases_host / ("lease_" + safe + ".json"), lease)
        return True

    def release_lease(self, proposal_id: str) -> None:
        safe = safe_artifact_name(proposal_id)
        lease = self.leases_host / ("lease_" + safe + ".json")
        if lease.is_file():
            lease.unlink(missing_ok=True)

    def ack(self, proposal_id: str, host_id: str = "host") -> Path:
        safe = safe_artifact_name(proposal_id)
        ack = {
            "schema_version": "pact_bus_ack_v1",
            "proposal_id": proposal_id,
            "host_id": host_id,
            "acknowledged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        return _atomic_write_json(
            self.acknowledgements_host / ("ack_" + safe + ".json"), ack)

    # ---- outcomes (protocol/outcomes_visible) ----
    def write_outcome(self, proposal_id: str, outcome: dict) -> Path:
        safe = safe_artifact_name(proposal_id)
        return _atomic_write_json(
            self.outcomes_visible / ("outcome_" + safe + ".json"), outcome)

    def list_outcomes(self) -> List[dict]:
        out = []
        for p in sorted(self.outcomes_visible.glob("outcome_*.json")):
            d = _read_json(p)
            if d:
                out.append(d)
        return out

    # ---- host-only stores (pact_control_host/state/pact) ----
    def save_seal(self, spec_id: str, seal: dict) -> Path:
        safe = safe_artifact_name(spec_id)
        path = _atomic_write_json(
            self.host_specs / ("seal_" + safe + ".json"), seal)
        self._chmod_private(path)
        return path

    def load_seal(self, spec_id: str) -> Optional[dict]:
        safe = safe_artifact_name(spec_id)
        return _read_json(self.host_specs / ("seal_" + safe + ".json"))

    def save_bundle(self, bundle: dict) -> Path:
        safe = safe_artifact_name(bundle["bundle_id"])
        path = _atomic_write_json(
            self.host_bundles / ("bundle_" + safe + ".json"), bundle)
        self._chmod_private(path)
        return path

    def load_bundle(self, bundle_id: str) -> Optional[dict]:
        safe = safe_artifact_name(bundle_id)
        return _read_json(self.host_bundles / ("bundle_" + safe + ".json"))

    def save_receipt(self, receipt: dict) -> Path:
        safe = safe_artifact_name(receipt["receipt_id"])
        path = _atomic_write_json(
            self.host_receipts / ("receipt_" + safe + ".json"), receipt)
        self._chmod_private(path)
        return path

    def save_promotion(self, record: dict) -> Path:
        path = _atomic_write_json(
            self.host_promotions / "promotion_record.json", record)
        self._chmod_private(path)
        return path

    def load_promotion(self) -> Optional[dict]:
        return _read_json(self.host_promotions / "promotion_record.json")

    def save_pointer(self, name: str, payload: dict) -> Path:
        return _atomic_write_json(
            self.host_pointers / (safe_artifact_name(name) + ".json"), payload)

    def append_terminal(self, record: dict) -> Path:
        path = self.host_terminal / "terminal_records.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True, ensure_ascii=False,
                                default=str) + "\n")
            fh.flush()
        return path

    def tree_report(self) -> Dict[str, int]:
        """Count files per top-level zone (used by tests/audits)."""
        report = {}
        for zone in ("workspace", "protocol", "pact_control_host"):
            zroot = self.root / zone
            report[zone] = len([p for p in zroot.rglob("*") if p.is_file()])
        report["control_plane_hardened"] = bool(
            getattr(self, "_harden_ok", False))
        return report
