# -*- coding: utf-8 -*-
"""pact/guards.py - Budget, timeout and mode guards (fail-closed)."""
import json
import os
import time
from pathlib import Path
from typing import Optional


class GuardError(RuntimeError):
    """Raised when a PACT guard refuses to continue (fail-closed)."""


def assert_legacy_l1_mode() -> None:
    """Fail-closed: the serial release must never run the self-evolution lifecycle."""
    if os.environ.get("PACT_STAGE4_SELF_EVOLUTION", "0") not in ("0", ""):
        raise GuardError(
            "PACT_STAGE4_SELF_EVOLUTION must be 0 for the serial HERA+PACT V2 loop")


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    os.replace(str(tmp), str(path))


def _read_json(path: Path) -> Optional[dict]:
    try:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        pass
    return None


class BudgetGuard:
    """Tracks the V2.2 three-limit budget model (persisted + transactional).

    Three INDEPENDENT limits, first one hit wins:
      MAX_GRANTS          - outer research decision opportunities (1 grant =
                            one HERA decision + its children)
      MAX_TOTAL_TRIALS    - total child trials committed across all grants
      total_budget (wall clock) - TOTAL_WALL_CLOCK, the highest authority

    Persistence (state_dir, optional but REQUIRED for 24h runs):
      budget RECEIPTS (budget_receipts/receipt_<grant_id>.json) are the
      SINGLE SOURCE OF TRUTH for grant/trial accounting. budget_state.json
      is a DERIVED cache: it holds run_started_at / wall_deadline_epoch
      (restored verbatim so a restart can never re-earn spent grants/trials
      or reset the 24h clock) and counters that are REBUILT from receipts on
      every load/commit/recovery. A crash between receipt write and state
      write can therefore never lose a committed grant.

    Crash-consistent grant flow (host-owned atomic commit):
      check_research_opportunity()   validate against all three limits
      begin_reservation(children, grant_id)  persist a PENDING reservation
      bus.freeze_grant(...)          freeze the grant on the File-as-Bus
      commit_grant(children, grant_id)       atomically create the
                                             authoritative receipt, then
                                             rebuild counters from receipts

    Recovery: recover_pending(frozen_grants) reconciles three facts:
      committed receipt exists          -> always counted (idempotent)
      frozen grant + pending + no receipt -> recovered receipt is written
                                             (children taken from the frozen
                                             grant's trial_budget), counted
      pending + not frozen + no receipt -> reservation discarded
    commit_grant() is idempotent: a receipt that already exists for a
    grant_id is never double-counted.
    """

    def __init__(self, total_budget_seconds: int = 86400,
                 round_timeout_seconds: int = 3600,
                 max_grants: int = 128,
                 max_total_trials: int = 256,
                 state_dir=None):
        self.total_budget = max(1, int(total_budget_seconds))
        self.round_timeout = max(1, int(round_timeout_seconds))
        self.max_grants = max(1, int(max_grants))
        self.max_total_trials = max(1, int(max_total_trials))
        self.state_dir = Path(state_dir) if state_dir else None
        self.grants_used = 0
        self.trials_used = 0
        self.committed_grant_ids = []
        self._run_started_at = time.time()
        self._wall_deadline_epoch = self._run_started_at + self.total_budget
        if self.state_dir is not None:
            self._load_state()

    # ---- persistence ----
    def _state_path(self) -> Path:
        return self.state_dir / "budget_state.json"

    def _reservation_dir(self) -> Path:
        return self.state_dir / "budget_reservations"

    def _receipt_dir(self) -> Path:
        return self.state_dir / "budget_receipts"

    def _load_state(self) -> None:
        """Restore the ORIGINAL wall deadline from budget_state.json and
        REBUILD grant/trial counters from the authoritative receipts
        (budget_state.json is only a derived cache after v2.2.1-rc3)."""
        state = _read_json(self._state_path())
        if state:
            try:
                started = float(state.get("run_started_at") or 0)
                deadline = float(state.get("wall_deadline_epoch") or 0)
                if started > 0:
                    self._run_started_at = started
                if deadline > 0:
                    self._wall_deadline_epoch = deadline
                else:
                    self._wall_deadline_epoch = self._run_started_at + self.total_budget
            except (TypeError, ValueError):
                pass
        self._rebuild_from_receipts()
        self._persist()

    def _load_receipts(self) -> list:
        """All committed receipts on disk (the authoritative accounting fact)."""
        if self.state_dir is None:
            return []
        out = []
        rdir = self._receipt_dir()
        if rdir.is_dir():
            for path in sorted(rdir.glob("receipt_*.json")):
                r = _read_json(path)
                if r and r.get("grant_id"):
                    out.append(r)
        return out

    def _rebuild_from_receipts(self) -> None:
        """Derive grants_used / trials_used / committed_grant_ids from the
        receipt set. Idempotent by construction: one receipt per grant."""
        receipts = self._load_receipts()
        self.committed_grant_ids = sorted(
            {str(r.get("grant_id")) for r in receipts})
        self.grants_used = len(self.committed_grant_ids)
        total = 0
        for r in receipts:
            try:
                total += max(1, int(r.get("children") or 1))
            except (TypeError, ValueError):
                total += 1
        self.trials_used = total

    def _persist(self) -> None:
        if self.state_dir is None:
            return
        try:
            _atomic_write_json(self._state_path(), {
                "run_started_at": round(self._run_started_at, 3),
                "wall_deadline_epoch": round(self._wall_deadline_epoch, 3),
                "grants_committed": self.grants_used,
                "max_grants": self.max_grants,
                "trials_reserved": self.trials_used,
                "max_total_trials": self.max_total_trials,
                "committed_grant_ids": list(self.committed_grant_ids),
            })
        except OSError as e:
            raise GuardError("budget state not writable: %s" % e)

    # ---- wall clock / limits ----
    def elapsed(self) -> float:
        return max(0.0, time.time() - self._run_started_at)

    def remaining(self) -> float:
        return max(0.0, self._wall_deadline_epoch - time.time())

    def budget_exhausted(self) -> bool:
        return self.remaining() <= 0

    def grants_remaining(self) -> int:
        return max(0, self.max_grants - self.grants_used)

    def trials_remaining(self) -> int:
        return max(0, self.max_total_trials - self.trials_used)

    def check_budget(self) -> None:
        if self.budget_exhausted():
            raise GuardError("wall-clock budget exhausted")

    def check_research_opportunity(self, children: int,
                                   est_cost_seconds: Optional[float] = None) -> None:
        """Fail-closed check before freezing one grant (three limits)."""
        children = max(1, int(children))
        if self.budget_exhausted():
            raise GuardError("wall-clock budget exhausted")
        if self.grants_used + 1 > self.max_grants:
            raise GuardError(
                "grant budget exhausted (%d/%d)" % (self.grants_used,
                                                    self.max_grants))
        if self.trials_used + children > self.max_total_trials:
            raise GuardError(
                "trial budget exhausted (%d+%d > %d)"
                % (self.trials_used, children, self.max_total_trials))
        if est_cost_seconds is not None and self.remaining() < est_cost_seconds:
            raise GuardError(
                "remaining wall clock (%.0fs) < estimated grant cost (%.0fs)"
                % (self.remaining(), est_cost_seconds))

    def begin_reservation(self, children: int, grant_id: str = "") -> str:
        """Persist a PENDING budget reservation BEFORE the grant is frozen.

        Returns the grant_id (or a generated one). The budget is NOT consumed
        yet; commit_grant() moves the reservation to an authoritative receipt
        after bus.freeze_grant() succeeds.
        """
        children = max(1, int(children))
        self.check_research_opportunity(children)
        grant_id = str(grant_id or "").strip() or (
            "grant_pending_" + str(int(time.time() * 1000)))
        if self.state_dir is not None:
            try:
                _atomic_write_json(
                    self._reservation_dir() / ("res_" + grant_id + ".json"),
                    {
                        "grant_id": grant_id,
                        "children": children,
                        "created_at_epoch": round(time.time(), 3),
                        "status": "pending",
                    })
            except OSError as e:
                raise GuardError("budget reservation not writable: %s" % e)
        return grant_id

    def cancel_reservation(self, grant_id: str = "") -> None:
        """Roll back a pending reservation when freezing fails on the bus."""
        grant_id = str(grant_id or "").strip()
        if not grant_id or self.state_dir is None:
            return
        try:
            (self._reservation_dir() / ("res_" + grant_id + ".json")).unlink()
        except OSError:
            pass

    def commit_grant(self, children: int, grant_id: str = "") -> None:
        """Commit one grant: ATOMICALLY create the authoritative receipt,
        then rebuild counters from the full receipt set (idempotent).

        Called AFTER the grant is frozen on the File-as-Bus. If a receipt
        already exists for grant_id this is a no-op - a crash between the
        receipt write and the budget_state write can never lose or
        double-count a committed grant.
        """
        children = max(1, int(children))
        grant_id = str(grant_id or "").strip() or (
            "grant_" + str(len(self.committed_grant_ids) + 1))
        if grant_id in self.committed_grant_ids:
            return  # idempotent: already committed
        if self.state_dir is not None:
            receipt_path = self._receipt_dir() / ("receipt_" + grant_id + ".json")
            if receipt_path.is_file():
                self._rebuild_from_receipts()
                return
            try:
                _atomic_write_json(receipt_path, {
                    "grant_id": grant_id,
                    "children": children,
                    "committed_at_epoch": round(time.time(), 3),
                    "status": "committed",
                })
            except OSError as e:
                raise GuardError("budget receipt not writable: %s" % e)
            # receipt is authoritative -> rebuild derived counters
            self._rebuild_from_receipts()
            try:
                res_path = self._reservation_dir() / ("res_" + grant_id + ".json")
                if res_path.is_file():
                    res_path.unlink()
            except OSError:
                pass
            self._persist()
        else:
            self.grants_used += 1
            self.trials_used += children
            if grant_id not in self.committed_grant_ids:
                self.committed_grant_ids.append(grant_id)

    def recover_pending(self, frozen_grants=None,
                          frozen_grant_ids=None) -> dict:
        """Crash recovery over the three facts: committed receipts, frozen
        grants and pending reservations.

        frozen_grants: list of grant dicts from bus.list_frozen() (each
        carries grant_id and trial_budget); a legacy list of plain ids is
        also accepted (frozen_grant_ids=... keyword kept for back-compat). Rules:
          - receipt already exists          -> always counted, reservation
                                               dropped (idempotent)
          - frozen grant + pending + no receipt -> write a RECOVERED receipt
                                               using the frozen grant's
                                               trial_budget, then count it
          - pending + not frozen + no receipt -> reservation discarded
          - frozen grant with no reservation and no receipt -> NOT counted
            (its budget was never reserved)
        Counters are rebuilt from the authoritative receipt set afterwards.
        """
        if self.state_dir is None:
            return {"recovered": [], "discarded": []}
        if frozen_grants is None:
            frozen_grants = frozen_grant_ids
        frozen_map = {}
        for g in (frozen_grants or []):
            if isinstance(g, dict):
                gid = str(g.get("grant_id") or "").strip()
                if gid:
                    frozen_map[gid] = g
            else:
                gid = str(g or "").strip()
                if gid:
                    frozen_map[gid] = {"grant_id": gid}
        frozen = set(frozen_map)
        recovered, discarded = [], []
        res_dir = self._reservation_dir()
        if res_dir.is_dir():
            for path in sorted(res_dir.glob("res_*.json")):
                res = _read_json(path)
                if not res:
                    continue
                gid = str(res.get("grant_id") or "")
                if gid in self.committed_grant_ids:
                    # receipt already exists: reservation is stale
                    try:
                        path.unlink()
                    except OSError:
                        pass
                    continue
                if gid in frozen:
                    fg = frozen_map[gid]
                    try:
                        children = max(1, int(
                            fg.get("trial_budget") or res.get("children") or 1))
                    except (TypeError, ValueError):
                        try:
                            children = max(1, int(res.get("children") or 1))
                        except (TypeError, ValueError):
                            children = 1
                    try:
                        _atomic_write_json(
                            self._receipt_dir() / ("receipt_" + gid + ".json"),
                            {
                                "grant_id": gid,
                                "children": children,
                                "committed_at_epoch": round(time.time(), 3),
                                "status": "committed",
                                "recovered": True,
                            })
                    except OSError as e:
                        raise GuardError("budget recovery receipt failed: %s" % e)
                    recovered.append(gid)
                else:
                    discarded.append(gid)
                try:
                    path.unlink()
                except OSError:
                    pass
        self._rebuild_from_receipts()
        self._persist()
        return {"recovered": recovered, "discarded": discarded}

    def status(self) -> dict:
        return {
            "grants_used": self.grants_used,
            "max_grants": self.max_grants,
            "trials_used": self.trials_used,
            "max_total_trials": self.max_total_trials,
            "wall_remaining": round(self.remaining(), 1),
            "wall_total": self.total_budget,
            "run_started_at": round(self._run_started_at, 1),
            "wall_deadline_epoch": round(self._wall_deadline_epoch, 1),
            "committed_grant_ids": list(self.committed_grant_ids),
        }

    def clamp_round_timeout(self, requested: int) -> int:
        """Effective timeout for one round: min(requested, round_timeout, remaining)."""
        remaining = max(1, int(self.remaining()))
        requested = int(requested) if requested and requested > 0 else self.round_timeout
        return max(1, min(requested, self.round_timeout, remaining))
