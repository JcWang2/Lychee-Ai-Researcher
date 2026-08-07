# -*- coding: utf-8 -*-
"""pact/host_supervisor.py - HostSupervisorService: unique trust root.

Design contract (inner PACT L1 transactional loop):
  - starts before the agent, polls protocol/pending_agent/
  - atomically claims proposals (claimed_host/ + lease)
  - validates scope / grant binding / mutation axis / budget
  - materializes TrialSpec into an isolated workspace, executes it
  - packages CandidateBundle, recomputes metric with TrustedEvaluator
  - writes EvaluatorReceipt + TrialReceipt + ledger
  - promotes/rejects against the certified incumbent (PromotionManager)
  - writes outcomes_visible/ for the agent, acknowledges the proposal

Host never trusts LLM claims: metric comes only from the evaluator.
"""
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from v2_contracts import (
    ActionOutcome, CandidateBundle, EvaluatorReceipt, MethodInvocationV1,
    PromotionRecord, ResearchProgramGrant, TrialReceipt, TrialSpec,
    new_id, now_iso,
)
from pact.agent_client import ProgramAgentClient
from pact.candidate import CandidateBundler
from pact.evaluator import TrustedEvaluator
from pact.executor import Executor, ExecOutcome
from pact.file_bus import FileBus, FileBusError
from pact.promotion import PromotionManager
from pact.quality_gate import check_code
from pact.receipt import dump_receipt


def trial_timeout_seconds(plan_budget_seconds, round_timeout, guards=None) -> int:
    """Effective per-trial container timeout (seconds).

    Priority: the plan's max_budget_seconds (LLM-chosen, clamped to the
    RESOURCE PROFILE by the planner) is authoritative; the daemon round
    timeout is the ceiling; guards (when present) additionally clamp to
    the round's remaining wall clock. v2_host_daemon passes guards=None,
    so this must NOT depend on guards to honor the plan budget.
    """
    timeout = max(1, int(round_timeout or 3600))
    try:
        plan_budget = int(plan_budget_seconds or 0)
    except (TypeError, ValueError):
        plan_budget = 0
    if plan_budget > 0:
        timeout = max(1, min(plan_budget, timeout))
    if guards is not None:
        timeout = guards.clamp_round_timeout(timeout)
    return max(1, timeout)


class HostSupervisorService:
    """Polls proposals and drives one child trial to a verified outcome."""

    def __init__(self, bus: FileBus, executor: Executor, bundler: CandidateBundler,
                 evaluator: TrustedEvaluator, promotion: PromotionManager,
                 implementer, guards=None, ledger=None,
                 competition: str = "unknown", max_budget_seconds: int = 3600,
                 host_id: str = "host", data_dir=None, sample_path: str = "",
                 gold_test_csv: str = "", test_csv: str = "",
                 state_dir=None, compiler=None, registry=None,
                 metric_min_delta: float = 0.01):
        self.bus = bus
        self.executor = executor
        self.bundler = bundler
        self.evaluator = evaluator
        self.promotion = promotion
        self.implementer = implementer
        self.compiler = compiler          # v2.3 ProgramCompiler (or None)
        self.registry = registry          # v2.3 CapabilityRegistry (or None)
        self.guards = guards
        self.ledger = ledger
        self.competition = competition
        self.metric_direction = getattr(self.evaluator, "metric_direction", "higher_is_better")
        # v2.3.6: per-metric improvement threshold (closed loop passes the
        # metric-family delta; tests/legacy callers keep the 0.01 default).
        self.min_delta = float(metric_min_delta)
        self.max_budget_seconds = max(1, int(max_budget_seconds))
        self.host_id = host_id
        self.work_dir = Path(executor.work_dir)
        self.data_dir = Path(data_dir) if data_dir else self.work_dir.parent
        self.sample_path = sample_path or ""
        self.gold_test_csv = gold_test_csv or ""
        self.test_csv = test_csv or ""
        self.state_dir = Path(state_dir) if state_dir else None

    # ---- Stage 1: claim + validate ----
    def _validate_proposal(self, proposal: dict, grant: dict) -> None:
        if proposal.get("grant_id") != grant.get("grant_id"):
            raise FileBusError("proposal_grant_mismatch")
        axis = proposal.get("mutation_axis") or ""
        if axis and axis != grant.get("mutation_axis"):
            raise FileBusError("proposal_mutation_axis_mismatch:%s" % axis)
        child = int(proposal.get("child_index") or 0)
        if child < 1 or child > int(grant.get("trial_budget") or 0):
            raise FileBusError("proposal_child_out_of_budget:%s" % child)

    def _materialize_spec(self, proposal: dict, grant: dict,
                          profile, plan_obj) -> TrialSpec:
        """Freeze the TrialSpec for one child (v2.3 compiler-first).

        When the proposal carries a MethodInvocationV1, the Program
        Compiler validates it against the registry + dataset contract and
        deterministically renders the code (0 LLM calls, no codegen).
        Legacy proposals (no invocation / no compiler / compile error) fall
        back to the LLM implementer so old tests and out-of-band flows keep
        working. The sealed TrialSpec (code_hash + invocation_hash +
        template_hash + grant/proposal binding) is written to the host-only
        specs store BEFORE execution: the executor refuses to run any spec
        whose code does not match its sealed hash.
        """
        invocation = proposal.get("invocation") or {}
        code = None
        template_hash = ""
        inv_dict = {}
        if (isinstance(invocation, dict) and invocation.get("method_id")
                and self.compiler is not None):
            try:
                inv = MethodInvocationV1.from_dict(invocation)
                manifest = self._dataset_manifest(profile)
                ok, reason = self.compiler.validate(inv, profile, manifest)
                if ok:
                    code, template_hash = self.compiler.render(
                        inv, profile, manifest)
                    inv_dict = inv.to_dict()
                    print("COMPILED proposal=%s method=%s code_len=%d "
                          "template_hash=%s"
                          % (proposal.get("proposal_id"), inv.method_id,
                             len(code), template_hash[:24]), flush=True)
                else:
                    print("COMPILER_REJECT proposal=%s reason=%s"
                          % (proposal.get("proposal_id"), reason), flush=True)
            except Exception as exc:  # noqa: BLE001 - legacy fallback
                print("COMPILER_ERROR proposal=%s err=%r; legacy implementer"
                      % (proposal.get("proposal_id"), exc), flush=True)
        if code is None:
            implement_kwargs = {}
            if self.sample_path:
                implement_kwargs["sample_path"] = self.sample_path
            implement_kwargs["branch"] = str(grant.get("selected_branch_id") or "")
            implement_kwargs["proposal"] = proposal
            ref = self._load_incumbent_ref()
            if ref and ref.get("code_path"):
                try:
                    ref_path = Path(ref["code_path"])
                    if ref_path.is_file():
                        implement_kwargs["reference_code"] = ref_path.read_text(
                            encoding="utf-8")
                        implement_kwargs["reference_meta"] = ref
                except OSError:
                    pass
            code = self.implementer.implement(
                plan_obj, profile, self.data_dir, self.work_dir,
                self.bus.ws_submission, **implement_kwargs)
        spec = TrialSpec.seal(
            competition=self.competition,
            plan=plan_obj,
            code=code,
            proposal_id=proposal.get("proposal_id", ""),
            grant_id=grant.get("grant_id", ""),
            invocation=inv_dict,
            template_hash=template_hash,
        )
        # write code + spec into the isolated workspace (agent-visible code zone)
        self._write_candidate_code(spec.spec_id, code)
        spec_path = self.bus.workspace / "code" / ("spec_" + spec.spec_id + ".json")
        spec_path.write_text(json.dumps(spec.to_dict(), ensure_ascii=False, default=str),
                             encoding="utf-8")
        # authoritative immutable seal record (host-only, hash-pinned)
        self.bus.save_seal(spec.spec_id, spec.seal_record())
        return spec

    def _dataset_manifest(self, profile) -> dict:
        """Dataset contract for the compiler: the director's
        work_dir/data_manifest.json when present, else a minimal contract
        derived from the analysis profile."""
        try:
            mp = self.work_dir / "data_manifest.json"
            if mp.is_file():
                data = json.loads(mp.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
        except (OSError, ValueError):
            pass
        out = {}
        if profile is not None:
            out["task_type"] = getattr(profile, "task_type", "") or ""
            out["metric_name"] = getattr(profile, "metric_name", "") or ""
            out["train_rows"] = int(getattr(profile, "train_rows", 0) or 0)
            if getattr(profile, "image_width", 0):
                out["train_images"] = "yes"
        return out

    def _write_candidate_code(self, spec_id: str, code: str) -> None:
        """Write candidate code into the agent-visible code zone.

        Repairs overwrite the original materialization so the archived code
        matches what actually ran in the container (the seal record still
        pins the original hash for execution immutability).
        """
        try:
            path = self.bus.ws_code / ("trial_" + str(spec_id) + ".py")
            path.write_text(code, encoding="utf-8")
        except OSError:
            pass

    def _load_incumbent_ref(self) -> Optional[dict]:
        """Round-continuity asset: the verified best code of a previous
        round/grant, persisted by the director under state_dir.

        The incumbent is an ASSET for the LLM to use or replace - never a
        platform-mandated method. Returns None when continuity is off (no
        state_dir) or no incumbent exists yet.
        """
        if not self.state_dir:
            return None
        path = self.state_dir / "incumbent_best.json"
        try:
            if path.is_file():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("code_path"):
                    return data
        except (OSError, ValueError):
            pass
        return None

    # ---- Stage 2: execute + evaluate ----
    @staticmethod
    def _outcome_tail(outcome: ExecOutcome) -> str:
        """Last stderr/stdout lines of a failed run (repair input)."""
        lines = [line.strip() for line in
                 ((outcome.stderr or "") + "\n" + (outcome.stdout or ""))
                 .strip().splitlines() if line.strip()]
        return "\n".join(lines[-15:])[:2000]

    def _execute(self, spec: TrialSpec, plan_obj=None, profile=None,
                 max_repairs: int = 2) -> ExecOutcome:
        """Execute with repair loop (v2.3 template-aware).

        Compiled specs (invocation_hash set) are repaired by DETERMINISTIC
        parameter micro-patches (compiler.patch_params: halve a budget/
        convergence knob) and re-rendered - 0 LLM calls. Legacy LLM-written
        code keeps the Arbor-style implementer.repair. Timeouts are not
        repaired (rerunning would only time out again); the deterministic
        artifact fallback still guarantees a verifiable metric. A verified
        failure of an ephemeral synthesized capability marks it broken so
        later children skip it.
        """
        orig_spec_id = spec.spec_id
        ok, reason = check_code(spec.code, self.gold_test_csv, self.test_csv,
                                gpu_mandatory=os.environ.get("V2_CPU_ONLY") != "1")
        if not ok:
            return ExecOutcome(
                returncode=-3, stdout="", stderr="CODE_QUALITY_REJECT: " + reason,
                error=reason, wall_clock_seconds=0.0, trial_work_dir="")
        timeout = trial_timeout_seconds(
            spec.plan_obj().max_budget_seconds,
            self.max_budget_seconds, self.guards)
        outcome = self.executor.run(spec, timeout)
        template_mode = bool(spec.invocation_hash and self.compiler is not None)
        if (outcome.returncode == 0 or outcome.timed_out
                or (self.implementer is None and not template_mode)
                or plan_obj is None or int(max_repairs) <= 0):
            self._mark_ephemeral_broken(spec, outcome)
            return outcome
        error = self._outcome_tail(outcome)
        for _ in range(int(max_repairs)):
            code2 = None
            template_hash = spec.template_hash
            inv_dict = dict(spec.invocation or {})
            if template_mode:
                try:
                    patched, note = self.compiler.patch_params(inv_dict, error)
                    if patched is not None:
                        manifest = self._dataset_manifest(profile)
                        code2, template_hash = self.compiler.render(
                            patched, profile, manifest)
                        inv_dict = patched.to_dict()
                        print("COMPILE_PATCH spec=%s %s" % (spec.spec_id, note),
                              flush=True)
                    else:
                        print("COMPILE_PATCH spec=%s %s"
                              % (spec.spec_id, note), flush=True)
                except Exception as exc:  # noqa: BLE001 - give up patching
                    print("COMPILE_PATCH spec=%s err=%r"
                          % (spec.spec_id, exc), flush=True)
            elif self.implementer is not None:
                code2 = self.implementer.repair(
                    spec.code, error, plan_obj, profile)
            if not code2 or code2.strip() == spec.code.strip():
                break
            ok2, reason2 = check_code(code2, self.gold_test_csv, self.test_csv,
                                      gpu_mandatory=os.environ.get("V2_CPU_ONLY") != "1")
            if not ok2:
                break
            old_dir = self.work_dir / spec.spec_id
            if old_dir.is_dir():
                import shutil
                shutil.rmtree(old_dir, ignore_errors=True)
            spec = TrialSpec.seal(
                competition=spec.competition, plan=plan_obj, code=code2,
                proposal_id=spec.proposal_id, grant_id=spec.grant_id,
                invocation=inv_dict or None,
                template_hash=template_hash)
            self._write_candidate_code(orig_spec_id, spec.code)
            outcome = self.executor.run(spec, timeout)
            if outcome.returncode == 0 or outcome.timed_out:
                break
            error = self._outcome_tail(outcome)
        self._mark_ephemeral_broken(spec, outcome)
        return outcome

    def _mark_ephemeral_broken(self, spec: TrialSpec, outcome: ExecOutcome
                               ) -> None:
        """Phase C: a verified non-timeout failure of a synthesized
        capability marks it broken (persisted for ephemerals) so later
        children never re-request the same dead adapter."""
        if (self.compiler is None or self.registry is None
                or outcome.returncode == 0 or outcome.timed_out
                or not spec.invocation):
            return
        mid = str(spec.invocation.get("method_id") or "")
        spec_obj = self.registry.get(mid)
        if spec_obj is not None and spec_obj.ephemeral:
            self.compiler.mark_broken(
                mid, "trial failure rc=%s after repairs" % outcome.returncode)
            print("EPHEMERAL_BROKEN method=%s rc=%s"
                  % (mid, outcome.returncode), flush=True)

    def _ensure_artifacts(self, spec: TrialSpec, outcome: ExecOutcome,
                          force_overwrite: bool = False) -> str:
        """Enforce the certification artifact contract: submission.csv AND
        oof.csv must exist for a trial to carry a trusted metric.

        Every certifiable trial needs both artifacts (submission for
        publishing, OOF for the trusted evaluator's independent recompute -
        log-parse is never accepted). If the candidate crashed, timed out,
        or produced an incomplete artifact set, PACT writes the deterministic
        majority/mean baseline artifacts host-side so the trial still yields
        a verifiable, recomputable metric instead of burning the budget. The
        original failure stays in the receipt (failure_reason).
        """
        trial_work = (Path(outcome.trial_work_dir) if outcome.trial_work_dir
                      else self.work_dir / spec.spec_id)
        if not trial_work.is_dir():
            trial_work.mkdir(parents=True, exist_ok=True)
        sub = trial_work / "submission.csv"
        oof = trial_work / "oof.csv"
        if (not force_overwrite and sub.is_file() and oof.is_file()
                and sub.stat().st_size > 0 and oof.stat().st_size > 0):
            return ""
        try:
            from pact.deterministic import write_deterministic_artifacts
            from data_layout import DatasetLayoutError, resolve_dataset_layout
            layout = resolve_dataset_layout(
                self.data_dir, sample_path=self.sample_path)
            result = write_deterministic_artifacts(
                layout, trial_work,
                metric_name=self.evaluator.metric_name)
        except (DatasetLayoutError, OSError, ValueError):
            return ""
        if not (result["submission"] or result["oof"]):
            return ""
        cause = ("failed rc=%s" % outcome.returncode
                 if outcome.returncode != 0
                 else "missing oof.csv/submission.csv (artifact contract)")
        return ("PACT_FALLBACK: candidate code %s; deterministic baseline "
                "artifacts written (pred=%r rows=%d)"
                % (cause, result["pred"], result["rows"]))

    def _oof_semantics_note(self, spec: TrialSpec, outcome: ExecOutcome,
                            profile) -> str:
        """Audit OOF semantics and return an informational note (never a
        rejection reason).

        MLE-Bench only scores submission.csv on the hidden test set, so how
        the agent uses train.csv is its own choice: IN-SAMPLE OOF
        predictions over the full training set are allowed and scored. This
        audit marks such trials so receipts/feedback stay honest that the
        metric is a training-fit signal rather than a generalization
        estimate. The row-count audit compares OOF rows against the code's
        EFFECTIVE row base (full train rows, or the subsample size the code
        requests - resource profiles mandate subsampling, so honest
        val/OOF counts are smaller than full-size split math):
          - no split/CV call in code + OOF covering ~the whole base
            (>=90%): note when predictions VARY (in-sample by construction);
            no note for CONSTANT predictions (majority/mean baseline or the
            emergency artifact: constant predictors cannot memorize)
          - train_test_split -> OOF covering ~the whole base (>=90%):
            in-sample note; smaller counts (subsamples, partial val) get no
            note; a declared >=90% holdout is held-out, no note
          - fold/holdout-family calls (KFold, ShuffleSplit, GroupKFold,
            TimeSeriesSplit, LeaveOneOut, ...) -> every row is predicted by
            a fold that did not train on it: held-out, no note. Only actual
            CALLS count (imports/comments do not).
        Returns "" when the OOF looks held-out/constant, else a short note.
        """
        code = spec.code or ""
        split_call = re.search(r"train_test_split\s*\(", code)
        cv_call = re.search(
            r"\b(?:(?:Stratified|Repeated)?KFold|"
            r"(?:Stratified)?ShuffleSplit|GroupKFold|GroupShuffleSplit|"
            r"TimeSeriesSplit|LeaveOneOut|PredefinedSplit)\s*\(", code)
        trial_work = (Path(outcome.trial_work_dir) if outcome.trial_work_dir
                      else self.work_dir / spec.spec_id)
        oof = trial_work / "oof.csv"
        if not oof.is_file() or oof.stat().st_size <= 0:
            return ""
        rows = []
        try:
            with oof.open("r", encoding="utf-8", errors="replace",
                          newline="") as fh:
                rows = list(csv.reader(fh))
        except (OSError, ValueError):
            return ""
        if len(rows) <= 1:
            return ""
        actual = len(rows) - 1  # minus header row
        header = [h.strip().lower() for h in (rows[0] or [])]
        pred_idx = [i for i, h in enumerate(header)
                    if h == "pred" or h.startswith("pred_")]
        pred_constant = False
        if pred_idx:
            pred_constant = True
            for i in pred_idx:
                seen = set()
                for r in rows[1:]:
                    if i < len(r):
                        seen.add(r[i].strip())
                if len(seen) > 1:
                    pred_constant = False
                    break
        train_rows = int(getattr(profile, "train_rows", 0) or 0)
        if train_rows <= 0:
            return ""
        # Effective row base: resource profiles tell the candidate to
        # subsample. Only .sample(n=..) / .sample(..) / .sample(frac=..)
        # count, and only when large enough (>=1000): exploratory calls like
        # print(df.head(5)) or print(df.sample(5)) must never shrink the
        # base, otherwise honest OOFs get falsely rejected.
        base = train_rows
        m = re.search(r"\.sample\(\s*n\s*=\s*(\d+)", code)
        if not m:
            m = re.search(r"\.sample\(\s*(\d+)", code)
        if m:
            try:
                n = int(m.group(1))
                if n >= 1000:
                    base = min(base, n)
            except ValueError:
                pass
        else:
            m = re.search(r"\.sample\(\s*frac\s*=\s*([0-9]*\.?[0-9]+)",
                          code)
            if m:
                try:
                    frac = float(m.group(1))
                    if 0 < frac <= 1:
                        base = max(1000, int(train_rows * frac))
                except ValueError:
                    pass
        if not split_call and not cv_call:
            # No split/CV call at all: OOF covering ~full train is
            # in-sample by construction UNLESS predictions are constant
            # (majority/mean baseline cannot memorize rows).
            if actual >= int(base * 0.9):
                if pred_constant:
                    return ""
                return ("no split or CV call in code but OOF covers ~full effective train set (%d/%d rows): in-sample predictions (training-fit signal, not generalization)" % (actual, base))
            return ""
        if cv_call and not split_call:
            return ""  # fold/holdout-family OOF is leak-free by construction
        if actual >= int(base * 0.9):
            # A declared >=90% holdout legitimately covers most rows.
            m = re.search(r"test_size\s*=\s*([0-9]*\.?[0-9]+)", code)
            if m:
                try:
                    if float(m.group(1)) >= 0.9:
                        return ""
                except ValueError:
                    pass
            return ("OOF covers ~full effective train set (%d/%d rows) with a single train_test_split: in-sample predictions (training-fit signal, not generalization)" % (actual, base))
        return ""

    def _stage_artifacts(self, spec: TrialSpec, outcome: ExecOutcome,
                         force: bool = False) -> None:
        """Copy artifacts ONLY from a successful trial's isolated workspace.

        Failed or timed-out trials publish nothing: otherwise stale files
        from a shared work dir can be re-staged and produce phantom metrics.
        force=True is used for the deterministic fallback: the artifacts were
        written by PACT itself (host-side), so they are trusted.
        """
        import shutil
        if outcome.returncode != 0 and not force:
            return
        trial_work = Path(outcome.trial_work_dir) if outcome.trial_work_dir \
            else self.work_dir / spec.spec_id
        stage = self.bus.ws_candidates / spec.spec_id
        stage.mkdir(parents=True, exist_ok=True)
        for name in ("submission.csv", "oof.csv"):
            src = trial_work / name
            if src.is_file():
                shutil.copy2(src, stage / name)

    def _evaluate(self, proposal_id: str, spec: TrialSpec,
                  outcome: ExecOutcome,
                  force: bool = False) -> EvaluatorReceipt:
        self._stage_artifacts(spec, outcome, force=force)
        bundle = self.bundler.build(spec.spec_id, proposal_id)
        return self.evaluator.evaluate(
            bundle, stdout=outcome.stdout, stderr=outcome.stderr,
            returncode=outcome.returncode)

    # ---- Stage 3: receipt + promotion + outcome ----
    def _better(self, candidate: float, incumbent: Optional[float]) -> bool:
        """Direction-aware improvement test (metric_direction)."""
        if incumbent is None:
            return True
        if self.metric_direction == "lower_is_better":
            return candidate < incumbent - self.min_delta
        return candidate > incumbent + self.min_delta

    def _worse(self, candidate: float, incumbent: Optional[float]) -> bool:
        if incumbent is None:
            return False
        if self.metric_direction == "lower_is_better":
            return candidate > incumbent + self.min_delta
        return candidate < incumbent - self.min_delta

    def _verdict(self, metric: Optional[float], best: Optional[float],
                 returncode: int) -> str:
        if returncode != 0 or metric is None:
            return "failure"
        if best is None:
            return "success"
        if self._better(metric, best):
            return "success"
        if not self._worse(metric, best):
            return "stagnant"
        return "regression"

    def _make_receipt(self, spec: TrialSpec, outcome: ExecOutcome,
                      eval_receipt: EvaluatorReceipt, bundle: Optional[CandidateBundle],
                      best_metric: Optional[float]) -> TrialReceipt:
        metric = eval_receipt.metric
        verdict = self._verdict(metric, best_metric, outcome.returncode)
        failure_reason = self._failure_reason(outcome, bundle)
        if verdict == "failure" and not failure_reason and eval_receipt.evidence:
            # metric=None with rc=0 (e.g. OOF incompatible with the metric)
            # leaves _failure_reason empty; surface the evaluator evidence so
            # failures stay diagnosable in outcome_*.json / run logs.
            failure_reason = (eval_receipt.evidence or "")[:300]
        if verdict == "failure":
            evidence = eval_receipt.evidence or (
                "Execution failed with returncode=%s" % outcome.returncode)
        elif best_metric is None:
            evidence = "First verified trial: metric=%.4f (%s)" % (
                metric, eval_receipt.evaluator)
        elif self._better(metric, best_metric):
            evidence = "Improved from %.4f to %.4f (%s)" % (
                best_metric, metric, eval_receipt.evaluator)
        else:
            evidence = "metric=%.4f vs best=%.4f (%s)" % (
                metric, best_metric, eval_receipt.evaluator)

        submission_path = ""
        if bundle is not None and bundle.submission_path:
            submission_path = bundle.submission_path

        return TrialReceipt(
            receipt_id=new_id("receipt"),
            spec_id=spec.spec_id,
            competition=self.competition,
            round_num=int(spec.round_num),
            returncode=outcome.returncode,
            stdout=(outcome.stdout or "")[-4000:],
            stderr=(outcome.stderr or "")[-2000:],
            metric=metric,
            metric_name=self.evaluator.metric_name,
            verdict=verdict,
            evidence=evidence,
            submission_exists=bool(submission_path),
            submission_path=submission_path,
            wall_clock_seconds=outcome.wall_clock_seconds,
            code_hash=spec.code_hash,
            verified=(verdict != "failure"),
            evaluator_receipt=eval_receipt.to_dict(),
            failure_reason=failure_reason,
        )

    def _load_bundle(self, trial_id: str) -> Optional[CandidateBundle]:
        """Find the CandidateBundle bound to a trial.

        Bundle files are named bundle_<bundle_id>.json; the binding lives in
        the payload (trial_id), so lookup is by content, never by filename
        guess.
        """
        for p in self.bus.host_bundles.glob("bundle_*.json"):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if d.get("trial_id") == trial_id:
                return CandidateBundle.from_dict(d)
        return None

    @staticmethod
    def _failure_reason(outcome: ExecOutcome,
                        bundle: Optional[CandidateBundle]) -> str:
        """Human-readable first cause of a failed trial (for logs/reports)."""
        if outcome.timed_out:
            return "timeout after %ss" % outcome.wall_clock_seconds
        if outcome.returncode != 0:
            tail = [line.strip() for line in
                    ((outcome.stderr or "") + "\n" + (outcome.stdout or ""))
                    .strip().splitlines() if line.strip()][-3:]
            msg = " ".join(tail)
            if not msg:
                msg = "exit code %s" % outcome.returncode
            return msg[:300]
        if bundle is None:
            return "no verifiable artifacts (submission.csv/oof.csv missing)"
        return ""

    def supervise_once(self, grant: dict, profile, plan_obj,
                       best_metric: Optional[float]) -> Optional[TrialReceipt]:
        """Claim, validate, execute, evaluate, promote and report ONE child."""
        pending = self.bus.list_pending()
        if not pending:
            return None
        # Serve the FIRST proposal bound to this grant; orphaned proposals
        # from a crashed previous daemon are quarantined instead of killing
        # this daemon with proposal_grant_mismatch.
        proposal = None
        for candidate in pending:
            try:
                self._validate_proposal(candidate, grant)
            except FileBusError as e:
                self.bus.quarantine_pending(
                    candidate.get("proposal_id", ""), reason=str(e))
                continue
            proposal = candidate
            break
        if proposal is None:
            return None
        proposal_id = proposal["proposal_id"]

        if not self.bus.claim(proposal_id, host_id=self.host_id):
            return None

        def _phase(msg: str) -> None:
            try:
                print("%s SUPERVISE %s" % (
                    time.strftime("%H:%M:%S", time.gmtime()), msg),
                    flush=True)
            except Exception:  # noqa: BLE001 - observability is best-effort
                pass

        _phase("claim proposal=%s child=%s" % (
            proposal_id, proposal.get("child_index")))
        try:
            _t0 = time.time()
            spec = self._materialize_spec(proposal, grant, profile, plan_obj)
            _phase("implement done spec=%s dt=%.1fs" % (
                spec.spec_id, time.time() - _t0))
            _t0 = time.time()
            outcome = self._execute(spec, plan_obj, profile)
            _phase("exec done spec=%s dt=%.1fs rc=%s" % (
                spec.spec_id, time.time() - _t0, outcome.returncode))
            fallback_note = self._ensure_artifacts(spec, outcome)
            # OOF policy (V2.2): MLE-Bench scores submission.csv on the
            # hidden test set only; how the agent uses train.csv is its own
            # choice. In-sample OOF predictions are therefore ALLOWED and
            # scored; the audit below only adds an informational note so
            # receipts/feedback stay honest that such a metric is a
            # training-fit signal, not a generalization estimate.
            insample_note = "" if fallback_note else \
                self._oof_semantics_note(spec, outcome, profile)
            eval_receipt = self._evaluate(proposal_id, spec, outcome,
                                          force=bool(fallback_note))

            bundle = self._load_bundle(spec.spec_id)

            receipt = self._make_receipt(spec, outcome, eval_receipt, bundle,
                                         best_metric)
            if fallback_note:
                receipt.failure_reason = (
                    (receipt.failure_reason or "").strip() + " | "
                    + fallback_note)[:400]
            if insample_note:
                receipt.evidence = (
                    ((receipt.evidence or "") + " | IN-SAMPLE: "
                     + insample_note)[:600])
            self.bus.save_receipt(receipt.to_dict())
            if self.ledger is not None:
                self.ledger.append(receipt, spec.plan_obj(), spec.code)

            promo = self.promotion.promote(
                spec.spec_id, receipt.metric, receipt.evidence,
                verified=(receipt.returncode == 0))
            new_best = promo.certified_best_metric

            outcome_msg = {
                "proposal_id": proposal_id,
                "child_index": proposal.get("child_index"),
                "competition": self.competition,
                "trial_id": spec.spec_id,
                "verdict": receipt.verdict,
                "metric": receipt.metric,
                "metric_name": receipt.metric_name,
                "evaluator": eval_receipt.evaluator,
                "evidence": receipt.evidence,
                "returncode": receipt.returncode,
                "failure_reason": receipt.failure_reason,
                "stderr": (receipt.stderr or "")[-500:],
                "submission_exists": receipt.submission_exists,
                "submission_path": receipt.submission_path,
                "code_hash": receipt.code_hash,
                "certified_best": promo.certified_best_metric,
                "certified_best_trial_id": promo.certified_best_trial_id,
                "created_at": now_iso(),
            }
            _phase("done proposal=%s verdict=%s metric=%s wall=%.1fs"
                       % (proposal_id, receipt.verdict, receipt.metric,
                          receipt.wall_clock_seconds))
            self.bus.write_outcome(proposal_id, outcome_msg)
            self.bus.ack(proposal_id, host_id=self.host_id)
            self.bus.release_lease(proposal_id)
            return receipt
        except Exception:  # noqa: BLE001 - fail-closed, release claim
            self.bus.release_lease(proposal_id)
            raise

    def run_children(self, grant: dict, profile, plan_obj,
                     max_children: Optional[int] = None) -> List[TrialReceipt]:
        """Drive all pending children until the grant budget is consumed."""
        receipts = []
        budget = int(grant.get("trial_budget") or 0)
        limit = budget if max_children is None else min(budget, int(max_children))
        best = None
        for _ in range(limit):
            receipt = self.supervise_once(grant, profile, plan_obj, best)
            if receipt is None:
                break
            receipts.append(receipt)
            if receipt.metric is not None and receipt.returncode == 0:
                if best is None or self._better(receipt.metric, best):
                    best = receipt.metric
        return receipts

    def serve(self, grant: dict, profile, plan_obj,
              max_children: Optional[int] = None,
              poll_interval: float = 2.0,
              idle_exit_seconds: int = 120,
              stop_file: Optional[str] = None,
              max_wall_seconds: Optional[int] = None) -> List[TrialReceipt]:
        """Resident daemon loop: poll pending_agent and drive one child per
        proposal to a verified outcome.

        This is the independent HostSupervisorService behavior from the
        architecture: it starts before the agent proposes, claims each
        proposal as it appears, and exits when (a) the grant budget is
        consumed, (b) the stop marker appears, (c) the wall-clock budget
        expires, or (d) no new proposal arrives for idle_exit_seconds.
        Returns the receipts in child order.
        """
        budget = int(grant.get("trial_budget") or 0)
        limit = budget if max_children is None else min(budget, int(max_children))
        receipts: List[TrialReceipt] = []
        best: Optional[float] = None
        started = time.time()
        last_activity = time.time()
        while len(receipts) < limit:
            if stop_file and os.path.exists(stop_file):
                break
            if max_wall_seconds and time.time() - started > max_wall_seconds:
                break
            try:
                receipt = self.supervise_once(grant, profile, plan_obj, best)
            except Exception as exc:  # noqa: BLE001 - daemon must survive
                # A single bad proposal (or transient IO) must not kill the
                # resident daemon: log, quarantine if possible, keep polling.
                import traceback as _tb
                print("SUPERVISE_ERROR: %r" % (exc,), flush=True)
                _tb.print_exc()
                time.sleep(1.0)
                continue
            if receipt is not None:
                receipts.append(receipt)
                last_activity = time.time()
                if receipt.metric is not None and receipt.returncode == 0:
                    if best is None or self._better(receipt.metric, best):
                        best = receipt.metric
                continue
            if time.time() - last_activity > idle_exit_seconds:
                break
            time.sleep(max(0.2, float(poll_interval)))
        return receipts

    def terminal_outcome(self, grant: dict) -> ActionOutcome:
        promo = self.promotion.certified_best()
        outcome = ActionOutcome(
            grant_id=grant.get("grant_id", ""),
            competition=self.competition,
            trials_completed=len(self.bus.list_outcomes()),
            certified_best_trial_id=promo.certified_best_trial_id,
            certified_best_metric=promo.certified_best_metric,
        )
        summary_lines = []
        for o in sorted(self.bus.list_outcomes(),
                        key=lambda x: int(x.get("child_index") or 0)):
            summary_lines.append(
                "R%s %s metric=%s" % (o.get("child_index"), o.get("verdict"),
                                      o.get("metric")))
        outcome.program_summary = "; ".join(summary_lines)
        self.bus.append_terminal(outcome.to_dict())
        return outcome
