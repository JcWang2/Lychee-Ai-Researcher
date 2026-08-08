# -*- coding: utf-8 -*-
"""v2_closed_loop.py - V2.2 four-layer closed-loop director.

Architecture (per design):
  Outer research loop: AI Scientist + HERA
      Analysis -> Plan -> Method Portfolio -> PrioritizationTicket
      -> freeze ResearchProgramGrant + SnapshotReadyV3 (protocol/frozen_visible/)
  Inner PACT L1 transactional loop:
      ProgramAgentClient proposes (pending_agent/)
      HostSupervisorService claims/validates/executes/evaluates/promotes
      (claimed_host/ -> candidate bundles -> evaluator receipt -> outcomes_visible/)
  Certified publish layer:
      ControlledPublisher publishes only the certified-best bundle
      -> submission.csv
  Bottom File-as-Bus:
      workspace/ (agent-visible), protocol/ (role-separated),
      pact_control_host/ (host-only)
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from hera import (Analyzer, Interpreter, MethodPortfolio, Planner,  # noqa: E402
                  Prioritizer, ResourceProfiler, ScientificMemory)
from hera.portfolio import (VALID_INTENTS, estimate_grant_cost,  # noqa: E402
                            resolve_children)
from data_layout import (materialize_dataset, sanitize_test_csv,
                        synthesize_train_labels)  # noqa: E402
from metrics_registry import (DEFAULT_MIN_DELTA,  # noqa: E402
                              get_metric_spec)  # noqa: E402
from pact import (BudgetGuard, CandidateBundler, ControlledPublisher,  # noqa: E402
                  Executor, FileBus, GuardError, HostSupervisorService,
                  Implementer, PactLedger, PromotionManager, ProgramAgentClient,
                  TrustedEvaluator, assert_legacy_l1_mode)  # noqa: E402
from pact.file_bus import safe_artifact_name  # noqa: E402
from v2_llm import codegen_llm_call, default_llm_call  # noqa: E402
from stage_controller import StageController, metric_norm  # noqa: E402
from v2_contracts import (MethodInvocationV1, TrialReceipt,  # noqa: E402
                          canonical_hash, now_iso)  # noqa: E402
from capability_registry import (CapabilityRegistry, MethodSpec,  # noqa: E402
                                 load_ephemeral_path,  # noqa: E402
                                 load_synthesis_usage,  # noqa: E402
                                 save_synthesis_usage)  # noqa: E402
from program_compiler import ProgramCompiler  # noqa: E402


def _daemon_log_tail(path, n: int = 25) -> str:
    """Last n lines of a daemon log (diagnostics when a round yields nothing)."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            tail = fh.readlines()[-n:]
        return "".join(tail).strip()
    except (OSError, ValueError):
        return ""


def build_parser():
    parser = argparse.ArgumentParser(
        description="v2.2.1-rc4 closed loop: outer HERA + inner PACT transactional loop")
    parser.add_argument("--competition", default=os.environ.get("COMPETITION", "unknown"))
    parser.add_argument("--task-prompt", default=os.environ.get("TASK_PROMPT", ""))
    parser.add_argument("--max-rounds", type=int,
                        default=int(os.environ.get("MAX_ROUNDS", "128")))
    parser.add_argument("--max-grants", type=int,
                        default=int(os.environ.get("MAX_GRANTS", "128")))
    parser.add_argument("--max-total-trials", type=int,
                        default=int(os.environ.get("MAX_TOTAL_TRIALS", "256")))
    parser.add_argument("--round-timeout", type=int,
                        default=int(os.environ.get("ROUND_TIMEOUT", "3600")))
    parser.add_argument("--total-budget", type=int,
                        default=int(os.environ.get(
                            "TOTAL_WALL_CLOCK",
                            os.environ.get("PACT_TOTAL_WALL_CLOCK_SECONDS",
                                           "86400"))))
    parser.add_argument("--pretrained-policy",
                        choices=["cache", "scratch", "auto"],
                        default=os.environ.get("PRETRAINED_POLICY", "cache"))
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "/home/data"))
    parser.add_argument("--work-dir", default=os.environ.get("WORK_DIR", "/home/code"))
    parser.add_argument("--submission-dir",
                        default=os.environ.get("SUBMISSION_DIR", "/home/submission"))
    parser.add_argument("--state-dir", default=os.environ.get("STATE_DIR", "/mnt/workspace"))
    parser.add_argument("--sample-path", default=os.environ.get("SAMPLE_PATH", ""))
    parser.add_argument("--trial-budget", type=int,
                        default=int(os.environ.get("TRIAL_BUDGET", "3")))
    parser.add_argument("--exec-image", default=os.environ.get("V2_EXEC_IMAGE", ""))
    parser.add_argument("--exec-python",
                        default=os.environ.get("V2_EXEC_PYTHON", "python3"))
    parser.add_argument("--stagnation-limit", type=int,
                        default=int(os.environ.get("V2_STAGNATION_LIMIT", "6")))
    parser.add_argument("--preflight", choices=["strict", "warn", "off"],
                        default=os.environ.get("V2_PREFLIGHT", "strict"))
    parser.add_argument("--torch-cache",
                        default=os.environ.get("V2_TORCH_CACHE", ""))
    parser.add_argument("--host-daemon", action="store_true",
                        default=os.environ.get("V2_HOST_DAEMON", "0")
                        in ("1", "true", "yes"))
    parser.add_argument("--daemon-poll-interval", type=float, default=2.0)
    parser.add_argument("--daemon-idle-exit-seconds", type=int, default=600)
    parser.add_argument("--daemon-python", default=sys.executable)
    return parser


class ClosedLoop:
    """Director: coordinates outer HERA loop and inner PACT transactional loop."""

    def __init__(self, args, llm_call_fn=None):
        self.competition = args.competition
        self.task_prompt = args.task_prompt
        self.max_rounds = max(1, args.max_rounds)
        self.max_grants = max(1, getattr(args, "max_grants", 128))
        self.max_total_trials = max(1, getattr(args, "max_total_trials", 256))
        self.total_budget = max(1, args.total_budget)
        self.round_timeout = max(1, args.round_timeout)
        # legacy static per-grant budget; v2.2 grants use the HERA-chosen
        # intent -> children mapping instead (this stays as fallback)
        self.trial_budget = max(1, args.trial_budget)
        self.pretrained_policy = getattr(args, "pretrained_policy", "cache")
        self.data_dir = Path(args.data_dir)
        self.work_dir = Path(args.work_dir)
        self.submission_dir = Path(args.submission_dir)
        self.state_dir = Path(args.state_dir)
        self.sample_path = args.sample_path or ""
        for d in (self.data_dir, self.work_dir, self.submission_dir, self.state_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Materialize zipped MLE-bench images BEFORE layout resolution so the
        # analyzer/implementer/preflight always see real image directories.
        materialized = materialize_dataset(self.data_dir)
        if materialized.get("extracted"):
            print("[%s] materialize: extracted %s"
                  % (time.strftime("%H:%M:%S", time.gmtime()),
                     ", ".join(materialized["extracted"])), flush=True)
        # Some image competitions ship NO public train labels table (labels
        # live in class folders / filename tokens, e.g. plant-seedlings,
        # dogs-vs-cats): synthesize the train CSV before analysis.
        synthesized = synthesize_train_labels(self.data_dir)
        if synthesized.get("written"):
            print("[%s] synthesize: wrote train CSV %s mode=%s rows=%s"
                  % (time.strftime("%H:%M:%S", time.gmtime()),
                     synthesized["written"], synthesized.get("mode", ""),
                     synthesized.get("rows", 0)), flush=True)
        # Remove the gold target column from the test source so candidate code
        # never reads private labels (physical isolation, not just a gate).
        sanitized = sanitize_test_csv(self.data_dir)
        if sanitized.get("written"):
            print("[%s] sanitize: wrote label-free test CSV %s"
                  % (time.strftime("%H:%M:%S", time.gmtime()),
                     sanitized["written"]), flush=True)

        # Bottom File-as-Bus under state_dir
        self.bus = FileBus(self.state_dir)

        self.guard = BudgetGuard(self.total_budget, self.round_timeout,
                                 max_grants=self.max_grants,
                                 max_total_trials=self.max_total_trials,
                                 state_dir=self.state_dir)
        # v2.2.1 crash consistency: reconcile pending budget reservations
        # against grants actually frozen on the File-as-Bus.
        try:
            _rec = self.guard.recover_pending(
                frozen_grants=self.bus.list_frozen())
            if _rec.get("recovered") or _rec.get("discarded"):
                print("[%s] budget recovery: committed=%s discarded=%s"
                      % (time.strftime("%H:%M:%S", time.gmtime()),
                         _rec.get("recovered"), _rec.get("discarded")),
                      flush=True)
        except GuardError as e:
            print("FATAL: budget recovery failed: %s" % e, file=sys.stderr)
            raise
        self.analyzer = Analyzer(self.data_dir, self.task_prompt,
                                 sample_path=self.sample_path)
        self.planner = Planner(llm_call_fn=llm_call_fn)
        self.prioritizer = Prioritizer(llm_call_fn=llm_call_fn)
        self.interpreter = Interpreter(llm_call_fn=llm_call_fn,
                                       stagnation_limit=args.stagnation_limit)
        self.memory = ScientificMemory(self.state_dir)
        # rc4: implementer codegen uses the bounded-latency role
        # (codegen_llm_call) so a hanging LLM cannot burn 10-30min per
        # child; injected callables (tests) keep their own behavior.
        self.implementer = Implementer(
            llm_call_fn=llm_call_fn if llm_call_fn is not None
            else codegen_llm_call)
        self.executor = Executor(self.work_dir, exec_image=args.exec_image,
                                 exec_python=args.exec_python,
                                 data_dir=self.data_dir,
                                 torch_cache=getattr(args, "torch_cache", ""))
        self.preflight = args.preflight
        self.host_daemon = bool(getattr(args, "host_daemon", False))
        self.daemon_poll_interval = max(0.5, float(
            getattr(args, "daemon_poll_interval", 2.0)))
        self.daemon_idle_exit_seconds = max(10, int(
            getattr(args, "daemon_idle_exit_seconds", 600)))
        self.daemon_python = getattr(args, "daemon_python", "") or sys.executable
        self.ledger = PactLedger(self.state_dir)

        spec = get_metric_spec(self.competition)
        self.metric_spec = spec
        self.evaluator = TrustedEvaluator(
            metric_name=spec["metric_name"],
            metric_direction=spec["metric_direction"],
            metric_alignment=spec["metric_alignment"],
            metric_label=spec["metric_label"],
            metric_params=spec["metric_params"])
        self.bundler = CandidateBundler(self.bus, self.work_dir)
        self.promotion = PromotionManager(
            self.bus,
            metric_direction=spec["metric_direction"],
            min_delta=float(spec.get("min_delta", DEFAULT_MIN_DELTA)))
        # v2.3 template-compiled execution: HERA picks methods, the
        # platform compiler renders them deterministically. The registry
        # persists run-local synthesized capabilities under state_dir.
        self.registry = CapabilityRegistry(
            ephemeral_path=load_ephemeral_path(self.state_dir))
        self.compiler = ProgramCompiler(self.registry)
        self.host = HostSupervisorService(
            bus=self.bus, executor=self.executor, bundler=self.bundler,
            evaluator=self.evaluator, promotion=self.promotion,
            implementer=self.implementer, compiler=self.compiler,
            registry=self.registry, guards=self.guard,
            ledger=self.ledger, competition=self.competition,
            max_budget_seconds=self.round_timeout,
            data_dir=self.data_dir, sample_path=self.sample_path,
            state_dir=self.state_dir,
            metric_min_delta=float(spec.get("min_delta",
                                            DEFAULT_MIN_DELTA)))
        self.publisher = ControlledPublisher(self.bus, self.submission_dir)

        self.round_num = 0
        self.best_metric = None
        self.best_receipt_id = None
        self.stagnation_count = 0
        self.last_interpretation = None
        self.total_trials = 0
        self.failed_trials = 0
        self.start_time = time.time()
        # v2.2.1-rc3: restore scientific continuity from trusted facts
        # (certified promotion record + evidence ledger + incumbent asset)
        # so a restart can never mistake a regression for NEW BEST, reset
        # stagnation, or lose the round counter.
        self._recover_scientific_state()

        # v2.2: derived resource profile + stage controller + F0 calibration
        self.resource = {}
        self.stage = None
        self._cached_weights_seen = []
        self.f0_calibration = self._load_f0()
        # v2.2.1-rc4: multi-size image cache map (size -> dir), exposed to
        # candidate code (V2_CACHE_DIRS) and HERA (platform facts).
        self.cache_dirs = {}
        self._manifest = {}

    def run(self) -> dict:
        self._log("V2.2 four-layer closed loop start: %s" % self.competition)
        self._log("exec_mode=%s" % self.executor.exec_mode())
        profile = self.analyzer.profile(self.competition)
        self.f0_calibration = self._load_f0(profile)
        self._log("profile: task_type=%s modality=%s rows=%s/%s text_cols=%s time_col=%s"
                  % (profile.task_type, profile.modality, profile.train_rows,
                     profile.test_rows,
                     ",".join(profile.text_columns) or "-",
                     profile.time_column or "-"))
        self.resource = self._derive_resource(profile)
        self._log("resource: %s" % json.dumps(self.resource,
                                              ensure_ascii=False))
        self.stage = StageController(self.state_dir, profile, self.resource,
                                     max_grants=self.max_grants,
                                     total_wall_clock=self.total_budget)
        self._log("stage: %s" % self.stage.guidance_summary())
        manifest = self._build_manifest(profile)
        self.executor.set_manifest(manifest)
        self.host.gold_test_csv = manifest.get("gold_test_csv", "") or ""
        self.host.test_csv = manifest.get("test_csv", "") or ""
        (self.work_dir / "data_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        self._log("manifest: layout=%s test_has_labels=%s images=%s"
                  % (manifest.get("layout"), manifest.get("test_has_labels"),
                     bool(manifest.get("train_images"))))
        if self.executor.exec_mode() == "container" and self.preflight != "off":
            pf = self.executor.preflight()
            self._log("preflight: mode=%s status=%s missing_modules=%s missing_files=%s"
                      % (pf.get("mode"), pf.get("status"),
                         pf.get("missing_modules") or [],
                         pf.get("missing_files") or []))
            if pf.get("status") == "fail" and self.preflight == "strict":
                self._log("PREFLIGHT WARN (non-fatal): %s"
                           % (pf.get("detail") or "")[:500])
                self._log("DEGRADED MODE: continuing without strict preflight; fallback artifacts will cover failures")
                self.preflight = "off"
            if pf.get("pretrained_available"):
                manifest["pretrained_available"] = pf["pretrained_available"]
                self.executor.set_manifest(manifest)
                (self.work_dir / "data_manifest.json").write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False),
                    encoding="utf-8")
                self._cached_weights_seen = list(pf["pretrained_available"])
                # v2.2.1-rc3: the cached-weight profile is now known, so any
                # persisted F0 calibration must also match it; otherwise it
                # is discarded before it can steer resource estimation.
                self.f0_calibration = self._load_f0(profile)
                self.resource = self._derive_resource(
                    profile, cached_weights=self._cached_weights_seen)
                self._log("pretrained cache: %d weights (%s)"
                          % (len(pf["pretrained_available"]),
                             ", ".join(pf["pretrained_available"][:6])))

        self._manifest = manifest
        if manifest.get("train_images") and not manifest.get("pixel_level"):
            # v2.3.7: pixel-level layouts (paired-image regression) skip the
            # per-image zero-decode cache: train.csv rows are PIXELS, not
            # images, so the cache id order can never match the CSV; the
            # dedicated pixel baseline reads the synthesized pixel CSV and
            # the sample submission instead.
            # v2.2.1-rc4: multi-size zero-decode cache (decode ONCE at the
            # largest derived size, downscale in memory to every smaller
            # size). Generic for ANY MLE-Bench image task: sizes come from
            # the GPU-memory-derived image_size_max and the standard
            # 64/128/192/256 ladder (V2_CACHE_SIZES overrides) - never from
            # the competition name. Failure is non-fatal (trials fall back
            # to in-container decoding, and lazy F0 stays active).
            try:
                from pact.data_cache import (ensure_image_caches,
                                             parse_sizes)
                _res = self.resource
                _max_size = int(_res.get("image_size_max") or 192)
                _env_sizes = (os.environ.get("V2_CACHE_SIZES") or "").strip()
                _sizes = parse_sizes(_env_sizes) if _env_sizes \
                    else (64, 128, 192, 256)
                _sizes = tuple(s for s in _sizes if s <= max(64, _max_size)) \
                    or (64,)
                _caches = ensure_image_caches(
                    work_dir=self.work_dir, manifest=manifest, sizes=_sizes,
                    docker_bin=self.executor.docker_bin,
                    exec_image=self.executor.exec_image,
                    exec_python=self.executor.exec_python)
                _dirs = {str(s): v.get("dir") for s, v in _caches.items()
                         if v.get("dir")}
                self.cache_dirs = _dirs
                manifest["cache_sizes"] = sorted(
                    int(s) for s in _caches)
                manifest["cache_dirs"] = _dirs
                # rc4: expose the {size: dir} map to the implementer
                # prompt (candidate code must LOAD, not decode).
                os.environ["V2_CACHE_DIRS"] = json.dumps(_dirs)
                self.executor.set_manifest(manifest)
                # v2.3.2: ALWAYS rewrite the manifest (even with an empty
                # cache map) so a stale manifest from an older run can never
                # hand trials cache dirs that no longer exist.
                (self.work_dir / "data_manifest.json").write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False),
                    encoding="utf-8")
                _first = next(iter(_caches.values()), {})
                self._log("image cache: sizes=%s rows=%s/%s"
                          % (sorted(int(s) for s in _caches),
                             _first.get("rows_train"),
                             _first.get("rows_test")))
            except Exception as _e:  # noqa: BLE001 - cache is an optimization
                self._log("image cache build failed (non-fatal): %s"
                          % str(_e)[:300])

        # v2.2.1-rc4: startup F0 calibration probe (generic, fail-open).
        # Measures a tiny fit + decode rate BEFORE grant #1 so HERA's
        # cheap_probe cost estimates are honest from the very first grant.
        self._maybe_probe_f0(profile, manifest)

        while True:
            try:
                self.guard.check_budget()
            except GuardError as e:
                self._log("guard: %s" % e)
                break
            if self.guard.grants_remaining() <= 0:
                self._log("guard: grant budget exhausted (%d/%d)"
                          % (self.guard.grants_used, self.guard.max_grants))
                break
            if self.guard.trials_remaining() <= 0:
                self._log("guard: trial budget exhausted (%d/%d)"
                          % (self.guard.trials_used,
                             self.guard.max_total_trials))
                break
            self.round_num += 1
            self._log("GRANT %d stage=%s (grants %d/%d trials %d/%d wall %ds left)"
                      % (self.round_num,
                         self.stage.stage if self.stage else "-",
                         self.guard.grants_used + 1, self.guard.max_grants,
                         self.guard.trials_used, self.guard.max_total_trials,
                         int(self.guard.remaining())))
            try:
                self._run_one_grant(profile)
            except _StopLoop:
                break
            except GuardError as e:
                self._log("guard: %s" % e)
                break
            except Exception as e:  # noqa: BLE001 - grant failure must not kill the loop
                self._log("grant %d failed: %s" % (self.round_num, e))
                continue

        self._publish_certified()
        return self._finalize()

    # ---- outer research loop ----
    def _run_one_grant(self, profile):
        # 0) PRE-GRANT wall-clock clipping (v2.2.1): if a full grant can no
        #    longer fit, switch to S4 BEFORE the planner runs so the next
        #    grant is planned cheap instead of being refused by the guard.
        if self.stage is not None:
            _prev = self.stage.stage
            _clipped = self.stage.pre_grant_clip(self.guard.remaining())
            if _clipped != _prev:
                self._log("STAGE TRANSITION (pre-grant clipping) -> %s"
                          % _clipped)

        # 1) Method portfolio + derived resource profile (HERA owns the
        #    method space). Resource comes from the competition-agnostic
        #    ResourceProfiler (modality/rows/dims/GPU/cache/F0), never from
        #    the competition name.
        portfolio_path = os.environ.get("PORTFOLIO_FILE", "") or str(
            self.state_dir / "portfolio.json")
        portfolio = MethodPortfolio.load_or_default(profile, portfolio_path)
        portfolio.resource_profile = dict(self.resource)
        resource = portfolio.resource_profile
        # 2) HERA plan (evidence capped to bound LLM latency); stage guidance
        #    and intent whitelist are soft context, the method choice stays
        #    with HERA.
        plan = self.planner.plan(
            profile=profile,
            evidence=(self._evidence_summary(profile) or "")[:2500],
            round_num=self.round_num,
            elapsed=int(self.guard.elapsed()),
            total_budget=self.guard.total_budget,
            resource=resource,
            stage_block=(self.stage.prompt_block() if self.stage else ""),
            intent_hints=(self.stage.intent_hints() if self.stage else None),
        )
        # 2b) PROVISIONAL intent/children from the planner. The prioritizer
        #     is the FINAL research-intent authority (v2.2.1), so children
        #     are re-derived from the FINAL intent after prioritization.
        provisional_intent = self._validated_intent(plan.research_intent)
        provisional_children = resolve_children(
            provisional_intent,
            requested=(plan.method_detail.get("children")
                       or self.trial_budget))
        # 3) Prioritization ticket (HERA may write new branches)
        ticket = self.prioritizer.prioritize(
            profile, portfolio, plan, trial_budget=provisional_children,
            research_intent=provisional_intent,
            stage=self.stage.stage if self.stage else "",
            platform_facts=self._platform_facts())
        portfolio.save()
        # 3b) SINGLE-AUTHORITY ORDER (v2.2.1):
        #     Planner proposal -> Prioritizer FINAL intent -> stage policy
        #     validation -> children from the FINAL intent -> budget check
        #     -> atomic freeze (reserve -> freeze -> commit receipt).
        final_intent = self._validated_intent(ticket.research_intent)
        children = resolve_children(
            final_intent,
            requested=(plan.method_detail.get("children")
                       or self.trial_budget))
        if (final_intent != ticket.research_intent
                or children != ticket.trial_budget):
            self._log("intent authority: %s(%d) -> %s(%d)"
                      % (ticket.research_intent, ticket.trial_budget,
                         final_intent, children))
            ticket.research_intent = final_intent
            ticket.trial_budget = children
            ticket.ticket_hash = ticket.compute_hash()
        est_cost = estimate_grant_cost(self.resource, children, final_intent)
        # Cap the estimate by the enforceable worst case: every child trial
        # is bounded by the round timeout, so a grant can never exceed
        # children * round_timeout even if the F0-based estimate is high.
        est_cost = min(est_cost, children * self.round_timeout)
        try:
            self.guard.check_research_opportunity(children, est_cost)
        except GuardError as e:
            self._log("guard refuses grant %d: %s" % (self.round_num, e))
            raise
        self._log("grant plan: stage=%s intent=%s children=%d est_cost=%ds"
                  % (self.stage.stage if self.stage else "-", final_intent,
                     children, int(est_cost)))
        # 3c) freeze grant + SnapshotReadyV3 on the File-as-Bus, atomically
        #     bound to the budget: reservation (pending) -> freeze -> commit
        #     receipt. A crash anywhere in between is reconciled on restart.
        grant = self.prioritizer.freeze_grant(
            self.competition, self.task_prompt, plan, ticket,
            stage=self.stage.stage if self.stage else "")
        ready = self.prioritizer.snapshot_ready(grant)
        self.guard.begin_reservation(children, grant.grant_id)
        try:
            self.bus.freeze_grant(grant.to_dict(), ready.to_dict())
        except Exception:
            self.guard.cancel_reservation(grant.grant_id)
            raise
        self.guard.commit_grant(children, grant.grant_id)
        self.bus.write_prioritized_tasks_md(
            "# Prioritized tasks\n- branch=%s axis=%s intent=%s children=%d budget=%ds\n"
            % (ticket.selected_branch_id, ticket.mutation_axis,
               ticket.research_intent, ticket.trial_budget,
               plan.max_budget_seconds))
        self.bus.write_plan_md("# Plan (grant %d)\n%s\n"
                               % (self.round_num, plan.hypothesis))
        incumbent = self._load_incumbent()
        if incumbent:
            self._log("grant %d inherits incumbent: round=%s metric=%s code=%s"
                      % (self.round_num, incumbent.get("round_num"),
                         incumbent.get("metric"), incumbent.get("code_path")))
        self._log("grant: %s branch=%s axis=%s intent=%s children=%d"
                  % (grant.grant_id, ticket.selected_branch_id,
                     ticket.mutation_axis, ticket.research_intent,
                     ticket.trial_budget))

        # 4) inner transactional loop: FeedbackView drives children one at a
        # time. The agent proposes child N+1 ONLY after child N's verified
        # outcome is collected, so the next proposal (and the code the
        # implementer writes) adapts to what already happened. The host side
        # runs inline (single process) or as an independent resident daemon.
        agent = ProgramAgentClient(self.bus, grant.to_dict(),
                                   host_id="agent",
                                   proposer=self._agent_proposer(grant.to_dict(),
                                                                 profile))
        best_before = self.best_metric
        if self.host_daemon:
            receipts = self._run_grant_daemon(agent, grant, profile, plan,
                                              ticket)
        else:
            receipts = self._run_grant_inline(agent, grant, profile, plan,
                                              ticket)
        if not receipts:
            self._log("grant %d: no children executed" % self.round_num)
            self._after_grant(profile, final_intent, children, best_before,
                              receipts, [])
            return
        failures = [r for r in receipts if r.verdict == "failure"]
        if failures:
            reasons = " || ".join(
                (r.failure_reason or "unknown")[:160] for r in failures[:3])
            self._log("round failures: %d/%d %s"
                      % (len(failures), len(receipts), reasons[:400]))

        self._after_grant(profile, final_intent, children, best_before, receipts,
                          [r for r in receipts if r.verdict == "regression"])

        # 5) HERA interpretation of the terminal outcome + memory update
        outcome = self.host.terminal_outcome(grant.to_dict())
        last_receipt = receipts[-1]
        interpretation = self.interpreter.interpret(
            last_receipt, self.best_metric, self.stagnation_count,
            self.max_rounds, int(self.guard.elapsed()), self.guard.total_budget)
        self.memory.update(plan, last_receipt, self.task_prompt,
                           self.best_metric)
        self.last_interpretation = interpretation
        self._log("interpret: %s %s" % (interpretation.stop_decision,
                                        outcome.program_summary))

        if interpretation.stop_decision == "stop":
            self._log("stop: %s" % (interpretation.next_direction or "decision"))
            self._stop = True
            raise _StopLoop()
        if interpretation.stop_decision == "switch_approach":
            self.stagnation_count = 0

    # ---- v2.2.1-rc3: scientific state recovery (restart continuity) ----
    def _ledger_best_record(self) -> Optional[dict]:
        """Direction-aware best VERIFIED record from the evidence ledger
        (metric is not None and returncode == 0)."""
        direction = self.metric_spec.get("metric_direction", "higher_is_better")
        best_rec = None
        for rec in self.ledger.trials(self.competition):
            m = rec.get("metric")
            if m is None or int(rec.get("returncode") or 0) != 0:
                continue
            try:
                m = float(m)
            except (TypeError, ValueError):
                continue
            if best_rec is None:
                best_rec = rec
                continue
            try:
                bm = float(best_rec["metric"])
            except (TypeError, ValueError, KeyError):
                best_rec = rec
                continue
            if direction == "lower_is_better" and m < bm:
                best_rec = rec
            elif direction != "lower_is_better" and m > bm:
                best_rec = rec
        return best_rec

    def _ledger_best_metric(self) -> Optional[float]:
        """Direction-aware best verified metric from the evidence ledger."""
        rec = self._ledger_best_record()
        if rec is None:
            return None
        try:
            return float(rec["metric"])
        except (TypeError, ValueError, KeyError):
            return None

    def _ledger_stagnation(self, window: int = 3) -> int:
        """Consecutive non-improving verified trials in the recent ledger
        window (direction-aware), mirroring ExperimentLedger.count_stagnation
        but correct for lower-is-better metrics."""
        recs = [r for r in self.ledger.trials(self.competition)
                if r.get("metric") is not None
                and int(r.get("returncode") or 0) == 0][-window:]
        if len(recs) < 2:
            return 0
        direction = self.metric_spec.get("metric_direction", "higher_is_better")
        best = None
        count = 0
        for r in recs:
            try:
                m = float(r["metric"])
            except (TypeError, ValueError):
                continue
            if best is None:
                best = m
                continue
            if direction == "lower_is_better":
                improved = m < best - 1e-12
            else:
                improved = m > best + 1e-12
            if improved:
                best = m
                count = 0
            else:
                count += 1
        return count

    def _recover_scientific_state(self) -> None:
        """Restore best_metric / best_receipt_id / round_num / stagnation /
        trial counters from trusted persisted facts. Priority:
          best_metric      promotion certified -> ledger best -> incumbent
          best_receipt_id  promotion certified id -> incumbent receipt_id
        A regression receipt after restart therefore compares against the
        REAL previous best instead of None."""
        promo = self.promotion.certified_best()
        cert_metric = getattr(promo, "certified_best_metric", None)
        cert_id = str(getattr(promo, "certified_best_trial_id", "") or "")
        inc = self._load_incumbent() or {}
        ledger_rec = self._ledger_best_record()
        ledger_best = self._ledger_best_metric()
        best = cert_metric if cert_metric is not None else ledger_best
        if best is None:
            best = inc.get("metric")
        # v2.3.6: the certified pointer can lag behind the ledger when the
        # old global 0.01 min_delta rejected a real improvement (e.g. AUC
        # 0.9997 vs 0.9972: +0.0025 < 0.01). Restore the BETTER of the two
        # and, when the ledger wins, sync its code asset so metric and code
        # stay in lockstep (new metric + old code must never survive a restart).
        from_ledger = bool(ledger_best is not None
                           and self._is_better(ledger_best, cert_metric))
        if from_ledger:
            best = ledger_best
        self.best_metric = best
        if from_ledger and ledger_rec is not None:
            self.best_receipt_id = str(ledger_rec.get("trial_id") or "")
        else:
            self.best_receipt_id = cert_id or str(inc.get("receipt_id") or "")
        max_ledger_round = 0
        for rec in self.ledger.trials(self.competition):
            try:
                max_ledger_round = max(
                    max_ledger_round, int(rec.get("round_num") or 0))
            except (TypeError, ValueError):
                pass
        self.round_num = max(self.guard.grants_used, max_ledger_round)
        self.stagnation_count = self._ledger_stagnation()
        all_trials = self.ledger.trials(self.competition)
        self.total_trials = len(all_trials)
        self.failed_trials = sum(
            1 for r in all_trials if str(r.get("verdict") or "") == "failure")
        if from_ledger and ledger_rec is not None:
            self._sync_incumbent_from_ledger(ledger_rec)
        if self.best_metric is not None or self.round_num > 0:
            self._log("restored scientific state: best=%s receipt=%s "
                      "round=%d stagnation=%d trials=%d"
                      % (self.best_metric, self.best_receipt_id,
                         self.round_num, self.stagnation_count,
                         self.total_trials))

    def _receipt_spec_id(self, receipt_id: str) -> str:
        """Map a receipt_id back to its spec_id via the host receipt store."""
        try:
            for p in self.bus.host_receipts.glob("receipt_*.json"):
                d = json.loads(p.read_text(encoding="utf-8"))
                if str(d.get("receipt_id") or "") == str(receipt_id):
                    return str(d.get("spec_id") or "")
        except (OSError, ValueError, AttributeError):
            pass
        return ""

    def _sync_incumbent_from_ledger(self, rec: dict) -> bool:
        """Best-effort: when the ledger holds a better verified metric than
        the certified pointer (old min_delta rejected it), persist its code
        as the incumbent asset so metric and code never drift. Non-fatal."""
        try:
            trial_id = str(rec.get("trial_id") or "")
            if not trial_id:
                return False
            spec_id = self._receipt_spec_id(trial_id)
            src = None
            if spec_id:
                cand = self.bus.ws_code / ("trial_" + spec_id + ".py")
                if cand.is_file():
                    src = cand
            if src is None:
                want = str(rec.get("code_hash") or "").replace("sha256:", "")
                for cand in self.bus.ws_code.glob("trial_*.py"):
                    if not want:
                        continue
                    try:
                        digest = hashlib.sha256(
                            cand.read_bytes()).hexdigest()
                    except OSError:
                        continue
                    if want == digest:
                        src = cand
                        break
            if src is None:
                self._log("incumbent ledger sync skipped: no code for "
                          "receipt=%s" % trial_id)
                return False
            stub = types.SimpleNamespace(
                spec_id=spec_id or src.stem[len("trial_"):],
                receipt_id=trial_id,
                grant_id=str(rec.get("grant_id") or ""),
                branch_id=str(rec.get("branch_id") or ""),
                metric=rec.get("metric"),
                metric_name=str(rec.get("metric_name") or ""),
            )
            return self._save_incumbent(stub)
        except (OSError, ValueError, AttributeError) as e:  # noqa: BLE001 - best-effort
            self._log("incumbent ledger sync skipped: %s" % e)
            return False

    # ---- v2.2 helpers: stage feedback, F0 calibration, resource ----
    def _after_grant(self, profile, intent, children, best_before, receipts,
                     regressions) -> None:
        '''Shared post-grant bookkeeping (stage transitions + F0 + logs).'''
        self._maybe_calibrate_f0(receipts, profile)
        new_best = self.best_metric != best_before
        submission = (self.submission_dir / "submission.csv").is_file() or any(
            getattr(r, "submission_exists", False) for r in receipts)
        if self.stage is not None:
            previous_stage = self.stage.stage
            stage = self.stage.on_grant_result({
                "grants_used": self.guard.grants_used,
                "remaining_wall_clock": self.guard.remaining(),
                "best_metric": self.best_metric,
                "metric_norm": metric_norm(self.best_metric, profile),
                "new_best": new_best,
                "stagnation_count": self.stagnation_count,
                "submission_exists": submission,
                "regressions": len(regressions),
                "intent": intent,
                "children": children,
            })
            if stage != previous_stage:
                self._log("STAGE TRANSITION -> %s" % stage)
            self._log("stage=%s grants=%d/%d trials=%d/%d wall=%ds left"
                      % (self.stage.stage, self.guard.grants_used,
                         self.guard.max_grants, self.guard.trials_used,
                         self.guard.max_total_trials,
                         int(self.guard.remaining())))

    def _validated_intent(self, intent: str) -> str:
        '''Whitelist + stage soft-guide: keep HERA's intent when it is valid
        and allowed in the current stage; otherwise fall back to the stage
        default. The platform never replaces HERA's method decision.'''
        name = str(intent or "").strip()
        allowed = set()
        if self.stage is not None:
            allowed = set(self.stage.allowed_intents())
        if name in VALID_INTENTS and (not allowed or name in allowed):
            return name
        if name and name in VALID_INTENTS and allowed and name not in allowed:
            self._log("intent %r not allowed in %s -> stage default"
                      % (name, self.stage.stage))
        if self.stage is not None:
            return self.stage.default_intent()
        return "cheap_probe"

    def _gpu_name(self) -> str:
        """Best-effort GPU model name(s) for the runtime fingerprint."""
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
            names = sorted({n.strip() for n in out.stdout.splitlines()
                            if n.strip()})
            return ",".join(names[:4])
        except Exception:  # noqa: BLE001 - best-effort
            return ""

    def _runtime_env_fingerprint(self) -> dict:
        """Runtime factors that change trial wall time (and therefore F0):
        GPU model/memory, execution image, Python version, pretrained
        policy. The cached-weight whitelist is tracked separately (it is
        only known after preflight)."""
        return {
            "gpu_memory_mb": self._gpu_memory_mb(),
            "gpu_name": self._gpu_name(),
            "exec_image": str(getattr(self.executor, "exec_image", "") or ""),
            "python_version": ".".join(str(v) for v in sys.version_info[:3]),
            "pretrained_policy": self.pretrained_policy,
        }

    def _profile_hash(self, profile) -> str:
        """Profile fingerprint the F0 calibration is bound to: dataset shape/
        layout AND runtime environment (GPU, image, Python). A calibration
        measured on another dataset or another runtime must never be reused."""
        layout = getattr(self.analyzer, "layout", None)
        return canonical_hash({
            "train_rows": int(getattr(profile, "train_rows", 0) or 0),
            "test_rows": int(getattr(profile, "test_rows", 0) or 0),
            "image_pixels": (int(getattr(profile, "image_width", 0) or 0)
                             * int(getattr(profile, "image_height", 0) or 0)),
            "modality": str(getattr(profile, "modality", "") or ""),
            "task_type": str(getattr(profile, "task_type", "") or ""),
            "metric_name": str(getattr(profile, "metric_name", "") or ""),
            "target_column": str(getattr(profile, "target_column", "") or ""),
            "layout": str(getattr(layout, "layout_name", "") or ""),
            "test_has_labels": bool(
                getattr(layout, "test_has_labels", False)),
            "runtime": self._runtime_env_fingerprint(),
        })

    def _load_f0(self, profile=None) -> dict:
        path = self.state_dir / "f0_calibration.json"
        try:
            if path.is_file():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data:
                    ph = str(data.get("profile_hash") or "")
                    if profile is not None and ph and \
                            ph != self._profile_hash(profile):
                        self._log("f0 calibration ignored: profile hash "
                                  "mismatch (dataset/runtime changed)")
                        return {}
                    # cached-weight profile is only validated when the
                    # current whitelist is actually known (after preflight);
                    # an unknown cache list never blocks reuse on restart.
                    cache_prof = data.get("cache_profile")
                    if cache_prof and self._cached_weights_seen:
                        if sorted(str(x) for x in cache_prof) != sorted(
                                str(x) for x in self._cached_weights_seen):
                            self._log("f0 calibration ignored: cached-weight "
                                      "profile changed")
                            return {}
                    return data
        except (OSError, ValueError):
            pass
        return {}

    def _save_f0(self, seconds: float, rows: int, pixels: int,
                 profile_hash: str = "", samples=None,
                 sample_ids=None) -> None:
        """Persist a COMPLETE F0 calibration (median of >=2 successful
        samples) together with the raw samples, receipt ids and the runtime
        cached-weight profile."""
        data = {
            "schema_version": "v2_f0_v3",
            "competition": self.competition,
            "f0_seconds": round(float(seconds), 1),
            "median_seconds": round(float(seconds), 1),
            "samples_seconds": [round(float(x), 1)
                                for x in (samples or [seconds])],
            "sample_receipt_ids": [str(x) for x in (sample_ids or [])],
            "cache_profile": sorted(set(self._cached_weights_seen)),
            "train_rows": int(rows),
            "image_pixels": int(pixels),
            "profile_hash": profile_hash,
            "measured_at": now_iso(),
        }
        self._write_f0(data)

    def _save_f0_partial(self, samples: list, sample_ids: list,
                         rows: int, pixels: int,
                         profile_hash: str = "") -> None:
        """Persist fewer than 2 samples so the NEXT grant can keep
        accumulating (cross-grant F0 sampling). No median is computed yet."""
        data = {
            "schema_version": "v2_f0_v3",
            "competition": self.competition,
            "f0_seconds": None,
            "median_seconds": None,
            "samples_seconds": [round(float(x), 1) for x in samples],
            "sample_receipt_ids": [str(x) for x in sample_ids],
            "cache_profile": sorted(set(self._cached_weights_seen)),
            "train_rows": int(rows),
            "image_pixels": int(pixels),
            "profile_hash": profile_hash,
            "measured_at": now_iso(),
        }
        self._write_f0(data)

    def _write_f0(self, data: dict) -> None:
        try:
            tmp = self.state_dir / "f0_calibration.json.tmp"
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                           encoding="utf-8")
            os.replace(str(tmp), str(self.state_dir / "f0_calibration.json"))
            self.f0_calibration = data
        except OSError:
            pass

    def _maybe_calibrate_f0(self, receipts, profile) -> None:
        '''F0 runtime calibration (v2.2.1-rc3): successful trial wall-times
        are accumulated ACROSS GRANTS (persisted as samples_seconds +
        sample_receipt_ids), and the MEDIAN is computed only after at least
        2 samples (capped at 3). A single success receipt never hard-codes
        F0; a restart keeps sampling from where the previous run stopped.'''
        if not receipts:
            return
        data = self._load_f0(profile) or {}
        if data.get("median_seconds") is not None:
            return  # already calibrated with >=2 samples
        samples = [float(x) for x in (data.get("samples_seconds") or [])]
        ids = [str(x) for x in (data.get("sample_receipt_ids") or [])]
        for r in receipts:
            verdict = str(getattr(r, "verdict", "") or "")
            rc = getattr(r, "returncode", -1)
            try:
                rc = int(rc) if rc is not None else -1
            except (TypeError, ValueError):
                rc = -1
            try:
                seconds = float(getattr(r, "wall_clock_seconds", 0) or 0)
            except (TypeError, ValueError):
                seconds = 0.0
            rid = str(getattr(r, "receipt_id", "") or "")
            if verdict != "failure" and rc == 0 and seconds > 0 and rid not in ids:
                samples.append(seconds)
                ids.append(rid)
                if len(samples) >= 3:
                    break
        if not samples:
            return
        rows = int(getattr(profile, "train_rows", 0) or 0)
        pixels = 0
        if int(getattr(profile, "image_width", 0) or 0) and int(
                getattr(profile, "image_height", 0) or 0):
            pixels = int(profile.image_width) * int(profile.image_height)
        ph = self._profile_hash(profile)
        if len(samples) < 2:
            self._save_f0_partial(samples, ids, rows, pixels, ph)
            self._log("F0 samples: %d/2 (median deferred)" % len(samples))
            return
        ordered = sorted(samples)
        mid = len(ordered) // 2
        if len(ordered) % 2 == 1:
            seconds = ordered[mid]
        else:
            seconds = (ordered[mid - 1] + ordered[mid]) / 2.0
        self._save_f0(seconds, rows, pixels, ph, samples, ids)
        self.resource = self._derive_resource(
            profile, cached_weights=self._cached_weights_seen)
        self._log("F0 calibration: %.1fs (median of %d successful samples) "
                  "-> t_est=%ds"
                  % (seconds, len(samples),
                     self.resource.get("t_est_seconds")))
    def _maybe_probe_f0(self, profile, manifest) -> None:
        """v2.2.1-rc4: startup F0 calibration probe (generic, fail-open).

        Runs ONCE before grant #1 (only when no calibration exists yet) so
        HERA's first cheap_probe already gets honest est_cost. The probe is
        competition-agnostic: it measures a tiny fit (LogisticRegression for
        classification / Ridge for regression, chosen from task_type) plus
        the image decode/load rate; if nothing measurable exists it skips
        and the lazy calibration (from real trials) remains active.
        """
        try:
            data = self._load_f0(profile) or {}
            if data.get("median_seconds") is not None:
                return
            from pact.calibration_probe import project_f0, run_calibration_probe
            probe = run_calibration_probe(
                work_dir=self.work_dir, manifest=manifest,
                cache_dirs=getattr(self, "cache_dirs", {}) or {},
                docker_bin=self.executor.docker_bin,
                exec_image=self.executor.exec_image,
                exec_python=self.executor.exec_python)
            if not probe:
                self._log("F0 probe skipped (no measurable features); "
                          "lazy calibration stays active")
                return
            rows = int(getattr(profile, "train_rows", 0) or 0)
            payload = project_f0(probe, rows)
            payload["profile_hash"] = self._profile_hash(profile)
            payload["cache_profile"] = sorted(set(self._cached_weights_seen))
            self._write_f0(payload)
            self.resource = self._derive_resource(
                profile, cached_weights=self._cached_weights_seen)
            self._log("F0 probe: fit=%.2fs rows=%d mode=%s cache=%s "
                      "-> f0=%.1fs t_est=%ds"
                      % (float(probe.get("fit_seconds") or 0),
                         int(probe.get("rows_measured") or 0),
                         probe.get("feature_mode", ""),
                         bool(probe.get("image_cache")),
                         float(payload.get("f0_seconds") or 0),
                         int(self.resource.get("t_est_seconds") or 0)))
        except Exception as _e:  # noqa: BLE001 - probe is fail-open
            self._log("F0 probe failed (non-fatal): %s" % str(_e)[:200])

    def _incumbent_kind(self) -> str:
        """Best verified trial's method kind: 'learned' | 'non_learned' | ''.

        Read from the evidence ledger method_detail (generic heuristic:
        model family names hinting at priors/majority/dummy/random are
        non-learned). Facts only - never used to choose methods.
        """
        hints = ("prior", "majority", "dummy", "random", "frequency",
                 "zero_rule", "baseline_prior")
        try:
            recs = self.ledger.trials(self.competition)
        except Exception:  # noqa: BLE001
            return ""
        if not recs:
            return ""
        direction = self.metric_spec.get("metric_direction",
                                         "higher_is_better")
        best_rec = None
        for rec in recs:
            m = rec.get("metric")
            if m is None or int(rec.get("returncode") or 0) != 0:
                continue
            try:
                m = float(m)
            except (TypeError, ValueError):
                continue
            if best_rec is None:
                best_rec = rec
                continue
            bm = best_rec.get("metric")
            try:
                bm = float(bm)
            except (TypeError, ValueError):
                best_rec = rec
                continue
            if direction == "lower_is_better" and m < bm:
                best_rec = rec
            elif direction != "lower_is_better" and m > bm:
                best_rec = rec
        if best_rec is None:
            return ""
        try:
            md = best_rec.get("method_detail") or {}
            if isinstance(md, str):
                md = json.loads(md)
            model = str(md.get("model") or md.get("model_family") or "").lower()
            if any(h in model for h in hints):
                return "non_learned"
            if model:
                return "learned"
        except Exception:  # noqa: BLE001
            pass
        return ""

    def _platform_facts(self) -> str:
        """Measured platform facts for HERA/implementer (facts only, never
        method guidance). Generic across MLE-Bench tasks."""
        lines = []
        cache = getattr(self, "cache_dirs", {}) or {}
        if cache:
            lines.append(
                "PLATFORM FACTS: prebuilt image caches (uint8 arrays, "
                "decode-once) available at sizes %s; prefer them for any "
                "trial, especially cheap probes"
                % sorted(int(s) for s in cache))
        f0 = getattr(self, "f0_calibration", None) or {}
        if f0.get("median_seconds"):
            lines.append(
                "PLATFORM FACTS: measured F0 cost: one cheap probe ~= %ss "
                "(measured, not guessed)" % int(float(f0["median_seconds"])))
        kind = self._incumbent_kind()
        if kind == "non_learned":
            lines.append(
                "PLATFORM FACTS: current best method is non-learned (does "
                "not read features/images); preprocessing/feature changes "
                "cannot change its output - use them only together with "
                "switching to a learned method")
        return "\n".join(lines)
    def _gpu_memory_mb(self) -> int:
        try:
            env_v = os.environ.get("V2_GPU_MEM_MB", "").strip()
            if env_v:
                return max(0, int(env_v))
        except ValueError:
            pass
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
            vals = [int(v) for v in out.stdout.split() if v.strip().isdigit()]
            return max(vals) if vals else 0
        except Exception:  # noqa: BLE001 - GPU info is best-effort
            return 0

    def _derive_resource(self, profile, cached_weights=None) -> dict:
        '''Competition-agnostic resource derivation (modality/rows/dims +
        GPU memory + cached weights + F0 calibration).'''
        return ResourceProfiler(
            gpu_memory_mb=self._gpu_memory_mb(),
            cached_weights=cached_weights if cached_weights is not None
            else self._cached_weights_seen,
            f0_calibration=self.f0_calibration,
            pretrained_policy=self.pretrained_policy,
        ).derive(profile)

    def _agent_proposer(self, grant: dict, profile=None):
        """HERA FeedbackView proposer (v2.3): structured MethodInvocation.

        The LLM receives the capability summary + dataset contract + the
        verified outcomes of the previous children in this grant and returns
        ONE MethodInvocationV1 (method_id + params + preprocessing +
        validation). The platform compiler renders it deterministically; NO
        full-script codegen happens on this path.

        Any failure (LLM down, bad JSON, schema-rejected invocation) falls
        back to the deterministic per-child proposer (first compatible
        capability + schema defaults) so the loop never stalls. When no
        compatible capability exists, Phase C synthesis is attempted once
        per gap (bounded by MAX_SYNTHESIS_ACTIONS); only if that also fails
        does the proposal carry an empty invocation (legacy implementer
        fallback keeps a safety net).
        """
        llm = codegen_llm_call if self.planner.llm_call is default_llm_call \
            else self.planner.llm_call
        manifest = dict(self._manifest or {})

        def _fallback_invocation(child_index: int) -> dict:
            modality = (getattr(profile, "modality", "") if profile else "") or ""
            task_type = (getattr(profile, "task_type", "") if profile else "") or ""
            metric = (manifest.get("metric_name")
                      or (getattr(profile, "metric_name", "") if profile else "")
                      or "")
            compat = self.registry.compatible(modality, task_type, metric)
            if not compat:
                return {}
            spec = None
            if getattr(profile, "string_target", False):
                # v2.5.6: string targets (high-cardinality non-numeric
                # output with a copy-source column) use the deterministic
                # string-lookup capability as the platform fallback; the
                # tabular classifiers cannot express millions of labels.
                _sl = self.registry.get("text.string_lookup.v1")
                if _sl is not None and not _sl.broken:
                    spec = _sl
            if spec is None:
                spec = compat[0]
            inv = self.compiler.normalize(MethodInvocationV1(
                method_id=spec.method_id,
                hypothesis="Child %d: default %s on %s (axis=%s)"
                           % (child_index, spec.method_id,
                              grant.get("selected_branch_id"),
                              grant.get("mutation_axis"))))
            inv = clamp_invocation_runnability(
                inv, getattr(self, "resource", None) or {},
                self.registry)
            return inv.to_dict()

        def proposer(child_index: int, g: dict, evidence: str) -> dict:
            deterministic = {
                "hypothesis": "Child %d on %s (axis=%s)"
                              % (child_index, g.get("selected_branch_id"),
                                 g.get("mutation_axis")),
                "mutation_axis": g.get("mutation_axis"),
                "param_overrides": {"child_index": int(child_index)},
                "invocation": _fallback_invocation(child_index),
            }
            data = None
            try:
                response = llm(self._child_proposal_prompt(
                    child_index, g, evidence or "", profile, manifest))
                data = _extract_json(response)
            except Exception:  # noqa: BLE001 - fall back to deterministic
                data = None
            if not (isinstance(data, dict) and data.get("hypothesis")):
                return deterministic
            inv = None
            try:
                mid = str(data.get("method_id") or "").strip()
                if not mid and str(data.get("capability_gap") or "").strip():
                    # HERA proves the registry cannot express its idea:
                    # ONE synthesis action may create the adapter.
                    spec = self._try_synthesize(
                        profile, str(data.get("capability_gap"))[:400])
                    if spec is not None:
                        mid = spec.method_id
                _rr_raw = (data.get("resource_request")
                           if isinstance(data.get("resource_request"), dict)
                           else {})
                _rr = {}
                try:
                    _mtr = int(_rr_raw.get("max_train_rows") or 0)
                except (TypeError, ValueError):
                    _mtr = 0
                if _mtr > 0:
                    _cap = int((getattr(self, "resource", None) or {}).get(
                        "train_rows_cap") or 0)
                    _rr["max_train_rows"] = (
                        _mtr if _cap <= 0 else min(_mtr, _cap))
                candidate = MethodInvocationV1(
                    method_id=mid,
                    params=(data.get("params")
                            if isinstance(data.get("params"), dict) else {}),
                    preprocessing=(data.get("preprocessing")
                                   if isinstance(data.get("preprocessing"),
                                                 list) else []),
                    validation=str(data.get("validation")
                                   or "stratified_kfold"),
                    hypothesis=str(data["hypothesis"])[:400],
                    resource_request=_rr,
                )
                inv = self.compiler.normalize(candidate)
                ok, reason = self.compiler.validate(inv, profile, manifest)
                if not ok:
                    self._log("proposer invocation rejected (%s); fallback"
                              % reason)
                    # v2.3.1 partial fallback: keep HERA's method/params/
                    # hypothesis; sanitize only schema-invalid preproc/
                    # validation fields so a synonym slip does not discard
                    # the whole research choice.
                    inv = self._sanitize_invocation(inv, profile, manifest)
                    if inv is not None:
                        ok2, reason2 = self.compiler.validate(
                            inv, profile, manifest)
                        if ok2:
                            self._log(
                                "proposer sanitized: kept method=%s "
                                "params=%s preproc=%s validation=%s"
                                % (inv.method_id,
                                   sorted((inv.params or {}).keys()),
                                   list(inv.preprocessing or []),
                                   inv.validation))
                            inv = self.compiler.normalize(inv)
                        else:
                            self._log(
                                "proposer sanitized still invalid (%s); "
                                "full fallback" % reason2)
                            inv = MethodInvocationV1.from_dict(
                                _fallback_invocation(child_index))
                    else:
                        inv = MethodInvocationV1.from_dict(
                            _fallback_invocation(child_index))
                    if not inv.method_id:
                        spec = self._try_synthesize(
                            profile, "no compatible capability: " + reason)
                        if spec is not None:
                            inv = self.compiler.normalize(
                                MethodInvocationV1(
                                    method_id=spec.method_id,
                                    hypothesis=str(data["hypothesis"])[:400]))
            except Exception as e:  # noqa: BLE001 - fall back to deterministic
                self._log("proposer invocation error: %s; fallback" % e)
                inv = None
            if inv is None:
                return deterministic
            inv = clamp_invocation_runnability(
                inv, getattr(self, "resource", None) or {},
                self.registry)
            return {
                "hypothesis": (inv.hypothesis
                               or "Child %d on %s"
                                  % (child_index,
                                     g.get("selected_branch_id")))[:400],
                "mutation_axis": g.get("mutation_axis"),
                "param_overrides": {"child_index": int(child_index)},
                "invocation": inv.to_dict(),
            }
        return proposer

    def _sanitize_invocation(self, inv, profile=None, manifest=None):
        """Partial fallback (v2.3.1): keep HERA's method + params + hypothesis;
        keep only schema-valid preprocessing tokens (platform family defaults
        fill an empty list); validation falls back to stratified_kfold.
        Returns None when the method itself is unknown/broken."""
        if inv is None or not inv.method_id:
            return None
        spec = self.registry.get(inv.method_id)
        if spec is None or spec.broken:
            return None
        offered = set(spec.preprocessing_options or [])
        pre = [p for p in (inv.preprocessing or []) if p in offered]
        val = inv.validation
        if val not in (spec.validation_schemes or []):
            val = "stratified_kfold"
        return self.compiler.normalize(MethodInvocationV1(
            method_id=inv.method_id,
            params=dict(inv.params or {}),
            preprocessing=pre,
            validation=val,
            hypothesis=inv.hypothesis,
            resource_request=dict(inv.resource_request or {}),
        ))

    def _child_proposal_prompt(self, child_index: int, grant: dict,
                               evidence: str, profile=None,
                               manifest=None) -> str:
        grant_view = {k: grant.get(k) for k in
                      ("grant_id", "selected_branch_id", "mutation_axis",
                       "research_intent", "stage", "trial_budget",
                       "competition", "task_prompt")
                      if grant.get(k)}
        contract = {}
        if profile is not None:
            contract = {
                "modality": getattr(profile, "modality", ""),
                "task_type": getattr(profile, "task_type", ""),
                "train_rows": getattr(profile, "train_rows", 0),
                "test_rows": getattr(profile, "test_rows", 0),
                "n_classes": getattr(profile, "n_classes", 0),
                "feature_dim": getattr(profile, "feature_dim", 0),
                "image_width": getattr(profile, "image_width", 0),
                "image_height": getattr(profile, "image_height", 0),
                "metric_name": getattr(profile, "metric_name", ""),
                "metric_direction": getattr(profile, "metric_direction", ""),
                "string_target": bool(
                    getattr(profile, "string_target", False)),
                "string_source_column": (
                    getattr(profile, "string_source_column", "") or ""),
                "train_rows_cap": int(
                    (getattr(self, "resource", None) or {}).get(
                        "train_rows_cap") or 0),
            }
        if manifest:
            contract["metric_name"] = manifest.get("metric_name") \
                or contract.get("metric_name")
            contract["cached_weights"] = len(
                manifest.get("pretrained_available") or [])
        modality = (getattr(profile, "modality", "") if profile else "") or ""
        task_type = (getattr(profile, "task_type", "") if profile else "") or ""
        metric = (manifest or {}).get("metric_name") or (
            getattr(profile, "metric_name", "") if profile else "") or ""
        return (
            "ROLE: YOU ARE A WORLD-CLASS EXPERIMENTAL RESEARCHER.\n"
            "You are the agent-side program scientist inside a frozen "
            "research grant. Children are SEQUENTIAL experiments: child %d "
            "must build on the verified outcome of the previous children. "
            "You choose a METHOD + PARAMETERS from the CAPABILITY "
            "REGISTRY; the platform compiles your choice deterministically "
            "(you never write code).\n\n"
            % int(child_index)
            + "GRANT:\n" + json.dumps(grant_view, ensure_ascii=False)[:800] + "\n\n"
            + "DATASET CONTRACT:\n" + json.dumps(
                {k: v for k, v in contract.items() if v},
                ensure_ascii=False)[:900] + "\n\n"
            + "CAPABILITY REGISTRY (compatible methods, choose method_id "
              "from these ONLY):\n"
            + self.registry.prompt_summary(modality, task_type, metric,
                                           max_chars=2400) + "\n\n"
            + "GRANT FEEDBACK (verified outcomes so far):\n"
            + (evidence or "(none)")[:1500] + "\n\n"
            + 'Return ONLY a JSON object of the form '
              '{"method_id": "...", "params": {...}, '
              '"preprocessing": [...], "validation": "...", '
              '"hypothesis": "...", "resource_request": '
              '{"max_train_rows": 20000}}.\n'
            + "resource_request is OPTIONAL; set max_train_rows only to "
              "train on a SMALLER subset than the platform default cap "
              "(DATASET CONTRACT train_rows_cap). Never exceed the cap; "
              "omit to use the platform default.\n"
            + "params keys MUST exist in the chosen method's parameter "
              "schema; preprocessing entries MUST be from its offered "
              "options; validation MUST be from its schemes. Keep the "
              "hypothesis concrete, falsifiable and tied to the feedback. "
              "If NO compatible capability can express your idea, return "
              '{"capability_gap": "why", "hypothesis": "..."} instead.\n'
        )

    # ---- Phase C: capability synthesis (bounded, run-local) ----
    def _try_synthesize(self, profile, gap_reason: str):
        """LLM writes ONE adapter template, registered run-local ephemeral.

        Budget: MAX_SYNTHESIS_ACTIONS (env, default 2) per task, persisted
        under <state_dir>/capabilities/synthesis_usage.json so a restart
        cannot mint fresh actions. The adapter is validated only by trial:
        a failed trial marks the capability broken (PACT).
        """
        budget = int(os.environ.get("MAX_SYNTHESIS_ACTIONS", "2") or 0)
        usage = load_synthesis_usage(self.state_dir)
        if budget <= 0 or int(usage.get("used", 0)) >= budget:
            self._log("synthesis budget exhausted (%d/%d)"
                      % (usage.get("used", 0), budget))
            return None
        llm = codegen_llm_call if self.planner.llm_call is default_llm_call \
            else self.planner.llm_call
        try:
            response = llm(self._synthesis_prompt(profile, gap_reason))
            data = _extract_json(response)
            if not (isinstance(data, dict) and data.get("method_id")):
                raise ValueError("synthesis response missing method_id")
            source = str(data.get("source_code") or "")
            if "def build_model" not in source:
                raise ValueError("synthesis source_code must define build_model")
            mid = "ephemeral." + re.sub(r"[^a-zA-Z0-9_.-]", "_",
                                        str(data["method_id"]))
            spec = MethodSpec(
                method_id=mid,
                family=str(data.get("family") or "ephemeral"),
                supported_modalities=list(data.get("supported_modalities")
                                          or ["tabular"]),
                supported_tasks=list(data.get("supported_tasks")
                                     or ["classification"]),
                metric_outputs=dict(data.get("metric_outputs")
                                    or {"accuracy": "class"}),
                parameter_schema=dict(data.get("parameter_schema") or {}),
                preprocessing_options=list(data.get("preprocessing_options")
                                           or ["missing_value_impute"]),
                validation_schemes=list(data.get("validation_schemes")
                                        or ["stratified_kfold",
                                            "single_holdout"]),
                renderer="ephemeral_sklearn",
                resource_model="ephemeral_sklearn_v1",
                gpu=False,
                description=str(data.get("description")
                                or gap_reason or "")[:200],
                ephemeral=True,
                source_code=source,
                template_hash="sha256:" + hashlib.sha256(
                    source.encode("utf-8")).hexdigest(),
                broken=False,
            )
            self.registry.register_ephemeral(spec)
            usage["used"] = int(usage.get("used", 0)) + 1
            usage.setdefault("actions", []).append(mid)
            save_synthesis_usage(self.state_dir, usage)
            self._log("SYNTHESIS registered method_id=%s source_len=%d "
                      "used=%d/%d" % (mid, len(source), usage["used"],
                                      budget))
            return spec
        except Exception as e:  # noqa: BLE001 - budget stays intact
            self._log("SYNTHESIS FAILED: %s" % str(e)[:300])
            return None

    @staticmethod
    def _synthesis_prompt(profile, gap_reason: str) -> str:
        contract = {}
        if profile is not None:
            contract = {
                "modality": getattr(profile, "modality", ""),
                "task_type": getattr(profile, "task_type", ""),
                "train_rows": getattr(profile, "train_rows", 0),
                "n_classes": getattr(profile, "n_classes", 0),
                "feature_dim": getattr(profile, "feature_dim", 0),
                "metric_name": getattr(profile, "metric_name", ""),
            }
        return (
            "ROLE: CAPABILITY SYNTHESIS ENGINEER.\n"
            "The research agent proved the capability registry cannot "
            "express its idea. Write ONE reusable sklearn-compatible "
            "adapter template. It is compiled into a deterministic harness "
            "and reused by later children with parameter-only changes.\n\n"
            + "CAPABILITY GAP:\n" + (gap_reason or "(none)")[:600] + "\n\n"
            + "DATASET CONTRACT:\n"
            + json.dumps({k: v for k, v in contract.items() if v},
                         ensure_ascii=False)[:800] + "\n\n"
            + 'Return ONLY JSON: {"method_id": "...", "description": "...", '
              '"supported_modalities": ["tabular"], "supported_tasks": '
              '["classification"], "metric_outputs": {"accuracy": "class"}, '
              '"parameter_schema": {"p": {"type": "float", "min": 0.0, '
              '"max": 1.0, "default": 0.5}}, "preprocessing_options": '
              '["missing_value_impute"], "validation_schemes": '
              '["stratified_kfold", "single_holdout"], "source_code": '
              '"def build_model(params, seed):\\n    ...return estimator"}\n'
            + "CONSTRAINTS: source_code must define build_model(params, "
              "seed) returning a sklearn-compatible estimator with "
              "fit/predict/predict_proba (predict_proba optional for "
              "class metrics that need probabilities); use only "
              "numpy/pandas/sklearn; every parameter_schema key must have "
              "a default and be used by build_model; keep it short and "
              "robust to missing values.\n"
        )
    # ---- grant-internal FeedbackView loop ----
    def _child_evidence(self, agent: ProgramAgentClient, profile) -> str:
        """Round evidence + grant-internal child feedback for the next child."""
        parts = []
        # rc4: grant-internal feedback FIRST - child N+1 must build on
        # child N before anything else (round evidence is context).
        feedback = agent.feedback_view()
        if feedback:
            parts.append("GRANT FEEDBACK (prior children in THIS grant):\n"
                         + feedback)
        round_evidence = (self._evidence_summary(profile) or "")[:2500]
        if round_evidence:
            parts.append("ROUND EVIDENCE (previous rounds):\n"
                         + round_evidence)
        return "\n\n".join(parts)

    def _absorb_receipt(self, r: TrialReceipt, cert_before=None) -> None:
        """Shared receipt bookkeeping for inline and daemon modes.

        cert_before: certified-best metric as of BEFORE this trial was
        promoted (snapshot taken by the caller ahead of supervise/outcome).
        When omitted (legacy direct calls in tests), the defense falls back
        to the current promotion record only if the record does NOT already
        include this receipt."""
        self._log("receipt: verdict=%s metric=%s rc=%s"
                  % (r.verdict, r.metric, r.returncode))
        self.total_trials += 1
        if r.verdict == "failure":
            self.failed_trials += 1
        # Only verified trials (rc==0) move the best pointer or count
        # stagnation: failed/rejected trials carry a fallback baseline
        # metric that must never masquerade as progress. In-sample OOF
        # trials are rc==0 and are allowed to move the best pointer.
        if r.metric is not None and r.returncode == 0:
            # v2.2.1-rc3 defense: never let a receipt below the CERTIFIED
            # incumbent overwrite the best code asset, even if the
            # in-process best_metric was lost (restart) or the host verdict
            # was computed against a stale best. Only a verdict-level
            # success/improvement or a metric strictly better than the
            # certified incumbent may update the incumbent asset.
            cert_metric = cert_before
            if cert_metric is None:
                # direct call (tests / legacy): if the current receipt was
                # already promoted, the record's incumbent_metric is the
                # certified best BEFORE it; otherwise certified_best_metric.
                pm = getattr(self, "promotion", None)
                if pm is not None:
                    promo = pm.certified_best()
                    if promo.certified_best_trial_id in (
                            str(getattr(r, "spec_id", "")),
                            str(getattr(r, "receipt_id", ""))):
                        cert_metric = promo.incumbent_metric
                    else:
                        cert_metric = promo.certified_best_metric
            verdict = str(getattr(r, "verdict", "") or "")
            if self.best_metric is None and cert_metric is not None \
                    and not self._is_better(r.metric, cert_metric):
                self._log("receipt below certified best %s (verdict=%s): "
                          "not accepted as best" % (cert_metric, verdict))
                self.stagnation_count += 1
                return
            if self.best_metric is None or self._is_better(r.metric, self.best_metric):
                self.best_metric = r.metric
                self.best_receipt_id = r.receipt_id
                self.stagnation_count = 0
                self._log("NEW BEST: %s" % self.best_metric)
                self._save_incumbent(r)
            else:
                self.stagnation_count += 1

    # ---- round-continuity: incumbent best-code asset ----
    # The platform NEVER chooses the method for HERA. It only persists the
    # latest VERIFIED best code as a first-class asset so the next round can
    # build on success (extend/surgically modify) instead of rewriting from
    # baseline. HERA/the implementer LLM decide how to use the asset.
    def _incumbent_json(self) -> Path:
        return self.state_dir / "incumbent_best.json"

    def _incumbent_dir(self) -> Path:
        return self.state_dir / "incumbent"

    def _load_incumbent(self) -> Optional[dict]:
        """Read the latest verified incumbent asset (code path + meta)."""
        path = self._incumbent_json()
        try:
            if path.is_file():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("code_path"):
                    return data
        except (OSError, ValueError):
            pass
        return None

    def _save_incumbent(self, r: TrialReceipt) -> bool:
        """Persist the verified best code so the next round/grant can inherit
        it as an asset (HERA still decides what to do with it). Non-fatal:
        continuity is an optimization, never a correctness gate.
        """
        try:
            src = self.bus.ws_code / ("trial_" + str(r.spec_id) + ".py")
            if not src.is_file():
                return False
            self._incumbent_dir().mkdir(parents=True, exist_ok=True)
            code_path = self._incumbent_dir() / (
                "best_code_%02d.py" % int(self.round_num or 0))
            shutil.copy2(str(src), str(code_path))
            shutil.copy2(str(src), self._incumbent_dir() / "best_code.py")
            digest = hashlib.sha256(code_path.read_bytes()).hexdigest()
            data = {
                "schema_version": "v2_incumbent_v1",
                "competition": self.competition,
                "round_num": int(self.round_num or 0),
                "receipt_id": getattr(r, "receipt_id", "") or "",
                "spec_id": getattr(r, "spec_id", "") or "",
                "grant_id": getattr(r, "grant_id", "") or "",
                "branch_id": getattr(r, "branch_id", "") or "",
                "metric": getattr(r, "metric", None),
                "metric_name": getattr(r, "metric_name", "") or "",
                "code_hash": "sha256:" + digest,
                "code_path": str(code_path),
                "created_at": now_iso(),
            }
            tmp = self._incumbent_json().with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            os.replace(str(tmp), str(self._incumbent_json()))
            self._log("INCUMBENT SAVED: round=%d metric=%s code=%s"
                      % (self.round_num, r.metric, code_path))
            return True
        except (OSError, ValueError, AttributeError, TypeError) as e:  # noqa: BLE001 - best-effort
            self._log("incumbent save skipped: %s" % e)
            return False

    def _run_grant_inline(self, agent: ProgramAgentClient, grant, profile,
                          plan, ticket) -> list:
        """Single-process FeedbackView loop: propose -> supervise -> absorb."""
        receipts = []
        best_child = None
        children_budget = int(grant.trial_budget or ticket.trial_budget or 1)
        for child_index in range(1, children_budget + 1):
            evidence = self._child_evidence(agent, profile)
            proposal = agent.propose_next(child_index, evidence)
            self._log("propose: child %d/%d proposal=%s"
                      % (child_index, children_budget, proposal.proposal_id))
            cert_before = self.promotion.certified_best().certified_best_metric
            receipt = self.host.supervise_once(
                grant.to_dict(), profile, plan, best_child)
            if receipt is None:
                self._log("round %d: child %d not executed (no claim)"
                          % (self.round_num, child_index))
                break
            receipts.append(receipt)
            self._absorb_receipt(receipt, cert_before)
            best_child = self._update_best_child(
                best_child, receipt.metric)
        return receipts

    def _run_grant_daemon(self, agent: ProgramAgentClient, grant, profile,
                          plan, ticket) -> list:
        """True two-process loop: independent resident host daemon.

        The daemon is launched before the agent proposes; the agent writes
        ONE proposal at a time and waits for the verified outcome, so child
        N+1 is generated from child N's outcome through the File-as-Bus.
        """
        daemon_script = HERE / "v2_host_daemon.py"
        grant_dict = grant.to_dict()
        grant_id = grant_dict.get("grant_id", "grant")
        stop_file = self.bus.host_control_state / (
            "stop_" + safe_artifact_name(grant_id) + ".json")
        if stop_file.exists():
            stop_file.unlink()
        daemon_log = self.state_dir / (
            "host_daemon_%s.log" % safe_artifact_name(grant_id))

        cmd = [self.daemon_python, str(daemon_script),
               "--state-dir", str(self.state_dir),
               "--data-dir", str(self.data_dir),
               "--work-dir", str(self.work_dir),
               "--competition", self.competition,
               "--task-prompt", self.task_prompt,
               "--round-timeout", str(self.round_timeout),
               "--poll-interval", str(self.daemon_poll_interval),
               "--idle-exit-seconds", str(self.daemon_idle_exit_seconds),
               "--max-children",
               str(int(grant.trial_budget or ticket.trial_budget or 1)),
               "--grant-id", str(grant_id)]
        if self.sample_path:
            cmd += ["--sample-path", self.sample_path]
        if self.executor.exec_image:
            cmd += ["--exec-image", self.executor.exec_image]
        if self.executor.exec_python:
            cmd += ["--exec-python", self.executor.exec_python]
        if self.executor.torch_cache:
            cmd += ["--torch-cache", self.executor.torch_cache]

        log_fh = open(daemon_log, "w", encoding="utf-8")
        try:
            proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT,
                                    env=os.environ.copy())
        except Exception as e:  # noqa: BLE001 - fall back to inline mode
            log_fh.close()
            self._log("host daemon start failed (%s); falling back to "
                      "inline host" % e)
            return self._run_grant_inline(agent, grant, profile, plan, ticket)
        self._log("host daemon started: pid=%d log=%s" % (proc.pid, daemon_log))

        receipts = []
        best_child = None
        children_budget = int(grant.trial_budget or ticket.trial_budget or 1)
        try:
            for child_index in range(1, children_budget + 1):
                evidence = self._child_evidence(agent, profile)
                proposal = agent.propose_next(child_index, evidence)
                self._log("propose: child %d/%d proposal=%s"
                          % (child_index, children_budget,
                             proposal.proposal_id))
                cert_before = self.promotion.certified_best().certified_best_metric
                outcome = self._wait_for_outcome(
                    proposal, proc, timeout=self.round_timeout + 300)
                if outcome is None:
                    tail = _daemon_log_tail(daemon_log, n=25)
                    self._log("round %d: child %d outcome not delivered "
                              "(daemon rc=%s)%s"
                              % (self.round_num, child_index, proc.poll(),
                                 ("\\n--- daemon log tail ---\\n" + tail)
                                  if tail else ""))
                    break
                receipt = self._receipt_from_outcome(outcome)
                if receipt is None:
                    break
                receipts.append(receipt)
                self._absorb_receipt(receipt, cert_before)
                best_child = self._update_best_child(
                    best_child, receipt.metric)
        finally:
            stop_file.write_text(json.dumps(
                {"grant_id": grant_id, "stopped_at": now_iso()},
                ensure_ascii=False), encoding="utf-8")
            try:
                proc.terminate()
                proc.wait(timeout=15)
            except Exception:  # noqa: BLE001 - process already gone
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
            log_fh.close()
        return receipts

    def _wait_for_outcome(self, proposal, proc: subprocess.Popen,
                          timeout: int):
        """Poll outcomes_visible/ for this proposal's verified outcome."""
        pid = proposal.proposal_id
        path = self.bus.outcomes_visible / (
            "outcome_" + safe_artifact_name(pid) + ".json")
        deadline = time.time() + max(30, int(timeout))

        def _read():
            if path.is_file():
                try:
                    return json.loads(path.read_text(encoding="utf-8"))
                except ValueError:
                    pass
            return None

        while time.time() < deadline:
            outcome = _read()
            if outcome is not None:
                return outcome
            if proc.poll() is not None:
                outcome = _read()  # late outcome written just before exit
                return outcome
            time.sleep(3)
        return _read()

    @staticmethod
    def _receipt_from_outcome(outcome: dict) -> Optional[TrialReceipt]:
        verdict = outcome.get("verdict") or "failure"
        return TrialReceipt(
            receipt_id="daemon_" + str(outcome.get("trial_id")
                                       or outcome.get("proposal_id", "")),
            spec_id=outcome.get("trial_id", ""),
            competition=outcome.get("competition", ""),
            round_num=1,
            # NB: `x or -1` would map returncode=0 to -1; use explicit None check
            returncode=int(outcome.get("returncode")
                            if outcome.get("returncode") is not None else -1),
            stdout="",
            stderr=(outcome.get("stderr") or "")[-2000:],
            metric=outcome.get("metric"),
            metric_name=outcome.get("metric_name", "accuracy"),
            verdict=verdict,
            evidence=outcome.get("evidence", ""),
            submission_exists=bool(outcome.get("submission_exists")),
            submission_path=outcome.get("submission_path", ""),
            code_hash=outcome.get("code_hash", ""),
            verified=verdict != "failure",
            failure_reason=outcome.get("failure_reason", ""),
        )

    def _evidence_summary(self, profile) -> str:
        best = self._ledger_best_metric()
        stagnation = self._ledger_stagnation(window=3)
        recent = self.ledger.recent_rounds(self.competition, n=3)
        strategies = self.memory.relevant_strategies(self.task_prompt, top_k=3)
        knowledge = self.memory.cross_task_knowledge(self.competition, top_k=5)
        incumbent = self._load_incumbent()

        lines = []
        stage = getattr(self, "stage", None)
        guard = getattr(self, "guard", None)
        if stage is not None and guard is not None:
            lines.append("Stage: %s (grants %d/%d, trials %d/%d, wall %ds "
                         "left)" % (stage.stage, guard.grants_used,
                                    guard.max_grants, guard.trials_used,
                                    guard.max_total_trials,
                                    int(guard.remaining())))
            lines.append("Stage guidance: %s" % stage.guidance_summary())
        if getattr(self, "resource", None):
            lines.append("Resource profile (derived, competition-agnostic): "
                         "budget=%ds folds<=%d t_est=%ds scale=%s policy=%s"
                         % (self.resource.get("max_budget_seconds"),
                            self.resource.get("max_folds"),
                            self.resource.get("t_est_seconds"),
                            self.resource.get("model_scale_ceiling"),
                            self.resource.get("pretrained_policy")))
        if incumbent:
            lines.append(
                "Incumbent best code (verified asset from round %s, "
                "metric=%s): %s"
                % (incumbent.get("round_num"), incumbent.get("metric"),
                   incumbent.get("code_path")))
            if incumbent.get("branch_id"):
                lines.append("  incumbent branch=%s code_hash=%s"
                             % (incumbent.get("branch_id"),
                                incumbent.get("code_hash") or ""))
        if best is not None:
            lines.append("Best verified metric so far: %s" % best)
        if stagnation > 0:
            lines.append("Stagnation: last %d rounds without improvement" % stagnation)
        if recent:
            lines.append("Recent rounds:")
            for r in recent:
                lines.append("  R%s %s metric=%s verdict=%s"
                             % (r.get("round_num"), r.get("approach_type"),
                                r.get("metric"), r.get("verdict")))
        if strategies:
            lines.append("Relevant strategies: %s"
                         % json.dumps(strategies, ensure_ascii=False))
        if self.last_interpretation is not None:
            interp = self.last_interpretation
            lines.append("Last interpretation: verdict=%s delta=%s stop=%s"
                         % (interp.verdict, interp.delta, interp.stop_decision))
            if interp.next_direction:
                lines.append("  next_direction: %s" % interp.next_direction[:200])
            if interp.causal_attribution and interp.causal_attribution != "Unknown":
                lines.append("  causal: %s" % interp.causal_attribution[:200])
            if interp.blind_spots:
                lines.append("  blind_spots: %s" % interp.blind_spots[:200])
        facts = self._platform_facts()
        if facts:
            lines.append(facts)
        return "\n".join(lines)

    # ---- certified publish layer ----
    def _publish_certified(self):
        try:
            path = self.publisher.publish_certified()
            self._log("published certified submission: %s" % path)
        except Exception as e:  # noqa: BLE001 - publish is best-effort
            self._log("publish skipped: %s" % e)

    def _official_grade(self, sub_path) -> dict:
        """Best-effort official re-eval of the final submission with the
        LOCAL mlebench package (the same grader the organizer runs).

        Generic for every MLE-Bench task: it is driven only by the
        competition id already in the manifest - there is no per-task or
        per-competition branch here. If mlebench is not importable in the
        control environment (or the dataset is not prepared), it raises and
        the caller records the skip; a failed re-eval NEVER blocks the run.
        """
        import importlib.util
        if importlib.util.find_spec("mlebench") is None:
            raise RuntimeError("mlebench not importable in control env")
        from mlebench.grade import grade_csv
        from mlebench.registry import registry
        cid = self.competition or ""
        comp = registry.get_competition(cid)
        report = grade_csv(Path(sub_path), comp)
        out = report.to_dict()
        out.pop("created_at", None)
        return out

    def _finalize(self) -> dict:
        elapsed = time.time() - self.start_time
        sub = self.submission_dir / "submission.csv"
        promo = self.promotion.certified_best()
        result = {
            "status": "completed",
            "version": "v2.2.1-rc4",
            "competition": self.competition,
            "rounds_completed": self.round_num,
            "best_metric": self.best_metric,
            "best_receipt_id": self.best_receipt_id,
            "total_trials": self.total_trials,
            "failed_trials": self.failed_trials,
            "certified_best_metric": promo.certified_best_metric,
            "certified_best_trial_id": promo.certified_best_trial_id,
            "total_time_seconds": round(elapsed, 2),
            "stagnation_count": self.stagnation_count,
            "stage": self.stage.stage if self.stage else "",
            "stage_history": (self.stage.history
                              if self.stage is not None else []),
            "grants_completed": self.guard.grants_used,
            "grants_budget": self.guard.max_grants,
            "trials_committed": self.guard.trials_used,
            "trials_budget": self.guard.max_total_trials,
            "resource_profile": self.resource,
            "f0_calibration": self.f0_calibration,
            "submission_exists": sub.is_file(),
            "submission_path": str(sub),
            "official_grade": self._finalize_official_grade(sub),
            "strategies_discovered": len(self.memory.strategy_pool.get_all()),
            "file_bus_zones": self.bus.tree_report(),
            "incumbent_code_path": str(
                (self._load_incumbent() or {}).get("code_path") or ""),
        }
        report = self.state_dir / "run_report.json"
        report.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str),
                          encoding="utf-8")
        self._log("RESEARCH CYCLE COMPLETE: rounds=%d best=%s certified=%s submission=%s"
                  % (self.round_num, self.best_metric,
                     promo.certified_best_metric, result["submission_exists"]))
        return result

    def _finalize_official_grade(self, sub) -> dict:
        """Fail-open wrapper: official grade runs only when the environment
        has mlebench + the prepared dataset; otherwise it is recorded as
        skipped. The run itself is never blocked by grading."""
        if not sub.is_file():
            return {"status": "skipped", "reason": "no submission"}
        try:
            grade = self._official_grade(sub)
            self._log("official grade: score=%s medal_any=%s valid=%s"
                      % (grade.get("score"), grade.get("any_medal"),
                         grade.get("valid_submission")))
            return {"status": "ok", **grade}
        except Exception as e:  # noqa: BLE001 - grading must not kill the loop
            self._log("official grade skipped (non-fatal): %s" % str(e)[:200])
            return {"status": "skipped", "reason": str(e)[:200]}

    def _build_manifest(self, profile) -> dict:
        import csv as _csv
        layout = self.analyzer.layout
        manifest = layout.manifest(
            train_rows=profile.train_rows,
            test_rows=profile.test_rows,
            target_column=profile.target_column,
            task_type=profile.task_type)
        header = []
        sample = layout.sample_submission_path
        if sample is not None and sample.is_file():
            try:
                with open(sample, "r", encoding="utf-8",
                          errors="replace", newline="") as fh:
                    header = list(next(_csv.reader(fh), []) or [])
            except OSError:
                pass
        manifest["sample_submission_header"] = header
        # v2.3.2: content evidence rides the manifest so the compiler can
        # render text (TF-IDF) / timeseries (lag) harnesses even when the
        # in-memory profile is not available (restart/daemon paths).
        manifest["modality"] = getattr(profile, "modality", "") or ""
        # v2.3.8: structural evidence (RLE mask target / bbox columns /
        # multi-row targets / audio file count) rides the manifest so the
        # compiler templates can render for mask/detection/audio tasks.
        manifest["mask_target"] = getattr(profile, "mask_target", "") or ""
        manifest["bbox_columns"] = list(
            getattr(profile, "bbox_columns", None) or [])
        manifest["multi_row_target"] = bool(
            getattr(profile, "multi_row_target", False))
        manifest["audio_file_count"] = int(
            getattr(profile, "audio_file_count", 0) or 0)
        manifest["text_columns"] = list(
            getattr(profile, "text_columns", None) or [])
        manifest["time_column"] = getattr(profile, "time_column", "") or ""
        manifest["string_target"] = bool(
            getattr(profile, "string_target", False))
        manifest["string_source_column"] = (
            getattr(profile, "string_source_column", "") or "")
        spec = get_metric_spec(self.competition)
        manifest["metric_name"] = spec["metric_name"]
        manifest["metric_direction"] = spec["metric_direction"]
        manifest["metric_alignment"] = spec["metric_alignment"]
        manifest["metric_label"] = spec["metric_label"]
        manifest["metric_params"] = spec["metric_params"]
        manifest["metric_min_delta"] = float(
            spec.get("min_delta", DEFAULT_MIN_DELTA))
        return manifest

    def _log(self, msg: str) -> None:
        print("[%s] %s" % (time.strftime("%H:%M:%S", time.gmtime()), msg), flush=True)

    def _is_better(self, candidate, incumbent) -> bool:
        """Direction-aware improvement test (metric_spec + per-metric delta).

        v2.3.6: the threshold comes from the metric family (METRIC_MIN_DELTA),
        not a global 0.01 - bounded score metrics must accept improvements of
        1e-4 (e.g. AUC 0.9997 > 0.9972), error metrics 1e-3.
        """
        if incumbent is None:
            return True
        direction = self.metric_spec.get("metric_direction", "higher_is_better")
        min_delta = float(self.metric_spec.get("min_delta",
                                               DEFAULT_MIN_DELTA))
        if direction == "lower_is_better":
            return candidate < incumbent - min_delta
        return candidate > incumbent + min_delta

    def _update_best_child(self, best_child, metric) -> Optional[float]:
        """Direction-aware running best for child trials inside one grant
        (lower-is-better metrics keep the MINIMUM, never max())."""
        if metric is None:
            return best_child
        if best_child is None or self._is_better(metric, best_child):
            return metric
        return best_child


def clamp_invocation_runnability(inv, resource=None, registry=None):
    """Platform runnability clamps (data-driven, never research choices).

    folds <= resource max_folds: on large-row datasets the derived
    resource profile caps fold counts (max_folds=2 above 10k rows) so a
    trial fits inside its budget; the LLM may choose fewer folds but
    never more. Applied to EVERY compiled invocation (LLM-chosen and
    deterministic fallback alike)."""
    if inv is None or not getattr(inv, "method_id", ""):
        return inv
    params = dict(inv.params or {})
    max_folds = 0
    if resource:
        try:
            max_folds = int(resource.get("max_folds") or 0)
        except (TypeError, ValueError):
            max_folds = 0
    if max_folds > 0 and registry is not None:
        spec = registry.get(inv.method_id)
        default_folds = 0
        if spec is not None and (spec.parameter_schema or {}).get("folds"):
            try:
                default_folds = int(
                    (spec.parameter_schema["folds"].get("default") or 0))
            except (TypeError, ValueError):
                default_folds = 0
        cur = 0
        try:
            cur = int(params.get("folds") or default_folds or 0)
        except (TypeError, ValueError):
            cur = 0
        if cur > max_folds:
            params["folds"] = int(max_folds)
            print("[closed-loop] runnability: folds %d -> %d (platform "
                  "max_folds)" % (cur, max_folds), flush=True)
    inv.params = params
    return inv


def _extract_json(response: str) -> Optional[dict]:
    """Best-effort JSON object extraction from an LLM response."""
    import re
    match = re.search(r"\{.*\}", response or "", re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
        return data if isinstance(data, dict) else None
    except (ValueError, TypeError):
        return None


class _StopLoop(Exception):
    """Internal signal: HERA decided to stop."""


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        assert_legacy_l1_mode()
    except GuardError as e:
        print("FATAL: %s" % e, file=sys.stderr)
        return 2

    loop = ClosedLoop(args)
    result = loop.run()
    print("\nRESULT_SUMMARY=" + json.dumps(result, ensure_ascii=False, default=str))
    return 0 if result.get("submission_exists") else 1


if __name__ == "__main__":
    raise SystemExit(main())
