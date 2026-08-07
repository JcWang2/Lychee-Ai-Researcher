"""v2_host_daemon.py - independent resident HostSupervisorService.

Architecture (per design):
  HostSupervisorService starts BEFORE the agent proposes and runs as its own
  process. It polls protocol/outbox/pending_agent, atomically claims each
  proposal, materializes the sealed TrialSpec, executes it in isolation,
  recomputes the metric (trusted, OOF-only), promotes/publishes-certified
  pointers, writes outcomes_visible/ and acks.

  The director (v2_closed_loop.py --host-daemon) freezes the grant, starts
  this daemon, then proposes children ONE AT A TIME, waiting for each
  verified outcome before proposing the next (FeedbackView loop). This
  daemon exits when the frozen grant's budget is consumed, the director's
  stop marker appears, or no proposal arrives for --idle-exit-seconds.

Usage (usually launched by the director, also runnable manually):
  python3 v2_host_daemon.py \
    --state-dir <state> --data-dir <data> --work-dir <code> \
    --competition <task> --task-prompt "<prompt>"
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from data_layout import (sanitize_test_csv,
                        synthesize_train_labels)  # noqa: E402
from metrics_registry import get_metric_spec  # noqa: E402
from hera import Analyzer  # noqa: E402
from pact import (CandidateBundler, Executor, FileBus,  # noqa: E402
                  HostSupervisorService, Implementer, PactLedger,
                  PromotionManager, TrustedEvaluator)
from pact.file_bus import safe_artifact_name  # noqa: E402
from capability_registry import (CapabilityRegistry,  # noqa: E402
                                 load_ephemeral_path)  # noqa: E402
from program_compiler import ProgramCompiler  # noqa: E402
from v2_contracts import ResearchProgramGrant  # noqa: E402
from v2_llm import codegen_llm_call  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(
        description="V2.1 independent resident HostSupervisorService daemon")
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--submission-dir", default="")
    parser.add_argument("--sample-path", default="")
    parser.add_argument("--competition", default="unknown")
    parser.add_argument("--task-prompt", default="")
    parser.add_argument("--exec-image",
                        default=os.environ.get("V2_EXEC_IMAGE", ""))
    parser.add_argument("--exec-python",
                        default=os.environ.get("V2_EXEC_PYTHON", "python3"))
    parser.add_argument("--torch-cache",
                        default=os.environ.get("V2_TORCH_CACHE", ""))
    parser.add_argument("--round-timeout", type=int,
                        default=int(os.environ.get("ROUND_TIMEOUT", "3600")))
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--idle-exit-seconds", type=int, default=600)
    parser.add_argument("--max-children", type=int, default=0)
    parser.add_argument("--grant-wait-seconds", type=int, default=900)
    parser.add_argument("--grant-id", default="",
                        help="Pin this daemon to one frozen grant"
                             "(set by the director); empty = latest frozen.")
    return parser


def _load_manifest(work_dir: Path, executor: Executor) -> dict:
    manifest_path = work_dir / "data_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            executor.set_manifest(manifest)
            return manifest
        except (OSError, ValueError):
            pass
    return {}


def _frozen_by_newest(bus: FileBus) -> list:
    """Frozen grants ordered newest-first (best effort).

    Grant ids are random (uuid4 hex), so alphabetical order says nothing
    about recency. Sort key is the grant's own ISO created_at (authoritative,
    set by the director at freeze time), with file mtime as a tie-breaker
    for same-second freezes (NFS mtime granularity can otherwise tie).
    """
    def _sort_key(p):
        try:
            mtime = p.stat().st_mtime
        except OSError:
            mtime = 0.0
        created = ""
        try:
            d = json.loads(Path(p).read_text(encoding="utf-8"))
            created = str(d.get("created_at") or "")
        except (OSError, ValueError):
            pass
        return (created, mtime)

    entries = []
    for p in sorted(bus.frozen_visible.glob("grant_*.json"),
                    key=_sort_key, reverse=True):
        try:
            d = json.loads(Path(p).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if d:
            entries.append(d)
    return entries

def _wait_for_grant(bus: FileBus, grant_wait_seconds: int,
                    grant_id: str = "") -> dict:
    """Resolve the grant this daemon serves.

    With a pinned --grant-id (director always sets it) we wait for THAT
    specific grant to be visible and frozen/active, so a slow-starting
    daemon can never bind to a newer grant and then reject the director's
    proposals. Without a pin (manual runs, tests) the latest frozen/active
    grant wins; terminal grants are never served.
    """
    deadline = time.time() + max(10, grant_wait_seconds)
    while time.time() < deadline:
        grants = bus.list_frozen()
        if grant_id:
            for g in grants:
                if (g.get("grant_id") == grant_id
                        and g.get("status") in ("frozen", "active")):
                    return g
        else:
            for g in _frozen_by_newest(bus):
                if g.get("status") in ("frozen", "active"):
                    return g
        time.sleep(2)
    return {}


def _purge_stale_pending(bus: FileBus, grant_id: str) -> int:
    """Quarantine proposals that can never be served by this daemon.

    Any proposal already in pending_agent when a fresh daemon starts is an
    orphan of a crashed previous daemon (the director proposes only AFTER
    the daemon is up). Moving them aside heals the bus so a stale
    proposal_grant_mismatch can no longer poison every subsequent round.
    """
    purged = 0
    for p in bus.list_pending():
        if str(p.get("grant_id") or "") != grant_id:
            bus.quarantine_pending(p.get("proposal_id", ""),
                                   reason="stale_grant_at_daemon_start")
            purged += 1
    return purged


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    state_dir = Path(args.state_dir)
    data_dir = Path(args.data_dir)
    work_dir = Path(args.work_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    bus = FileBus(state_dir)
    synthesize_train_labels(data_dir)  # class-dir/flat-prefix train CSV fallback
    sanitize_test_csv(data_dir)  # physical gold isolation (idempotent)
    executor = Executor(work_dir, exec_image=args.exec_image,
                        exec_python=args.exec_python, data_dir=str(data_dir),
                        torch_cache=args.torch_cache)
    manifest = _load_manifest(work_dir, executor)

    grant = _wait_for_grant(bus, args.grant_wait_seconds,
                            grant_id=args.grant_id)
    if not grant:
        print("DAEMON_NO_GRANT grant=%s after %ss; exiting"
              % (args.grant_id or "<latest>", args.grant_wait_seconds),
              flush=True)
        return 1
    purged = _purge_stale_pending(bus, str(grant.get("grant_id") or ""))
    print("DAEMON_GRANT grant=%s purged_stale=%d"
          % (grant.get("grant_id"), purged), flush=True)

    competition = grant.get("competition") or args.competition
    task_prompt = grant.get("task_prompt") or args.task_prompt
    profile = Analyzer(data_dir, task_prompt,
                       sample_path=args.sample_path or "").profile(competition)

    spec = get_metric_spec(competition)
    if manifest.get("metric_name"):
        spec = {
            "metric_name": manifest.get("metric_name") or spec["metric_name"],
            "metric_direction": manifest.get("metric_direction") or spec["metric_direction"],
            "metric_alignment": manifest.get("metric_alignment") or spec["metric_alignment"],
            "metric_label": manifest.get("metric_label") or spec["metric_label"],
            "metric_params": manifest.get("metric_params") or spec["metric_params"],
            "min_delta": float(manifest.get("metric_min_delta")
                               or spec.get("min_delta") or 0.01),
        }

    ledger = PactLedger(state_dir)
    # v2.3: the daemon builds its own registry (persisted ephemeral specs
    # from the state dir) + compiler so compiled invocations render
    # deterministically in this process too (no LLM codegen).
    registry = CapabilityRegistry(
        ephemeral_path=load_ephemeral_path(state_dir))
    compiler = ProgramCompiler(registry)
    host = HostSupervisorService(
        bus=bus, executor=executor,
        bundler=CandidateBundler(bus, work_dir),
        evaluator=TrustedEvaluator(
            metric_name=spec["metric_name"],
            metric_direction=spec["metric_direction"],
            metric_alignment=spec["metric_alignment"],
            metric_label=spec["metric_label"],
            metric_params=spec["metric_params"]),
        promotion=PromotionManager(bus, metric_direction=spec["metric_direction"],
                                   min_delta=float(spec.get("min_delta", 0.01))),
        implementer=Implementer(llm_call_fn=codegen_llm_call),
        compiler=compiler, registry=registry,
        guards=None, ledger=ledger, competition=competition,
        max_budget_seconds=max(1, args.round_timeout),
        data_dir=data_dir, sample_path=args.sample_path or "",
        gold_test_csv=manifest.get("gold_test_csv", "") or "",
        test_csv=manifest.get("test_csv", "") or "",
        state_dir=state_dir,
        metric_min_delta=float(spec.get("min_delta", 0.01)))

    grant_dict = grant
    stop_file = state_dir / "pact_control_host" / "state" / "pact" / "control" \
        / ("stop_" + safe_artifact_name(grant_dict.get("grant_id", "grant")) + ".json")
    max_children = args.max_children or int(grant_dict.get("trial_budget") or 0)
    plan = ResearchProgramGrant.from_dict(grant_dict).plan_obj()

    print("DAEMON_START grant=%s budget=%d mode=%s poll=%.1fs idle=%ds"
          % (grant_dict.get("grant_id"), max_children, executor.exec_mode(),
             args.poll_interval, args.idle_exit_seconds), flush=True)
    try:
        receipts = host.serve(grant_dict, profile, plan,
                              max_children=max_children,
                              poll_interval=args.poll_interval,
                              idle_exit_seconds=args.idle_exit_seconds,
                              stop_file=str(stop_file))
    except Exception as exc:  # noqa: BLE001 - crash must be visible
        import traceback as _tb
        print("DAEMON_CRASH grant=%s err=%r"
              % (grant_dict.get("grant_id"), exc), flush=True)
        _tb.print_exc()
        return 2
    print("DAEMON_DONE receipts=%d" % len(receipts), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
