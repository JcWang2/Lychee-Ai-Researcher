# -*- coding: utf-8 -*-
"""test_v2_l1_transactional.py - Four-layer design conformance tests.

Covers the design diagram:
  outer loop (portfolio/prioritization/frozen grant)
  inner PACT transactional loop (agent propose -> host claim/execute/
  evaluate/promote -> outcomes)
  certified publish layer (ControlledPublisher)
  bottom File-as-Bus (workspace/protocol/pact_control_host zones)
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from hera import MethodPortfolio, Prioritizer  # noqa: E402
from pact import (CandidateBundler, ControlledPublisher, FileBus,  # noqa: E402
                  HostSupervisorService, PromotionManager, TrustedEvaluator)
from v2_contracts import (ResearchPlan, TrialReceipt,  # noqa: E402
                          TrialSpec)

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("[OK] " + name)
    else:
        print("[FAIL] " + name + (" | " + detail if detail else ""))
        FAILURES.append(name)


def _make_loop(tmp):
    bus = FileBus(tmp / "state")
    plan = ResearchPlan(hypothesis="H", method_detail={"model": "xgb"})
    ticket = Prioritizer(llm_call_fn=lambda p: "{}").prioritize(
        None, MethodPortfolio.default_for(None), plan, trial_budget=2)
    grant = Prioritizer(llm_call_fn=lambda p: "{}").freeze_grant(
        "demo", "classification", plan, ticket)
    return bus, plan, grant


def test_file_bus_zones():
    tmp = Path(tempfile.mkdtemp(prefix="v2_l1_"))
    try:
        bus = FileBus(tmp / "state")
        check("workspace zone exists", bus.workspace.is_dir())
        check("protocol zone exists", bus.protocol.is_dir())
        check("pact_control_host zone exists", bus.host_control.is_dir())
        check("pending_agent dir", bus.pending_agent.is_dir())
        check("claimed_host dir", bus.claimed_host.is_dir())
        check("leases_host dir", bus.leases_host.is_dir())
        check("acknowledgements dir", bus.acknowledgements_host.is_dir())
        check("frozen_visible dir", bus.frozen_visible.is_dir())
        check("outcomes_visible dir", bus.outcomes_visible.is_dir())
        check("host bundles dir", bus.host_bundles.is_dir())
        check("host promotions dir", bus.host_promotions.is_dir())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_outer_loop_freeze_grant():
    tmp = Path(tempfile.mkdtemp(prefix="v2_l1_"))
    try:
        bus, plan, grant = _make_loop(tmp)
        ready = {"grant_id": grant.grant_id, "snapshot_hash": grant.grant_hash,
                 "status": "ready"}
        bus.freeze_grant(grant.to_dict(), ready)
        frozen = bus.read_frozen_grant(grant.grant_id)
        check("grant frozen", frozen is not None)
        check("grant hash matches", frozen.get("grant_hash") == grant.grant_hash)
        check("grant ticket branch", grant.ticket_obj().selected_branch_id == "baseline")
        check("grant no code in plan", "code" not in grant.plan)
        ready_files = list(bus.frozen_visible.glob("*.ready"))
        check("snapshot ready marker", len(ready_files) == 1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_agent_propose_host_claim():
    tmp = Path(tempfile.mkdtemp(prefix="v2_l1_"))
    try:
        bus, plan, grant = _make_loop(tmp)
        from pact import ProgramAgentClient
        agent = ProgramAgentClient(bus, grant.to_dict(), host_id="agent")
        proposals = agent.propose_all()
        check("agent proposed budget children", len(proposals) == 2,
              str(len(proposals)))
        check("pending files written", len(bus.list_pending()) == 2)
        ok = bus.claim(proposals[0].proposal_id, host_id="host")
        check("host claimed atomically", ok)
        check("pending now 1", len(bus.list_pending()) == 1)
        check("claimed file exists",
              bus.claimed_host.glob("proposal_*.json"))
        lease_files = list(bus.leases_host.glob("lease_*.json"))
        check("lease written", len(lease_files) == 1)
        bus.release_lease(proposals[0].proposal_id)
        check("lease released", len(list(bus.leases_host.glob("lease_*.json"))) == 0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_host_supervisor_full_child():
    tmp = Path(tempfile.mkdtemp(prefix="v2_l1_"))
    try:
        from pact import Executor, Implementer
        bus, plan, grant = _make_loop(tmp)
        work = tmp / "work"
        work.mkdir()

        code = (
            "import csv\n"
            "with open('submission.csv', 'w', newline='') as f:\n"
            "    w = csv.writer(f)\n"
            "    w.writerow(['Id', 'Prediction'])\n"
            "    for i in range(5):\n"
            "        w.writerow([i, 0])\n"
            "with open('oof.csv', 'w', newline='') as f:\n"
            "    w = csv.writer(f)\n"
            "    w.writerow(['true', 'pred'])\n"
            "    w.writerows([[1, 1], [0, 0], [1, 1], [0, 0], [1, 1]])\n"
            "print('accuracy: 0.8500')\n"
        )

        class _Imp:
            def implement(self, plan, profile, data_dir, code_dir, sub_dir, **kwargs):
                return code

        # Force host mode: on A100 V2_EXEC_IMAGE is exported and docker is up,
        # but this transactional unit test uses temp data that is not mounted
        # in any container. Container mode is exercised only by real runs.
        executor = Executor(work, docker_bin="definitely-not-a-docker")
        bundler = CandidateBundler(bus, work)
        evaluator = TrustedEvaluator(metric_name="accuracy")
        promotion = PromotionManager(bus)
        host = HostSupervisorService(
            bus=bus, executor=executor, bundler=bundler, evaluator=evaluator,
            promotion=promotion, implementer=_Imp(),
            competition="demo", max_budget_seconds=30)
        # agent proposes, then host runs the children
        from pact import ProgramAgentClient
        agent = ProgramAgentClient(bus, grant.to_dict(), host_id="agent")
        agent.propose_all()
        receipts = host.run_children(grant.to_dict(), None, plan,
                                     max_children=2)
        check("children executed", len(receipts) == 2, str(len(receipts)))
        r0 = receipts[0]
        check("evaluator recomputed metric", r0.metric == 1.0, str(r0.metric))
        check("evaluator source trusted_recompute",
              r0.evaluator_receipt.get("evaluator", "").startswith("trusted_recompute"),
              str(r0.evaluator_receipt.get("evaluator")))
        check("outcomes written", len(bus.list_outcomes()) == 2)
        ack_files = list(bus.acknowledgements_host.glob("ack_*.json"))
        check("acks written", len(ack_files) == 2)
        receipt_files = list(bus.host_receipts.glob("receipt_*.json"))
        check("receipts stored host-only", len(receipt_files) == 2)
        promo = promotion.certified_best()
        check("promotion record exists", promo.certified_best_trial_id != "")
        check("promotion certified metric", promo.certified_best_metric == 1.0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_host_daemon_serve_loop():
    """HostSupervisorService.serve(): resident polling loop that executes
    proposals as they arrive and exits once the grant budget is consumed."""
    import threading
    import time as _time
    from pact import Executor, Implementer, ProgramAgentClient
    tmp = Path(tempfile.mkdtemp(prefix="v2_l1_daemon_"))
    try:
        bus, plan, grant = _make_loop(tmp)
        work = tmp / "work"
        work.mkdir()

        code = (
            "import csv\n"
            "with open('submission.csv', 'w', newline='') as f:\n"
            "    w = csv.writer(f)\n"
            "    w.writerow(['Id', 'Prediction'])\n"
            "    w.writerow([0, 0])\n"
            "with open('oof.csv', 'w', newline='') as f:\n"
            "    w = csv.writer(f)\n"
            "    w.writerow(['true', 'pred'])\n"
            "    w.writerows([[1, 1], [0, 0]])\n"
            "print('accuracy: 1.0000')\n"
        )

        class _Imp:
            def implement(self, plan, profile, data_dir, code_dir,
                          sub_dir, **kwargs):
                return code

        executor = Executor(work, docker_bin="definitely-not-a-docker")
        host = HostSupervisorService(
            bus=bus, executor=executor,
            bundler=CandidateBundler(bus, work),
            evaluator=TrustedEvaluator(metric_name="accuracy"),
            promotion=PromotionManager(bus), implementer=_Imp(),
            competition="demo", max_budget_seconds=30)
        agent = ProgramAgentClient(bus, grant.to_dict(), host_id="agent")

        result = {}
        def _serve():
            result["receipts"] = host.serve(
                grant.to_dict(), None, plan, max_children=2,
                poll_interval=0.5, idle_exit_seconds=60)
        thread = threading.Thread(target=_serve, daemon=True)
        thread.start()
        # agent proposes child 1, waits, then child 2 (feedback cadence)
        agent.propose_next(1, "")
        _time.sleep(2.0)
        agent.propose_next(2, "")
        thread.join(timeout=90)
        check("daemon thread finished", not thread.is_alive())
        receipts = result.get("receipts") or []
        check("daemon executed both children", len(receipts) == 2,
              str(len(receipts)))
        check("daemon metrics trusted", all(
            r.metric == 1.0 for r in receipts),
            str([r.metric for r in receipts]))
        outcomes = bus.list_outcomes()
        check("daemon wrote outcomes", len(outcomes) == 2, str(len(outcomes)))
        check("daemon acked", len(list(
            bus.acknowledgements_host.glob("ack_*.json"))) == 2)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_publisher_certified_only():
    tmp = Path(tempfile.mkdtemp(prefix="v2_l1_"))
    try:
        from pact import Executor, Implementer
        bus, plan, grant = _make_loop(tmp)
        work = tmp / "work"
        sub = tmp / "submission"
        work.mkdir(); sub.mkdir()

        code = (
            "import csv\n"
            "with open('oof.csv', 'w', newline='') as f:\n"
            "    w = csv.writer(f)\n"
            "    w.writerow(['true', 'pred'])\n"
            "    w.writerows([[1, 1], [0, 0]])\n"
            "with open('submission.csv', 'w', newline='') as f:\n"
            "    w = csv.writer(f)\n"
            "    w.writerow(['Id', 'Prediction'])\n"
            "    w.writerow([0, 0])\n"
            "print('accuracy: 1.0000')\n"
        )

        class _Imp:
            def implement(self, plan, profile, data_dir, code_dir, sub_dir, **kwargs):
                return code

        executor = Executor(work, docker_bin="definitely-not-a-docker")
        bundler = CandidateBundler(bus, work)
        evaluator = TrustedEvaluator(metric_name="accuracy")
        promotion = PromotionManager(bus)
        host = HostSupervisorService(
            bus=bus, executor=executor, bundler=bundler, evaluator=evaluator,
            promotion=promotion, implementer=_Imp(),
            competition="demo", max_budget_seconds=30)
        from pact import ProgramAgentClient
        agent = ProgramAgentClient(bus, grant.to_dict(), host_id="agent")
        agent.propose_all()
        host.run_children(grant.to_dict(), None, plan, max_children=1)

        publisher = ControlledPublisher(bus, sub)
        published = publisher.publish_certified()
        check("certified submission published", published.is_file())
        check("published content", "Prediction" in published.read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_publisher_refuses_uncertified():
    tmp = Path(tempfile.mkdtemp(prefix="v2_l1_"))
    try:
        bus = FileBus(tmp / "state")
        sub = tmp / "submission"
        sub.mkdir()
        publisher = ControlledPublisher(bus, sub)
        try:
            publisher.publish_certified()
            check("uncertified publish refused", False, "no exception")
        except Exception:
            check("uncertified publish refused", True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_evaluator_recompute_vs_log():
    tmp = Path(tempfile.mkdtemp(prefix="v2_l1_"))
    try:
        from pact import Executor, Implementer
        bus, plan, grant = _make_loop(tmp)
        work = tmp / "work"
        work.mkdir()

        # code prints a LYING metric but writes honest OOF predictions
        code = (
            "import csv\n"
            "with open('oof.csv', 'w', newline='') as f:\n"
            "    w = csv.writer(f)\n"
            "    w.writerow(['true', 'pred'])\n"
            "    w.writerows([[1, 0], [0, 1]])\n"  # 0% accuracy
            "print('accuracy: 0.9900')\n"          # lie
        )

        class _Imp:
            def implement(self, plan, profile, data_dir, code_dir, sub_dir, **kwargs):
                return code

        bundler = CandidateBundler(bus, work)
        evaluator = TrustedEvaluator(metric_name="accuracy")
        spec = TrialSpec.seal("demo", plan, code)
        from pact.executor import Executor as _Exec
        # Force host mode: on A100 V2_EXEC_IMAGE is exported and docker is up,
        # but this transactional unit test uses temp data that is not mounted
        # in any container. Container mode is exercised only by real runs.
        outcome = _Exec(work, docker_bin="definitely-not-a-docker").run(
            spec, 30)
        # stage artifacts like HostSupervisorService does, then bundle
        stage = bus.ws_candidates / spec.spec_id
        stage.mkdir(parents=True, exist_ok=True)
        import shutil
        trial_work = work / spec.spec_id
        if (trial_work / "oof.csv").is_file():
            shutil.copy2(trial_work / "oof.csv", stage / "oof.csv")
        bundle = bundler.build(spec.spec_id, "p1")
        receipt = evaluator.evaluate(bundle, stdout=outcome.stdout,
                                     stderr=outcome.stderr,
                                     returncode=outcome.returncode)
        check("recompute ignores lying log", receipt.metric == 0.0,
              str(receipt.metric))
        check("evaluator source trusted_recompute",
              receipt.evaluator.startswith("trusted_recompute"),
              receipt.evaluator)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print("=== V2.1 four-layer conformance tests ===\n")
    test_file_bus_zones()
    test_outer_loop_freeze_grant()
    test_agent_propose_host_claim()
    test_host_supervisor_full_child()
    test_host_daemon_serve_loop()
    test_publisher_certified_only()
    test_publisher_refuses_uncertified()
    test_evaluator_recompute_vs_log()
    print("\nRESULT=" + ("PASS" if not FAILURES else "FAIL:" + ",".join(FAILURES)))
    sys.exit(0 if not FAILURES else 1)
