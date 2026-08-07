# -*- coding: utf-8 -*-
"""test_v2_pact.py - PACT execution/verification/guard tests (no LLM)."""
import csv as _csv
import json
import io
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from pact import (BudgetGuard, Executor, GuardError, Verifier,  # noqa: E402
                  assert_legacy_l1_mode)
from data_layout import DatasetLayout  # noqa: E402
from v2_contracts import ResearchPlan, TrialSpec  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("[OK] " + name)
    else:
        print("[FAIL] " + name + (" | " + detail if detail else ""))
        FAILURES.append(name)


def _make_env():
    tmp = Path(tempfile.mkdtemp(prefix="v2_pact_test_"))
    work = tmp / "work"
    sub = tmp / "submission"
    work.mkdir(); sub.mkdir()
    return tmp, work, sub


def test_success_path():
    tmp, work, sub = _make_env()
    try:
        code = (
            "import csv\n"
            "with open('submission.csv', 'w', newline='') as f:\n"
            "    w = csv.writer(f)\n"
            "    w.writerow(['Id', 'Prediction'])\n"
            "    for i in range(5):\n"
            "        w.writerow([i, 0])\n"
            "print('accuracy: 0.8300')\n"
        )
        spec = TrialSpec.seal("demo", ResearchPlan(method_detail={"model": "stub"}), code)
        # Force host mode: on A100 V2_EXEC_IMAGE is exported and docker is up,
        # but these unit tests use temp data that is not mounted in any
        # container. Container mode is exercised only by real runs.
        outcome = Executor(work, docker_bin="definitely-not-a-docker").run(
            spec, timeout_seconds=30)
        check("executor rc=0", outcome.returncode == 0, str(outcome.returncode))
        receipt = Verifier(sub, work_dir=work / spec.spec_id).verify(
            spec, outcome, best_metric=None)
        check("verdict success on first trial", receipt.verdict == "success",
              receipt.verdict)
        check("metric parsed", receipt.metric == 0.83, str(receipt.metric))
        check("submission found", receipt.submission_exists)
        check("submission hash", receipt.submission_hash.startswith("sha256:"))
        check("receipt ids", receipt.receipt_id.startswith("receipt_")
              and receipt.spec_id == spec.spec_id)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_failure_on_error():
    tmp, work, sub = _make_env()
    try:
        code = "raise RuntimeError('boom')"
        spec = TrialSpec.seal("demo", ResearchPlan(method_detail={"model": "stub"}), code)
        outcome = Executor(work, docker_bin="definitely-not-a-docker").run(
            spec, timeout_seconds=30)
        receipt = Verifier(sub, work_dir=work / spec.spec_id).verify(
            spec, outcome, best_metric=None)
        check("verdict failure on crash", receipt.verdict == "failure", receipt.verdict)
        check("no metric on crash", receipt.metric is None)
        check("failure reason on crash", "boom" in receipt.failure_reason,
              receipt.failure_reason)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_fail_closed_no_metric():
    tmp, work, sub = _make_env()
    try:
        code = (
            "import csv\n"
            "with open('submission.csv', 'w', newline='') as f:\n"
            "    f.write('Id,Prediction\\n0,0\\n')\n"
            "print('done')\n"
        )
        spec = TrialSpec.seal("demo", ResearchPlan(method_detail={"model": "stub"}), code)
        outcome = Executor(work, docker_bin="definitely-not-a-docker").run(
            spec, timeout_seconds=30)
        receipt = Verifier(sub, work_dir=work / spec.spec_id).verify(
            spec, outcome, best_metric=None)
        check("fail-closed: no metric -> failure", receipt.verdict == "failure",
              receipt.verdict)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_timeout():
    tmp, work, sub = _make_env()
    try:
        code = "import time\ntime.sleep(30)\n"
        spec = TrialSpec.seal("demo", ResearchPlan(method_detail={"model": "stub"}), code)
        outcome = Executor(work, docker_bin="definitely-not-a-docker").run(
            spec, timeout_seconds=1)
        check("timeout -> rc=-9", outcome.returncode == -9, str(outcome.returncode))
        receipt = Verifier(sub, work_dir=work / spec.spec_id).verify(
            spec, outcome, best_metric=None)
        check("timeout -> failure", receipt.verdict == "failure", receipt.verdict)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_verdict_comparison():
    tmp, work, sub = _make_env()
    try:
        code = (
            "import csv\n"
            "with open('submission.csv', 'w', newline='') as f:\n"
            "    f.write('Id,Prediction\\n0,0\\n')\n"
            "print('accuracy: 0.8500')\n"
        )
        spec = TrialSpec.seal("demo", ResearchPlan(method_detail={"model": "stub"}), code)
        outcome = Executor(work, docker_bin="definitely-not-a-docker").run(
            spec, timeout_seconds=30)
        verifier = Verifier(sub, work_dir=work / spec.spec_id)
        r1 = verifier.verify(spec, outcome, best_metric=0.80)
        check("success vs lower best", r1.verdict == "success", r1.verdict)
        r2 = verifier.verify(spec, outcome, best_metric=0.85)
        check("stagnant vs equal best", r2.verdict == "stagnant", r2.verdict)
        r3 = verifier.verify(spec, outcome, best_metric=0.90)
        check("regression vs higher best", r3.verdict == "regression", r3.verdict)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_budget_guard():
    guard = BudgetGuard(total_budget_seconds=100, round_timeout_seconds=30)
    check("clamp caps at round timeout", guard.clamp_round_timeout(999) <= 30)
    check("clamp respects requested", guard.clamp_round_timeout(5) == 5)
    exhausted = BudgetGuard(total_budget_seconds=1, round_timeout_seconds=1)
    import time as _t
    _t.sleep(1.1)
    try:
        exhausted.check_budget()
        check("budget exhausted raises", False, "no exception")
    except GuardError:
        check("budget exhausted raises", True)


def test_budget_guard_three_limits():
    """V2.2: MAX_GRANTS / MAX_TOTAL_TRIALS / wall clock, first hit wins."""
    guard = BudgetGuard(total_budget_seconds=86400, round_timeout_seconds=3600,
                        max_grants=3, max_total_trials=6)
    check("grant allowed initially",
          guard.grants_remaining() == 3 and guard.trials_remaining() == 6)
    guard.check_research_opportunity(children=2, est_cost_seconds=100)
    guard.commit_grant(2)
    check("grant committed reserves trials",
          guard.grants_used == 1 and guard.trials_used == 2)
    # trial limit: 4 trials left, requesting 5 must be refused
    try:
        guard.check_research_opportunity(children=5, est_cost_seconds=100)
        check("trial limit refuses", False, "no exception")
    except GuardError:
        check("trial limit refuses", True)
    # grant limit: 2 left, both consumed -> third refused
    guard.check_research_opportunity(children=2, est_cost_seconds=100)
    guard.commit_grant(2)
    guard.check_research_opportunity(children=2, est_cost_seconds=100)
    guard.commit_grant(2)
    check("grants exhausted", guard.grants_remaining() == 0)
    try:
        guard.check_research_opportunity(children=1, est_cost_seconds=100)
        check("grant limit refuses", False, "no exception")
    except GuardError:
        check("grant limit refuses", True)
    # est-cost wall-clock check
    tiny = BudgetGuard(total_budget_seconds=10, round_timeout_seconds=3600,
                       max_grants=128, max_total_trials=256)
    try:
        tiny.check_research_opportunity(children=1, est_cost_seconds=99999)
        check("est cost wall clock refuses", False, "no exception")
    except GuardError:
        check("est cost wall clock refuses", True)
    # status dict for logs/monitor
    st = guard.status()
    check("budget status exposes limits",
          st.get("max_grants") == 3 and st.get("max_total_trials") == 6
          and st.get("grants_used") == 3 and st.get("trials_used") == 6,
          str(st))


def test_budget_guard_persistence_and_crash_recovery():
    """v2.2.1: budget counters + wall deadline survive restarts, and the
    reserve -> freeze -> commit flow reconciles crash leftovers."""
    import time as _t
    tmp = Path(tempfile.mkdtemp(prefix="v2_budget_"))
    try:
        guard = BudgetGuard(total_budget_seconds=86400,
                            round_timeout_seconds=3600,
                            max_grants=3, max_total_trials=6,
                            state_dir=tmp)
        guard.begin_reservation(2, "grant_alpha")
        guard.commit_grant(2, "grant_alpha")
        guard.begin_reservation(1, "grant_beta")
        st = guard.status()
        deadline = st["wall_deadline_epoch"]
        # restart: same state dir -> counters + deadline restored
        guard2 = BudgetGuard(total_budget_seconds=86400,
                             round_timeout_seconds=3600,
                             max_grants=3, max_total_trials=6,
                             state_dir=tmp)
        check("restart restores grants", guard2.grants_used == 1,
              str(guard2.grants_used))
        check("restart restores trials", guard2.trials_used == 2,
              str(guard2.trials_used))
        check("restart keeps wall deadline",
              abs(guard2.status()["wall_deadline_epoch"] - deadline) < 2.0,
              str((guard2.status()["wall_deadline_epoch"], deadline)))
        # crash between freeze and commit: grant_beta was frozen on the bus
        rec = guard2.recover_pending(frozen_grant_ids=["grant_beta"])
        check("recovery commits frozen grant",
              rec["recovered"] == ["grant_beta"] and guard2.grants_used == 2
              and guard2.trials_used == 3,
              str((rec, guard2.status())))
        # crashed BEFORE freeze: reservation without a frozen grant is dropped
        guard2.begin_reservation(2, "grant_gamma")
        rec2 = guard2.recover_pending(frozen_grant_ids=[])
        check("recovery discards unfrozen reservation",
              rec2["discarded"] == ["grant_gamma"] and guard2.grants_used == 2
              and guard2.trials_used == 3,
              str((rec2, guard2.status())))
        # freeze after reservation, then commit -> receipts persisted
        guard2.begin_reservation(2, "grant_delta")
        guard2.commit_grant(2, "grant_delta")
        check("receipt written",
              (tmp / "budget_receipts" / "receipt_grant_delta.json").is_file())
        check("reservation cleaned",
              not (tmp / "budget_reservations" / "res_grant_delta.json").exists())
        # a third restart sees everything
        guard3 = BudgetGuard(total_budget_seconds=86400,
                             round_timeout_seconds=3600,
                             max_grants=3, max_total_trials=6,
                             state_dir=tmp)
        check("final restart restores all",
              guard3.grants_used == 3 and guard3.trials_used == 5
              and guard3.committed_grant_ids == ["grant_alpha", "grant_beta",
                                                 "grant_delta"],
              str(guard3.status()))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_budget_commit_idempotent():
    """v2.2.1-rc2: committing the same grant twice must count it once."""
    guard = BudgetGuard(total_budget_seconds=86400, round_timeout_seconds=3600,
                        max_grants=3, max_total_trials=9)
    guard.begin_reservation(2, "grant_x")
    guard.commit_grant(2, "grant_x")
    guard.commit_grant(2, "grant_x")
    check("idempotent in-memory grants", guard.grants_used == 1,
          str(guard.grants_used))
    check("idempotent in-memory trials", guard.trials_used == 2,
          str(guard.trials_used))
    tmp = Path(tempfile.mkdtemp(prefix="v2_budget_idem_"))
    try:
        g2 = BudgetGuard(total_budget_seconds=86400, round_timeout_seconds=3600,
                         max_grants=3, max_total_trials=9, state_dir=tmp)
        g2.begin_reservation(3, "grant_p")
        g2.commit_grant(3, "grant_p")
        g2.commit_grant(3, "grant_p")
        g3 = BudgetGuard(total_budget_seconds=86400, round_timeout_seconds=3600,
                         max_grants=3, max_total_trials=9, state_dir=tmp)
        check("idempotent persisted grants", g3.grants_used == 1,
              str(g3.grants_used))
        check("idempotent persisted trials", g3.trials_used == 3,
              str(g3.trials_used))
        check("one receipt only",
              len(list((tmp / "budget_receipts").glob("receipt_*.json"))) == 1,
              str(list((tmp / "budget_receipts").glob("receipt_*.json"))))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_budget_receipt_authoritative_crash_window():
    """v2.2.1-rc2: receipt written but budget_state stale (crash window)
    -> a fresh BudgetGuard rebuilds counters from the authoritative
    receipts instead of trusting the stale derived state."""
    tmp = Path(tempfile.mkdtemp(prefix="v2_budget_crash_"))
    try:
        (tmp / "budget_receipts").mkdir()
        (tmp / "budget_receipts" / "receipt_grant_a.json").write_text(
            json.dumps({"grant_id": "grant_a", "children": 2,
                        "status": "committed"}),
            encoding="utf-8")
        # stale derived state says nothing happened (rc1 crash window)
        (tmp / "budget_state.json").write_text(
            json.dumps({"run_started_at": 1000.0,
                        "wall_deadline_epoch": 999999.0,
                        "grants_committed": 0, "trials_reserved": 0,
                        "committed_grant_ids": []}),
            encoding="utf-8")
        g = BudgetGuard(total_budget_seconds=86400, round_timeout_seconds=3600,
                        max_grants=3, max_total_trials=9, state_dir=tmp)
        check("receipt authoritative grants", g.grants_used == 1,
              str(g.grants_used))
        check("receipt authoritative trials", g.trials_used == 2,
              str(g.trials_used))
        check("receipt authoritative ids",
              g.committed_grant_ids == ["grant_a"], str(g.committed_grant_ids))
        check("deadline from state metadata",
              abs(g.status()["wall_deadline_epoch"] - 999999.0) < 1.0,
              str(g.status()["wall_deadline_epoch"]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_budget_recover_uses_frozen_trial_budget():
    """v2.2.1-rc2: recovery receives the FULL frozen grant dict and rebuilds
    children from the frozen grant's trial_budget, not the reservation."""
    tmp = Path(tempfile.mkdtemp(prefix="v2_budget_frozen_"))
    try:
        g = BudgetGuard(total_budget_seconds=86400, round_timeout_seconds=3600,
                        max_grants=3, max_total_trials=9, state_dir=tmp)
        g.begin_reservation(1, "grant_f")  # reservation says 1 child
        rec = g.recover_pending(frozen_grants=[
            {"grant_id": "grant_f", "trial_budget": 3, "status": "frozen"}])
        check("frozen grant recovered", rec["recovered"] == ["grant_f"],
              str(rec))
        check("children from frozen grant",
              g.grants_used == 1 and g.trials_used == 3, str(g.status()))
        rp = tmp / "budget_receipts" / "receipt_grant_f.json"
        check("recovered receipt children=3",
              rp.is_file() and json.loads(
                  rp.read_text(encoding="utf-8")).get("children") == 3,
              rp.read_text(encoding="utf-8") if rp.is_file() else "missing")
        # double recovery must not double-count
        rec2 = g.recover_pending(frozen_grants=[
            {"grant_id": "grant_f", "trial_budget": 3}])
        check("recovery idempotent",
              not rec2["recovered"] and g.grants_used == 1
              and g.trials_used == 3, str((rec2, g.status())))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_budget_recover_receipt_first():
    """v2.2.1-rc2: pending reservation + existing receipt -> counted once
    via the receipt, never twice."""
    tmp = Path(tempfile.mkdtemp(prefix="v2_budget_rf_"))
    try:
        g = BudgetGuard(total_budget_seconds=86400, round_timeout_seconds=3600,
                        max_grants=3, max_total_trials=9, state_dir=tmp)
        g.begin_reservation(2, "grant_r")
        g.commit_grant(2, "grant_r")
        g.begin_reservation(2, "grant_r")  # stale reservation reappears
        rec = g.recover_pending(frozen_grants=[
            {"grant_id": "grant_r", "trial_budget": 2}])
        check("receipt wins, nothing recovered",
              not rec["recovered"] and g.grants_used == 1,
              str((rec, g.status())))
        check("trials counted once", g.trials_used == 2, str(g.trials_used))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_mode_guard():
    old = os.environ.get("PACT_STAGE4_SELF_EVOLUTION")
    os.environ["PACT_STAGE4_SELF_EVOLUTION"] = "1"
    try:
        try:
            assert_legacy_l1_mode()
            check("mode guard rejects=1", False, "no exception")
        except GuardError:
            check("mode guard rejects=1", True)
    finally:
        if old is None:
            os.environ.pop("PACT_STAGE4_SELF_EVOLUTION", None)
        else:
            os.environ["PACT_STAGE4_SELF_EVOLUTION"] = old
    assert_legacy_l1_mode()  # must not raise in default mode
    check("mode guard allows=0", True)


def test_implementer_stub_llm():
    from pact.implementer import Implementer
    from v2_contracts import ResearchPlan

    def stub(prompt):
        if "Write a COMPLETE Python script" in prompt:
            return "```python\nprint('accuracy: 0.9100')\n```"
        return "{}"

    imp = Implementer(llm_call_fn=stub)
    code = imp.implement(ResearchPlan(hypothesis="H", method_detail={"model": "xgb"}),
                         None, "/d", "/c", "/s")
    check("implementer strips fences", "accuracy" in code and "```" not in code,
          code[:80])


def test_implementer_fallback():
    from pact.implementer import Implementer, FALLBACK_CODE
    from v2_contracts import ResearchPlan

    imp = Implementer(llm_call_fn=lambda p: "{}")
    code = imp.implement(ResearchPlan(), None, "/d", "/c", "/s")
    check("implementer fallback on {} response", code == FALLBACK_CODE)


def test_executor_container_fallback():
    from pact.executor import Executor
    tmp, work, sub = _make_env()
    try:
        code = (
            "import csv\n"
            "with open('submission.csv', 'w', newline='') as f:\n"
            "    w = csv.writer(f)\n"
            "    w.writerow(['Id', 'Prediction'])\n"
            "    w.writerow([0, 0])\n"
            "print('accuracy: 0.5000')\n"
        )
        spec = TrialSpec.seal("demo", ResearchPlan(method_detail={"model": "stub"}), code)
        ex = Executor(work, exec_image="fake:img", exec_python="python3",
                      docker_bin="definitely-not-a-docker",
                      data_dir=str(tmp / "data"))
        check("container fallback mode host", ex.exec_mode() == "host",
              ex.exec_mode())
        outcome = ex.run(spec, timeout_seconds=30)
        check("container fallback rc", outcome.returncode == 0,
              str(outcome.returncode))
        check("container fallback metric", "accuracy: 0.5000" in outcome.stdout,
              outcome.stdout[:80])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_executor_docker_cmd_shape():
    from pact.executor import Executor
    tmp, work, sub = _make_env()
    try:
        # Build the manifest exactly like v2_closed_loop does (real path).
        from data_layout import resolve_dataset_layout
        data_root = tmp / "data"
        pub = data_root / "prepared" / "public"
        priv = data_root / "prepared" / "private"
        pub.mkdir(parents=True)
        priv.mkdir(parents=True)
        (pub / "train.csv").write_text("id,label\n1,0\n", encoding="utf-8")
        (pub / "test.csv").write_text("id\n2\n", encoding="utf-8")
        (pub / "sample_submission.csv").write_text(
            "id,label\n2,0\n", encoding="utf-8")
        (priv / "test.csv").write_text("id,label\n2,0\n", encoding="utf-8")
        manifest = resolve_dataset_layout(data_root).manifest()
        manifest.update({
            "train_csv": str(pub / "train.csv"),
            "test_csv": str(pub / "test.csv"),
            "target_column": "label",
            "task_type": "classification",
        })
        ex = Executor(work, exec_image="exec:v2",
                      exec_python="conda run --no-capture-output -n base python3",
                      docker_bin="docker", data_dir=str(data_root),
                      manifest=manifest)
        spec = TrialSpec.seal("demo", ResearchPlan(), "x")
        cmd = ex.docker_cmd(spec, "round_1_x.py", str(work),
                            {"CUDA_VISIBLE_DEVICES": "4"})
        joined = " ".join(cmd)
        check("docker cmd image", "exec:v2" in joined, joined)
        check("docker cmd entrypoint python", "--entrypoint conda" in joined
              and "run --no-capture-output" in joined and "python3" in joined,
              joined)
        check("docker cmd work mount", ":%s" % str(work) in joined)
        check("docker cmd public ro mount",
              ":%s:ro" % manifest["public_dir"] in joined)
        check("docker cmd gpu passthrough", "--gpus" in joined
              and "CUDA_VISIBLE_DEVICES=4" in joined)
        check("docker cmd thread caps",
              "OMP_NUM_THREADS=8" in joined and "MKL_NUM_THREADS=8" in joined
              and "VECLIB_MAXIMUM_THREADS=8" in joined, joined)
        check("docker cmd manifest env",
              "-e TRAIN_CSV=%s" % manifest["train_csv"] in joined
              and "-e TARGET_COLUMN=label" in joined, joined)
        check("docker cmd mounts public not data root",
              ":%s:ro" % manifest["public_dir"] in joined
              and ":%s:ro" % str(data_root) not in joined, joined)
        check("docker cmd never mounts private/gold",
              "private" not in joined, joined)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_executor_cache_dirs_env():
    """Candidate containers/host trials receive V2_CACHE_DIRS (JSON
    {size: dir}) so generated code LOADs caches instead of decoding."""
    from pact.executor import Executor
    from v2_contracts import ResearchPlan, TrialSpec
    tmp, work, sub = _make_env()
    try:
        pub = tmp / "data" / "prepared" / "public"
        pub.mkdir(parents=True)
        cache_dirs = {"64": str(work / "data_cache" / "k" / "64"),
                      "128": str(work / "data_cache" / "k" / "128")}
        for _d in cache_dirs.values():
            Path(_d).mkdir(parents=True, exist_ok=True)
        manifest = {"public_dir": str(pub),
                    "train_csv": str(pub / "train.csv"),
                    "test_csv": str(pub / "test.csv"),
                    "task_type": "classification",
                    "target_column": "label",
                    "cache_dirs": cache_dirs}
        ex = Executor(work, exec_image="exec:v2", exec_python="python3",
                      docker_bin="docker", data_dir=str(tmp / "data"),
                      manifest=manifest)
        env = ex._manifest_env({})
        import json as _json
        got = _json.loads(env.get("V2_CACHE_DIRS") or "{}")
        check("manifest env carries V2_CACHE_DIRS",
              got == cache_dirs, str(env.get("V2_CACHE_DIRS")))
        # v2.3.2: cache dirs that no longer exist must be filtered out of
        # V2_CACHE_DIRS (a missing dir must never become a hard trial
        # failure; the harness falls back to raw decode).
        missing = str(work / "data_cache" / "gone" / "64")
        manifest2 = dict(manifest)
        manifest2["cache_dirs"] = dict(cache_dirs)
        manifest2["cache_dirs"]["256"] = missing
        ex2 = Executor(work, exec_image="exec:v2", exec_python="python3",
                       docker_bin="docker", data_dir=str(tmp / "data"),
                       manifest=manifest2)
        env2 = ex2._manifest_env({})
        got2 = _json.loads(env2.get("V2_CACHE_DIRS") or "{}")
        check("missing cache dir filtered from V2_CACHE_DIRS",
              got2 == cache_dirs and "256" not in got2,
              str(env2.get("V2_CACHE_DIRS")))
        cmd = ex.docker_cmd(TrialSpec.seal("demo", ResearchPlan(), "x"),
                            "round_1_x.py", str(work), {})
        joined = " ".join(cmd)
        check("docker cmd carries V2_CACHE_DIRS",
              "V2_CACHE_DIRS=" in joined and "data_cache" in joined, joined)
        # cache dirs live under work_dir => no extra mounts required
        check("cache dirs need no extra mounts",
              "-v %s" % cache_dirs["64"] not in joined
              and "-v %s" % cache_dirs["128"] not in joined, joined)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_v2_llm_role_latency_bounds():
    """rc4-hotfix: codegen role must be streamed and bounded
    (<=600s per-chunk read, overall cap, <=2 attempts, <=4000 tokens)."""
    from v2_llm import _role_config
    codegen = _role_config("codegen")
    general = _role_config("general")
    check("codegen timeout allows qwen codegen",
          480.0 <= float(codegen["timeout"]) <= 720.0, codegen)
    check("codegen total cap present",
          float(codegen.get("total") or 0.0) >= 900.0, codegen)
    check("codegen attempts bounded", int(codegen["attempts"]) <= 2, codegen)
    check("codegen tokens bounded", int(codegen["max_tokens"]) <= 4000,
          codegen)
    check("general keeps retries", int(general["attempts"]) >= 2, general)
    check("general timeout bounded for visibility",
          float(general["timeout"]) <= 600.0, general)

def test_v2_llm_codegen_streaming_parse():
    """rc4-hotfix: codegen streams SSE deltas and handles [DONE];
    regression guard for the NameError that killed streaming."""
    import v2_llm
    class _FakeResp:
        status_code = 200
        def iter_lines(self):
            for ln in [
                'data: {"choices":[{"delta":{"content":"import "}}]}',
                'data: {"choices":[{"delta":{"content":"csv\\n"}}]}',
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
                'data: [DONE]',
            ]:
                yield ln
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
    class _FakeStream:
        def __init__(self, *a, **k):
            pass
        def __enter__(self):
            return _FakeResp()
        def __exit__(self, *a):
            return False
    class _FakeHttpx:
        @staticmethod
        def Timeout(*a, **k):
            return ("timeout", a, k)
        @staticmethod
        def stream(*a, **k):
            return _FakeStream()
    old = v2_llm.httpx
    v2_llm.httpx = _FakeHttpx
    try:
        content, err, t_first = v2_llm._stream_completion(
            "http://x", {}, {}, 600.0, 1200.0)
    finally:
        v2_llm.httpx = old
    check("streamed content accumulated",
          content == "import csv\n", content)
    check("no error on stream", err is None, err)
    check("t_first measured", t_first is not None, t_first)


def test_v2_llm_codegen_single_json_fallback():
    """Compatible endpoints may ignore stream=true and return one JSON
    body; the fallback parser must still return content."""
    import v2_llm
    class _FakeResp:
        status_code = 200
        def iter_lines(self):
            yield '{"choices":[{"message":{"content":"OK body"}}]}'
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
    class _FakeStream:
        def __init__(self, *a, **k):
            pass
        def __enter__(self):
            return _FakeResp()
        def __exit__(self, *a):
            return False
    class _FakeHttpx:
        @staticmethod
        def Timeout(*a, **k):
            return ("timeout", a, k)
        @staticmethod
        def stream(*a, **k):
            return _FakeStream()
    old = v2_llm.httpx
    v2_llm.httpx = _FakeHttpx
    try:
        content, err, t_first = v2_llm._stream_completion(
            "http://x", {}, {}, 600.0, 1200.0)
    finally:
        v2_llm.httpx = old
    check("single-json fallback parsed", content == "OK body", content)
    check("no error on json fallback", err is None, err)

def test_v2_llm_codegen_total_cap_hard():
    """Wall-clock cap must abort even when keep-alive/reasoning lines
    keep arriving with no content: regression for unbounded hangs."""
    import time as _time
    import v2_llm
    class _SlowResp:
        status_code = 200
        def iter_lines(self):
            for _i in range(10):
                _time.sleep(0.02)
                yield 'data: {"choices":[{"delta":{},"finish_reason":null}]}'
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
    class _SlowStream:
        def __init__(self, *a, **k):
            pass
        def __enter__(self):
            return _SlowResp()
        def __exit__(self, *a):
            return False
    class _FakeHttpx:
        @staticmethod
        def Timeout(*a, **k):
            return ("timeout", a, k)
        @staticmethod
        def stream(*a, **k):
            return _SlowStream()
    old = v2_llm.httpx
    v2_llm.httpx = _FakeHttpx
    try:
        content, err, t_first = v2_llm._stream_completion(
            "http://x", {}, {}, 600.0, 0.03)
    finally:
        v2_llm.httpx = old
    check("cap aborts on keep-alive lines", content is None, content)
    check("cap error reported", bool(err) and "cap" in err, err)
    check("no t_first when nothing generated", t_first is None, t_first)



def test_data_cache_builder_extension_probe():
    """Generic builder: CSV ids WITHOUT extensions (aptos style) must
    still resolve image files via the recursive id->path index (flat dirs,
    class subdirs, deeper nesting)."""
    from pact.data_cache import _builder_script
    script = _builder_script()
    check("builder indexes image extensions",
          '".png"' in script and '".jpg"' in script
          and "index_images" in script and "setdefault(e.stem" in script,
          script[:400])
    check("builder tracks per-kind rows (no rebuild on restart)",
          '"rows_" + kind' in script and '"rows_train"' in script
          and '"rows_test"' in script, script[:600])


def test_probe_script_extension_probe():
    """F0 probe raw-decode path resolves extension-less ids too."""
    from pact.calibration_probe import probe_script
    script = probe_script()
    check("probe indexes image extensions",
          '".png"' in script and '".jpg"' in script
          and "img_idx" in script and "setdefault(_e.stem" in script,
          script[:400])


def test_implementer_deterministic_baseline():
    import csv
    from pact.implementer import Implementer
    from pact.executor import Executor
    from v2_contracts import ResearchPlan
    tmp, work, sub = _make_env()
    try:
        data = tmp / "data"
        data.mkdir()
        (data / "train.csv").write_text(
            "id,label\n1,0\n2,1\n3,0\n4,0\n", encoding="utf-8")
        (data / "test.csv").write_text("id\n101\n102\n", encoding="utf-8")
        (data / "sample_submission.csv").write_text(
            "id,Prediction\n101,\n102,\n", encoding="utf-8")

        def boom(prompt):
            raise AssertionError("LLM must not be called for baseline branch")

        old = os.environ.get("V2_DETERMINISTIC_BASELINE")
        os.environ["V2_DETERMINISTIC_BASELINE"] = "1"
        try:
            imp = Implementer(llm_call_fn=boom)
            code = imp.implement(ResearchPlan(hypothesis="H"), None,
                                 data, work, sub, branch="baseline")
        finally:
            if old is None:
                os.environ.pop("V2_DETERMINISTIC_BASELINE", None)
            else:
                os.environ["V2_DETERMINISTIC_BASELINE"] = old
        check("baseline is deterministic stdlib",
              "csv" in code and "majority" not in code and "```" not in code,
              code[:80])
        spec = TrialSpec.seal("demo", ResearchPlan(), code)
        # Force host mode: on A100 V2_EXEC_IMAGE is exported, but this test's
        # temp data dir is not mounted into any container.
        outcome = Executor(work, docker_bin="definitely-not-a-docker").run(
            spec, timeout_seconds=30)
        check("baseline host mode", outcome.returncode != -2,
              str(outcome.returncode))
        check("baseline rc=0", outcome.returncode == 0, str(outcome.returncode))
        check("baseline prints accuracy",
              "accuracy: 0.7500" in outcome.stdout, outcome.stdout[:120])
        sub_rows = list(csv.reader(open(work / spec.spec_id / "submission.csv",
                                        encoding="utf-8")))
        check("baseline submission format", sub_rows[0] == ["id", "Prediction"]
              and len(sub_rows) == 3 and sub_rows[1][1] == "0", str(sub_rows))
        oof_rows = list(csv.reader(open(work / spec.spec_id / "oof.csv",
                                        encoding="utf-8")))
        check("baseline oof true,pred", oof_rows[0] == ["true", "pred"]
              and len(oof_rows) == 5, str(oof_rows[:2]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_executor_preflight_host_skipped():
    from pact.executor import Executor
    tmp, work, sub = _make_env()
    try:
        old = os.environ.get("V2_EXEC_IMAGE")
        os.environ.pop("V2_EXEC_IMAGE", None)
        try:
            report = Executor(work).preflight()
        finally:
            if old is not None:
                os.environ["V2_EXEC_IMAGE"] = old
        check("preflight host skipped",
              report.get("status") == "skipped", str(report))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_preflight_script_smoke():
    """The preflight -c script must import and run under this interpreter."""
    import subprocess
    import sys
    script = (
        "import importlib.util, os, glob\n"
        "missing = [m for m in ['os', 'csv'] if importlib.util.find_spec(m) is None]\n"
        "print('PREFLIGHT_MISSING=' + ','.join(missing))\n"
        "miss_files = [p for p in ['/nonexistent_v2_xyz'] "
        "if not os.path.isdir(p) and not os.path.isfile(p)]\n"
        "print('PREFLIGHT_MISSING_FILES=' + ','.join(miss_files))\n"
        "ckpts = sorted(glob.glob(os.path.expanduser('~/.cache/torch/hub/checkpoints/*')))\n"
        "print('PREFLIGHT_PRETRAINED=' + ','.join(os.path.basename(p) for p in ckpts))\n"
        "gold = '/nonexistent_v2_gold'\n"
        "print('PREFLIGHT_GOLD_VISIBLE=' + ('YES' if gold and "
        "os.path.exists(gold) else 'NO'))\n"
        "print('PREFLIGHT_OK')\n")
    proc = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, timeout=30)
    check("preflight script rc=0", proc.returncode == 0, proc.stderr)
    check("preflight script reports imports ok",
          "PREFLIGHT_MISSING=" in proc.stdout, proc.stdout)
    check("preflight script reports missing files",
          "PREFLIGHT_MISSING_FILES=/nonexistent_v2_xyz" in proc.stdout,
          proc.stdout)
    check("preflight script reports gold invisible",
          "PREFLIGHT_GOLD_VISIBLE=NO" in proc.stdout, proc.stdout)
    check("preflight script reports pretrained cache line",
          "PREFLIGHT_PRETRAINED=" in proc.stdout, proc.stdout)


def test_materialize_dataset():
    import zipfile
    from data_layout import materialize_dataset
    tmp, work, sub = _make_env()
    try:
        public = tmp / "data" / "prepared" / "public"
        public.mkdir(parents=True)
        with zipfile.ZipFile(public / "train.zip", "w") as zf:
            zf.writestr("a.png", b"png1")
            zf.writestr("b.png", b"png2")
        with zipfile.ZipFile(public / "test.zip", "w") as zf:
            zf.writestr("test/c.png", b"png3")
        report = materialize_dataset(tmp / "data")
        check("materialize extracted 2 zips",
              len(report.get("extracted", [])) == 2, str(report))
        check("flat zip -> public/train",
              (public / "train" / "a.png").is_file())
        check("nested zip -> public/test",
              (public / "test" / "c.png").is_file())
        report2 = materialize_dataset(tmp / "data")
        check("materialize idempotent",
              len(report2.get("extracted", [])) == 0, str(report2))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_sanitize_test_csv():
    from data_layout import sanitize_test_csv
    tmp, work, sub = _make_env()
    try:
        public = tmp / "data" / "prepared" / "public"
        private = tmp / "data" / "prepared" / "private"
        public.mkdir(parents=True)
        private.mkdir(parents=True)
        (public / "train.csv").write_text(
            "id,label\n1,0\n2,1\n", encoding="utf-8")
        (private / "test.csv").write_text(
            "id,label\n101,0\n102,1\n", encoding="utf-8")
        report = sanitize_test_csv(tmp / "data")
        out = (public / "test.csv").read_text(encoding="utf-8")
        check("sanitize wrote public test", bool(report.get("written")), str(report))
        check("sanitize dropped label column",
              out == "id\n101\n102\n", out)
        report2 = sanitize_test_csv(tmp / "data")
        check("sanitize idempotent",
              not report2.get("written"), str(report2))
        # flat layout: no prepared dir -> no-op
        flat = tmp / "flat"
        flat.mkdir()
        check("sanitize flat no-op",
              not sanitize_test_csv(flat).get("written"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_layout_gold_only_for_real_split():
    """Flat layouts have no gold; only public/private splits expose one."""
    from data_layout import resolve_dataset_layout
    tmp, work, sub = _make_env()
    try:
        flat = tmp / "flat_data"
        flat.mkdir()
        (flat / "train.csv").write_text("id,label\n1,0\n", encoding="utf-8")
        (flat / "test.csv").write_text("id\n2\n", encoding="utf-8")
        m = resolve_dataset_layout(flat).manifest()
        check("flat layout has no gold", m.get("gold_test_csv") == "",
              m.get("gold_test_csv"))
        check("flat layout no labels", m.get("test_has_labels") is False)

        ml = tmp / "mle_data"
        pub = ml / "prepared" / "public"
        priv = ml / "prepared" / "private"
        pub.mkdir(parents=True)
        priv.mkdir(parents=True)
        (pub / "train.csv").write_text("id,label\n1,0\n", encoding="utf-8")
        (priv / "test.csv").write_text("id,label\n2,0\n", encoding="utf-8")
        m2 = resolve_dataset_layout(ml).manifest()
        check("mlebench layout has gold", "private" in m2.get("gold_test_csv", ""),
              m2.get("gold_test_csv"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_quality_gate():
    from pact.quality_gate import check_code
    ok, reason = check_code("def broken(:\n")
    check("gate rejects syntax error", not ok and "syntax" in reason, reason)
    ok, reason = check_code("import os\nos.system('rm -rf /')\n")
    check("gate rejects os.system", not ok and "os.system" in reason, reason)
    ok, reason = check_code("print('please pip install torch first')\n")
    check("gate rejects pip install", not ok and "pip install" in reason, reason)
    gold = "/mnt/data/mle-bench/data/x/prepared/private/test.csv"
    ok, reason = check_code("df = pd.read_csv('%s')\n" % gold, gold_path=gold)
    check("gate rejects gold read", not ok and "gold" in reason, reason)
    ok, reason = check_code(
        "df = pd.read_csv('%s')\n" % gold, gold_path=gold, test_path=gold)
    check("gate allows designated private test source", ok, reason)
    ok, reason = check_code("import csv\nprint('accuracy: 0.5')\n", gold_path=gold)
    check("gate passes clean code", ok, reason)
    # ---- v2.3.2 regression: GPU gate must not reject the idiomatic cuda/cpu
    # dispatch (the compiled image harness renders exactly this pattern). ----
    ok, reason = check_code(
        'DEVICE = "cuda" if torch.cuda.is_available() else "cpu"\n'
        "model = model.to(DEVICE)\n", gpu_mandatory=True)
    check("gate allows cuda/cpu dispatch idiom", ok, reason)
    ok, reason = check_code(
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
        gpu_mandatory=True)
    check("gate allows torch.device dispatch idiom", ok, reason)
    ok, reason = check_code("device = 'cpu'\n", gpu_mandatory=True)
    check("gate rejects hardcoded device cpu", not ok, reason)
    ok, reason = check_code("model.to('cpu')\n", gpu_mandatory=True)
    check("gate rejects .to('cpu')", not ok, reason)
    ok, reason = check_code("torch.load(p, map_location='cpu')\n",
                            gpu_mandatory=True)
    check("gate allows map_location cpu", ok, reason)
    # both compiled image harness templates must pass the full gate
    import program_compiler
    for _tmpl in (program_compiler._IMAGE_EMBED_TEMPLATE,
                  program_compiler._IMAGE_FINETUNE_TEMPLATE):
        _ok, _reason = check_code(_tmpl, gpu_mandatory=True)
        check("gate allows compiled image template", _ok, _reason)


def test_host_quality_gate_rejects_before_run():
    from pact import (CandidateBundler, FileBus, HostSupervisorService,
                      PromotionManager, TrustedEvaluator)
    from pact.executor import Executor
    from v2_contracts import ResearchPlan
    tmp, work, sub = _make_env()
    try:
        bus = FileBus(tmp / "state")
        host = HostSupervisorService(
            bus=bus, executor=Executor(work, docker_bin="definitely-not-a-docker"),
            bundler=CandidateBundler(bus, work),
            evaluator=TrustedEvaluator(), promotion=PromotionManager(bus),
            implementer=None, gold_test_csv="/data/prepared/private/test.csv")
        spec = TrialSpec.seal(
            "demo", ResearchPlan(),
            "import subprocess\nsubprocess.run(['rm', '-rf', '/'])\n")
        outcome = host._execute(spec)
        check("host gate rc=-3", outcome.returncode == -3, str(outcome.returncode))
        check("host gate reason", "CODE_QUALITY_REJECT" in outcome.stderr,
              outcome.stderr)
        check("host gate skipped run", outcome.trial_work_dir == "",
              outcome.trial_work_dir)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_executor_mount_fail_closed():
    """A manifest without public_dir must refuse container mounts instead of
    silently falling back to the data root (which can hold private/gold)."""
    from pact.executor import Executor
    tmp, work, sub = _make_env()
    try:
        ex = Executor(work, exec_image="exec:v2",
                      exec_python="python3", docker_bin="docker",
                      data_dir=str(tmp / "data"),
                      manifest={"train_csv": "x", "test_csv": "y"})
        spec = TrialSpec.seal("demo", ResearchPlan(), "x")
        try:
            ex.docker_cmd(spec, "round_1_x.py", str(work), {})
            check("mount fail-closed raises", False, "no exception")
        except ValueError as exc:
            check("mount fail-closed raises",
                  "public_dir" in str(exc), str(exc))
        except Exception as exc:  # noqa: BLE001
            check("mount fail-closed raises", False, type(exc).__name__)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_host_fallback_artifacts_on_failure():
    """PACT writes deterministic baseline artifacts when candidate code fails,
    so every trial yields a verifiable, recomputable metric."""
    import csv as _csv
    from pact import (CandidateBundler, FileBus, HostSupervisorService,
                      PromotionManager, TrustedEvaluator)
    from pact.executor import Executor
    tmp, work, sub = _make_env()
    try:
        data = tmp / "data"
        data.mkdir()
        (data / "train.csv").write_text(
            "id,label\n1,0\n2,1\n3,0\n4,0\n", encoding="utf-8")
        (data / "test.csv").write_text("id\n101\n102\n", encoding="utf-8")
        (data / "sample_submission.csv").write_text(
            "id,Prediction\n101,\n102,\n", encoding="utf-8")

        class _Imp:
            def implement(self, plan, profile, data_dir, code_dir,
                          sub_dir, **kwargs):
                return "raise RuntimeError('boom')\n"

        bus = FileBus(tmp / "state")
        host = HostSupervisorService(
            bus=bus, executor=Executor(work, docker_bin="definitely-not-a-docker"),
            bundler=CandidateBundler(bus, work),
            evaluator=TrustedEvaluator(metric_name="accuracy"),
            promotion=PromotionManager(bus), implementer=_Imp(),
            data_dir=str(data), sample_path=str(data / "sample_submission.csv"),
            competition="demo", max_budget_seconds=30)
        spec = TrialSpec.seal("demo", ResearchPlan(), "raise RuntimeError('boom')\n")
        outcome = host._execute(spec)
        check("fallback keeps failure rc", outcome.returncode == 1,
              str(outcome.returncode))
        note = host._ensure_artifacts(spec, outcome)
        check("fallback note produced", "PACT_FALLBACK" in note, note)
        trial_work = work / spec.spec_id
        check("fallback submission written",
              (trial_work / "submission.csv").is_file())
        check("fallback oof written", (trial_work / "oof.csv").is_file())
        rows = list(_csv.reader(
            open(trial_work / "submission.csv", encoding="utf-8")))
        check("fallback submission shape",
              len(rows) == 3 and rows[1][1] == "0", str(rows))
        eval_receipt = host._evaluate("p1", spec, outcome, force=True)
        check("fallback metric recomputed", eval_receipt.metric == 0.75,
              str(eval_receipt.metric))
        check("fallback evaluator source",
              eval_receipt.evaluator.startswith("trusted_recompute"),
              eval_receipt.evaluator)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_executor_seal_immutability():
    """Sealed TrialSpec: tampered code_hash is refused (rc=-4); a valid spec
    materializes its immutable seal record next to the script."""
    import json as _json
    from pact import Executor as _Exec
    tmp, work, sub = _make_env()
    try:
        spec = TrialSpec.seal("demo", ResearchPlan(), "print('ok')\n")
        outcome = _Exec(work, docker_bin="definitely-not-a-docker").run(
            spec, timeout_seconds=30)
        check("sealed spec executed", outcome.returncode == 0,
              str(outcome.returncode))
        seal_path = work / spec.spec_id / "sealed_spec.json"
        check("seal record materialized", seal_path.is_file())
        if seal_path.is_file():
            seal = _json.loads(seal_path.read_text(encoding="utf-8"))
            check("seal pins code hash", seal.get("code_hash") == spec.code_hash,
                  str(seal.get("code_hash")))
            check("seal immutable flag", seal.get("immutable") is True,
                  str(seal))

        tampered = TrialSpec.seal("demo", ResearchPlan(), "print('ok')\n")
        tampered.code_hash = "sha256:" + "0" * 64  # forged
        bad = _Exec(work, docker_bin="definitely-not-a-docker").run(
            tampered, timeout_seconds=30)
        check("seal mismatch refused rc=-4", bad.returncode == -4,
              str(bad.returncode))
        check("seal mismatch reason", "SEAL_MISMATCH" in bad.stderr,
              bad.stderr[:120])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_evaluator_never_log_parse():
    """TrustedEvaluator: execution-log parsing is not a metric source."""
    from pact import TrustedEvaluator as _TE
    tmp, work, sub = _make_env()
    try:
        ev = _TE(metric_name="accuracy")
        receipt = ev.evaluate(None, stdout="accuracy: 0.9900",
                              stderr="", returncode=0)
        check("no oof -> no metric", receipt.metric is None,
              str(receipt.metric))
        check("evaluator marked none", receipt.evaluator == "none",
              receipt.evaluator)
        check("evidence explains oof requirement",
              "OOF" in receipt.evidence, receipt.evidence)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_executor_docker_cmd_torch_cache():
    """V2_TORCH_CACHE: shared pretrained cache mounted into trial containers."""
    from data_layout import resolve_dataset_layout
    from pact.executor import Executor as _Exec
    tmp, work, sub = _make_env()
    try:
        data_root = tmp / "data"
        pub = data_root / "prepared" / "public"
        priv = data_root / "prepared" / "private"
        pub.mkdir(parents=True)
        priv.mkdir(parents=True)
        (pub / "train.csv").write_text("id,label\n1,0\n", encoding="utf-8")
        (pub / "test.csv").write_text("id\n2\n", encoding="utf-8")
        manifest = resolve_dataset_layout(data_root).manifest()
        manifest.update({"train_csv": str(pub / "train.csv"),
                         "test_csv": str(pub / "test.csv"),
                         "target_column": "label",
                         "task_type": "classification"})
        ex = _Exec(work, exec_image="exec:v2", docker_bin="docker",
                   manifest=manifest, torch_cache="/mnt/data/torch_cache")
        spec = TrialSpec.seal("demo", ResearchPlan(), "x")
        cmd = ex.docker_cmd(spec, "round_1_x.py", str(work), {})
        joined = " ".join(cmd)
        check("torch cache mounted at container cache",
              "-v /mnt/data/torch_cache:/root/.cache/torch" in joined, joined)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_implementer_prompt_sota_contract():
    """SOTA contract: pretrained cache listed, budget 10-25 min, no bans."""
    from pact.implementer import Implementer, _read_pretrained_cache
    from v2_contracts import ResearchPlan
    import json as _json
    tmp, work, sub = _make_env()
    try:
        data = tmp / "data"
        data.mkdir()
        (data / "train.csv").write_text(
            "id,label\n1,0\n2,1\n", encoding="utf-8")
        (data / "test.csv").write_text("id\n3\n", encoding="utf-8")
        # manifest with a preflight-verified cache entry
        (work / "data_manifest.json").write_text(_json.dumps({
            "pretrained_available": ["efficientnet_b0_rwightman-7f5810bc.pth"],
        }), encoding="utf-8")
        cached = _read_pretrained_cache(work)
        check("pretrained cache read from manifest",
              cached == ["efficientnet_b0_rwightman-7f5810bc.pth"], str(cached))
        imp = Implementer(llm_call_fn=lambda p: "{}")
        prompt = imp.build_code_prompt(
            ResearchPlan(hypothesis="H"), None, data, work, sub,
            pretrained_available=cached)
        check("prompt lists pretrained cache",
              "PRETRAINED WEIGHT CACHE" in prompt
              and "efficientnet_b0_rwightman-7f5810bc.pth" in prompt,
              prompt[:200])
        check("prompt allows cached weights",
              "ALLOWED but ONLY from the PRETRAINED WEIGHT CACHE" in prompt,
              prompt[:300])
        check("prompt 10-25 min budget",
              "10-25 MINUTES" in prompt, prompt[:300])
        check("prompt no pretrained ban",
              "pretrained" in prompt and "forbidden" not in prompt,
              prompt[:300])
        check("prompt no multi-fold ban",
              "multi-fold" not in prompt
              and "5-fold" in prompt, prompt[:400])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_executor_repair_loop():
    """Arbor-style: a crashing script is repaired from its traceback and
    re-executed within the same trial budget."""
    from pact import (CandidateBundler, FileBus, HostSupervisorService,
                      PromotionManager, TrustedEvaluator)
    from pact.executor import Executor
    from pact.implementer import Implementer
    tmp, work, sub = _make_env()
    try:
        good = (
            "import csv\n"
            "with open('submission.csv', 'w', newline='') as f:\n"
            "    w = csv.writer(f)\n"
            "    w.writerow(['Id', 'Prediction'])\n"
            "    w.writerow([0, 0])\n"
            "with open('oof.csv', 'w', newline='') as f:\n"
            "    w = csv.writer(f)\n"
            "    w.writerow(['true', 'pred'])\n"
            "    w.writerows([[0, 0], [1, 1]])\n"
            "print('accuracy: 1.0000')\n")
        calls = []

        def repair_llm(prompt):
            calls.append(prompt)
            return good

        bus = FileBus(tmp / "state")
        host = HostSupervisorService(
            bus=bus, executor=Executor(work, docker_bin="definitely-not-a-docker"),
            bundler=CandidateBundler(bus, work),
            evaluator=TrustedEvaluator(metric_name="accuracy"),
            promotion=PromotionManager(bus),
            implementer=Implementer(llm_call_fn=repair_llm),
            competition="demo", max_budget_seconds=30)
        plan = ResearchPlan(hypothesis="H")
        spec = TrialSpec.seal("demo", plan, "raise RuntimeError('boom')")
        outcome = host._execute(spec, plan, None, max_repairs=2)
        check("repair rerun succeeded", outcome.returncode == 0,
              str(outcome.returncode))
        check("repair called llm", len(calls) >= 1, str(len(calls)))
        check("repair prompt has traceback",
              "RuntimeError" in (calls[0] if calls else ""),
              (calls[0] if calls else "")[:120])
        check("repair metric printed", "accuracy: 1.0000" in outcome.stdout,
              outcome.stdout[:120])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_oof_insample_allowed():
    """In-sample OOF (full-train predictions) is now ALLOWED: the audit returns an informational note only."""
    import csv as _csv
    from pact.executor import ExecOutcome
    from pact.host_supervisor import HostSupervisorService

    class _Profile:
        train_rows = 14175
        test_rows = 3325

    tmp_dirs = []

    def make(code, oof_rows, header=True, varying_pred=False):
        tmp = Path(tempfile.mkdtemp(prefix="v2_oof_leak_"))
        tmp_dirs.append(tmp)
        work = tmp / "work"
        work.mkdir()
        svc = object.__new__(HostSupervisorService)
        svc.host_id = "host"
        svc.work_dir = tmp
        spec = object.__new__(TrialSpec)
        spec.code = code
        spec.spec_id = "spec_test"
        with (work / "oof.csv").open("w", encoding="utf-8",
                                     newline="") as fh:
            w = _csv.writer(fh)
            if header:
                w.writerow(["true", "pred"])
            for i in range(oof_rows):
                pred = (i % 5) / 10.0 if varying_pred else 0.5
                w.writerow([i % 2, pred])
        outcome = ExecOutcome(returncode=0, trial_work_dir=str(work))
        return svc, spec, outcome

    try:
        leak_code = ("train_test_split(df, test_size=0.2, random_state=42)\n"
                     "oof = model.predict(df_orig_train)\n")
        svc, spec, outcome = make(leak_code, 14175)
        reason = svc._oof_semantics_note(spec, outcome, _Profile())
        check("single split + full OOF gets in-sample note",
              "in-sample" in (reason or ""), reason)

        good_code = ("train_test_split(df, test_size=0.2, random_state=42)\n"
                     "oof = model.predict(val_df)\n")
        svc, spec, outcome = make(good_code, 2835)
        reason = svc._oof_semantics_note(spec, outcome, _Profile())
        check("single split + val-only OOF ok", reason == "", reason)

        kfold_code = ("StratifiedKFold(n_splits=5)\n"
                      "for fold, (tr, va) in enumerate(kf.split(df)):\n"
                      "    oof[va] = model.predict(X[va])\n")
        svc, spec, outcome = make(kfold_code, 14175)
        reason = svc._oof_semantics_note(spec, outcome, _Profile())
        check("kfold full OOF ok", reason == "", reason)

        kfold_short = ("StratifiedKFold(n_splits=5)\n"
                       "oof = model.predict(X_val_only)\n")
        svc, spec, outcome = make(kfold_short, 2835)
        reason = svc._oof_semantics_note(spec, outcome, _Profile())
        check("kfold short OOF accepted (honest, not a leak)",
              reason == "", reason)

        subsample_ok = ("df = df.sample(n=10000, random_state=42)\n"
                        "train_test_split(df, test_size=0.2, random_state=42)\n"
                        "oof = model.predict(val_df)\n")
        svc, spec, outcome = make(subsample_ok, 2000)
        reason = svc._oof_semantics_note(spec, outcome, _Profile())
        check("subsample split val-only OOF ok",
              reason == "", reason)

        subsample_leak = ("df = df.sample(n=10000, random_state=42)\n"
                          "train_test_split(df, test_size=0.2, random_state=42)\n"
                          "oof = model.predict(df)\n")
        svc, spec, outcome = make(subsample_leak, 10000)
        reason = svc._oof_semantics_note(spec, outcome, _Profile())
        check("subsample split full-set OOF gets in-sample note",
              "in-sample" in (reason or ""), reason)

        partial_val_ok = ("train_test_split(df, test_size=0.2, random_state=42)\n"
                          "oof = model.predict(val_df.head(2000))\n")
        svc, spec, outcome = make(partial_val_ok, 2000)
        reason = svc._oof_semantics_note(spec, outcome, _Profile())
        check("partial val OOF accepted (honest subset)",
              reason == "", reason)

        print_head_ok = ("print(df.head(5))\n"
                         "print(df.sample(3))\n"
                         "train_test_split(df, test_size=0.2, random_state=42)\n"
                         "oof = model.predict(val_df)\n")
        svc, spec, outcome = make(print_head_ok, 2000)
        reason = svc._oof_semantics_note(spec, outcome, _Profile())
        check("exploratory head/sample(5) not treated as subsample",
              reason == "", reason)

        large_split_ok = ("train_test_split(df, test_size=0.9, random_state=42)\n"
                          "oof = model.predict(val_df)\n")
        svc, spec, outcome = make(large_split_ok, 12757)
        reason = svc._oof_semantics_note(spec, outcome, _Profile())
        check("declared 90% holdout accepted",
              reason == "", reason)

        import_mention_kfold = ("from sklearn.model_selection import "
                                "train_test_split, StratifiedKFold\n"
                                "skf = StratifiedKFold(n_splits=5)\n"
                                "for tr, va in skf.split(df):\n"
                                "    oof[va] = model.predict(X[va])\n")
        svc, spec, outcome = make(import_mention_kfold, 14175)
        reason = svc._oof_semantics_note(spec, outcome, _Profile())
        check("import mention does not trigger split audit",
              reason == "", reason)

        decorative_kfold_leak = ("from sklearn.model_selection import "
                                 "StratifiedKFold  # unused\n"
                                 "train_test_split(df, test_size=0.2)\n"
                                 "oof = model.predict(df)\n")
        svc, spec, outcome = make(decorative_kfold_leak, 14175)
        reason = svc._oof_semantics_note(spec, outcome, _Profile())
        check("decorative KFold import + full-df predict gets in-sample note",
              "in-sample" in (reason or ""), reason)

        no_split_full = ("model.fit(X_train, y_train)\n"
                         "oof = model.predict(X)\n")
        svc, spec, outcome = make(no_split_full, 14175,
                                  varying_pred=True)
        reason = svc._oof_semantics_note(spec, outcome, _Profile())
        check("no split + full-train VARYING OOF gets in-sample note",
              "no split or CV" in (reason or ""), reason)

        no_split_constant = ("model.fit(X_train, y_train)\n"
                             "oof = model.predict(X)\n")
        svc, spec, outcome = make(no_split_constant, 14175)
        reason = svc._oof_semantics_note(spec, outcome, _Profile())
        check("no split + full-train CONSTANT OOF accepted (baseline)",
              reason == "", reason)

        no_split_partial = ("model.fit(X_train, y_train)\n"
                            "oof = model.predict(X[:2000])\n")
        svc, spec, outcome = make(no_split_partial, 2000)
        reason = svc._oof_semantics_note(spec, outcome, _Profile())
        check("no split + partial OOF accepted (cannot prove leak)",
              reason == "", reason)

        shufflesplit_full = ("StratifiedShuffleSplit(n_splits=5, test_size=0.2)\n"
                             "for tr, va in sss.split(X, y):\n"
                             "    oof[va] = model.predict(X[va])\n")
        svc, spec, outcome = make(shufflesplit_full, 14175)
        reason = svc._oof_semantics_note(spec, outcome, _Profile())
        check("ShuffleSplit full OOF accepted (holdout family)",
              reason == "", reason)
    finally:
        for tmp in tmp_dirs:
            shutil.rmtree(tmp, ignore_errors=True)


def test_promotion_unverified_metric_rejected():
    """A rejected trial (rc!=0) must never become the certified best,
    even when its metric number beats the incumbent."""
    from pact import FileBus, PromotionManager
    tmp = Path(tempfile.mkdtemp(prefix="v2_promo_"))
    try:
        bus = FileBus(tmp / "state")
        promo = PromotionManager(bus)
        r1 = promo.promote("trial_leaky", 0.975, "leaky evidence",
                           verified=False)
        check("leaky trial not certified",
              r1.certified_best_metric is None, r1.decision)
        check("leaky decision reject", r1.decision == "reject", r1.decision)
        r2 = promo.promote("trial_honest", 0.90, "honest evidence",
                           verified=True)
        check("honest trial becomes certified",
              r2.certified_best_metric == 0.90,
              str(r2.certified_best_metric))
        check("leaky never certified pointer",
              r2.certified_best_trial_id == "trial_honest",
              r2.certified_best_trial_id)
        r3 = promo.promote("trial_leaky2", 0.99, "leaky evidence 2",
                           verified=False)
        check("unverified high metric still not certified",
              r3.certified_best_metric == 0.90,
              str(r3.certified_best_metric))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_deterministic_artifacts_atomic_overwrite():
    """Fallback artifacts must replace candidate (possibly root-owned) files
    via atomic rename, leaving no .tmp litter and no leaky content."""
    from pact.deterministic import write_deterministic_artifacts
    tmp = Path(tempfile.mkdtemp(prefix="v2_det_atomic_"))
    try:
        work = tmp / "work"
        work.mkdir()
        train = tmp / "train.csv"
        sample = tmp / "sample.csv"
        with train.open("w", encoding="utf-8", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["id", "label"])
            w.writerows([[0, 0], [1, 0], [2, 1], [3, 1]])
        with sample.open("w", encoding="utf-8", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["id", "label"])
            w.writerows([[100, 0], [101, 0]])
        with (work / "submission.csv").open("w") as fh:
            fh.write("LEAKY_SUB\n")
        with (work / "oof.csv").open("w") as fh:
            fh.write("LEAKY_OOF\n")
        layout = DatasetLayout(root=tmp, train_path=train,
                               test_path=tmp / "test.csv",
                               sample_submission_path=sample,
                               public_dir=tmp, private_dir=tmp,
                               train_image_dir=None, test_image_dir=None,
                               layout_name="flat")
        result = write_deterministic_artifacts(layout, work,
                                               metric_name="accuracy")
        check("submission written", result["submission"], str(result))
        check("oof written", result["oof"], str(result))
        sub = (work / "submission.csv").read_text(encoding="utf-8")
        oof = (work / "oof.csv").read_text(encoding="utf-8")
        check("submission replaced", "LEAKY_SUB" not in sub, sub[:80])
        check("oof replaced", "LEAKY_OOF" not in oof, oof[:80])
        check("no tmp litter", not list(work.glob("*.tmp")),
              str(list(work.glob("*.tmp"))))
        rows = list(_csv.reader(io.StringIO(oof)))
        check("oof header+rows", len(rows) == 5, str(len(rows)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)




def test_stale_proposal_quarantine():
    """Orphaned proposals (crashed previous daemon) must be quarantined,
    never allowed to kill the new daemon with proposal_grant_mismatch."""
    from pact.file_bus import FileBus
    from pact.host_supervisor import HostSupervisorService
    from v2_host_daemon import _purge_stale_pending, _wait_for_grant
    from v2_contracts import ResearchProgramGrant, new_id

    tmp = Path(tempfile.mkdtemp(prefix="v2_stale_prop_"))
    try:
        bus = FileBus(tmp)
        # two frozen grants; the newest is the one a fresh daemon serves
        g_old = {"grant_id": "grant_old", "status": "frozen",
                 "competition": "c", "task_prompt": "t",
                 "selected_branch_id": "b", "mutation_axis": "m",
                 "trial_budget": 1, "directive_hash": "h",
                 "plan": {"hypothesis": "h", "max_budget_seconds": 60},
                 "created_at": "2026-08-04T00:00:00Z"}
        g_new = dict(g_old, grant_id="grant_new",
                      created_at="2026-08-04T00:00:01Z")
        bus.freeze_grant(g_old, {"ready": True})
        bus.freeze_grant(g_new, {"ready": True})

        # 1) _wait_for_grant pins to the requested grant, not the newest
        got = _wait_for_grant(bus, grant_wait_seconds=5,
                              grant_id="grant_old")
        check("pinned grant wins", got.get("grant_id") == "grant_old",
              str(got.get("grant_id")))
        got2 = _wait_for_grant(bus, grant_wait_seconds=5)
        # NFS-style coarse mtime: both files forced to the same second;
        # created_at must still order them (regression for the remote
        # V2_INSTALL=FAIL on the A100).
        import os as _os
        _now = 1754294400.0
        for _f in bus.frozen_visible.glob("grant_*.json"):
            _os.utime(_f, (_now, _now))
        got3 = _wait_for_grant(bus, grant_wait_seconds=5)
        check("created_at breaks mtime tie",
              got3.get("grant_id") == "grant_new", str(got3.get("grant_id")))
        check("unpinned uses latest frozen",
              got2.get("grant_id") == "grant_new", str(got2.get("grant_id")))

        # 2) stale proposal + fresh proposal in pending_agent
        stale = {"proposal_id": new_id("proposal"),
                 "grant_id": "grant_old", "child_index": 1,
                 "mutation_axis": "m", "hypothesis": "stale"}
        fresh = {"proposal_id": new_id("proposal"),
                 "grant_id": "grant_new", "child_index": 1,
                 "mutation_axis": "m", "hypothesis": "fresh"}
        bus.propose(stale)
        bus.propose(fresh)

        # 3) daemon-start purge heals the bus (stale moved, fresh kept)
        purged = _purge_stale_pending(bus, "grant_new")
        check("startup purge quarantines stale", purged == 1, str(purged))
        pending = bus.list_pending()
        check("fresh proposal kept",
              len(pending) == 1 and pending[0]["proposal_id"] == fresh["proposal_id"],
              str([p.get("proposal_id") for p in pending]))
        stale_files = [f for f in (tmp / "protocol" / "outbox" / "stale_agent").glob("proposal_*.json") if not f.name.endswith(".meta.json")]
        check("stale proposal moved to stale_agent", len(stale_files) == 1,
              str(stale_files))

        # 4) supervise_once on a mismatched proposal quarantines + returns
        #    None instead of raising (daemon survives). The daemon serves
        #    grant_new, so a leftover grant_old proposal must be skipped.
        bus.claim(fresh["proposal_id"], host_id="host")  # clear the valid one
        bus.propose(stale)  # only a stale one remains in pending
        svc = object.__new__(HostSupervisorService)
        svc.host_id = "host"
        svc.bus = bus
        rec = svc.supervise_once(g_new, None, None, None)
        check("supervise_once quarantines mismatch, no raise",
              rec is None, str(rec))
        check("mismatch gone from pending",
              all(p["grant_id"] != "grant_old" for p in bus.list_pending()),
              str([p.get("grant_id") for p in bus.list_pending()]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_implementer_oof_contract_by_metric():
    """Prompt must give the metric-correct oof.csv contract: logloss needs
    per-class probability columns; accuracy keeps the true,pred pair. The
    old prompt said "two columns true,pred, never probabilities" for ALL
    metrics, which made every logloss trial fail with
    "OOF columns/values incompatible with metric logloss" (dog-breed).
    Also: image tasks must pre-decode/resize once (GPU-idle rc=-9)."""
    from pact.implementer import Implementer
    from v2_contracts import ResearchPlan
    tmp, work, sub = _make_env()
    try:
        data = tmp / "data"
        data.mkdir()
        (data / "train.csv").write_text("id,label\n1,a\n2,b\n", encoding="utf-8")
        (data / "test.csv").write_text("id\n3\n", encoding="utf-8")
        imp = Implementer(llm_call_fn=lambda p: "{}")
        p_logloss = imp.build_code_prompt(
            ResearchPlan(hypothesis="H"), None, data, work, sub,
            metric_info={"metric_name": "logloss",
                         "metric_direction": "lower_is_better",
                         "metric_label": "multi-class log loss"})
        p_acc = imp.build_code_prompt(
            ResearchPlan(hypothesis="H"), None, data, work, sub,
            metric_info={"metric_name": "accuracy",
                         "metric_direction": "higher_is_better",
                         "metric_label": "accuracy"})
        check("logloss prompt has per-class probability contract",
              "pred_<class>" in p_logloss and "never class IDs" in p_logloss,
              p_logloss[:600])
        check("logloss prompt drops the two-column integer rule",
              "exactly two columns: true,pred" not in p_logloss,
              p_logloss[:600])
        check("accuracy prompt keeps true,pred contract",
              "exactly two columns: true,pred" in p_acc, p_acc[:600])
        check("logloss fallback rule uses probability columns",
              "pred_<class> probability column" in p_logloss, p_logloss[:900])
        check("logloss self-review rule uses probability columns",
              "one true column plus one pred_<class>" in p_logloss,
              p_logloss[:1200])
        check("image prompt has pre-decode cache rule",
              "IMAGE DATA PIPELINE" in p_acc
              and "decode and resize EVERY image ONCE" in p_acc,
              p_acc[:900])
        check("image pre-decode uses resource image cap",
              "<=192px" in p_acc, p_acc[:900])
        check("prompt has budget self-check rule",
              "BUDGET SELF-CHECK" in p_acc and "max_budget_seconds" in p_acc,
              p_acc[:1500])
        check("prompt demands cuda assert",
              "assert next(model.parameters()).is_cuda" in p_acc, p_acc[:1500])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)



def test_executor_docker_cmd_thread_caps_override():
    """V2_CPU_THREADS must override the default 8-thread cap so an admin
    can tune CPU/GPU balance without editing code (thread oversubscription
    caused GPU-idle rc=-9 timeouts)."""
    from pact.executor import Executor
    tmp, work, sub = _make_env()
    old = os.environ.get("V2_CPU_THREADS")
    try:
        os.environ["V2_CPU_THREADS"] = "4"
        ex = Executor(work, exec_image="exec:v2", exec_python="python3",
                      docker_bin="docker", data_dir=str(tmp),
                      manifest={"public_dir": str(tmp)})
        cmd = ex.docker_cmd(TrialSpec.seal("demo", ResearchPlan(), "x"),
                            "round_1_x.py", str(work), {})
        joined = " ".join(cmd)
        check("thread cap override applied",
              "OMP_NUM_THREADS=4" in joined and "MKL_NUM_THREADS=4" in joined,
              joined)
    finally:
        if old is None:
            os.environ.pop("V2_CPU_THREADS", None)
        else:
            os.environ["V2_CPU_THREADS"] = old
        shutil.rmtree(tmp, ignore_errors=True)


def test_host_trial_timeout_uses_plan_budget():
    """v2_host_daemon passes guards=None, so the plan's max_budget_seconds
    used to be ignored and every trial ran to the 60-min round timeout
    (rc=-9 far too late; a 1500s trial survived 33+ min). The per-trial
    timeout must honor the plan budget even without guards."""
    from pact.host_supervisor import trial_timeout_seconds
    check("plan budget honored without guards",
          trial_timeout_seconds(1500, 3600, None) == 1500)
    check("plan budget capped by round timeout",
          trial_timeout_seconds(7200, 3600, None) == 3600)
    check("missing plan budget falls back to round timeout",
          trial_timeout_seconds(None, 3600, None) == 3600)
    check("zero plan budget falls back to round timeout",
          trial_timeout_seconds(0, 3600, None) == 3600)
    check("plan budget keeps 1s floor",
          trial_timeout_seconds(1, 3600, None) == 1)




def test_cache_key_content_addressed():
    """v2 size-free content key: same data => same key; any change => new.
    The size dimension lives INSIDE the cache tree (multi-size rc4)."""
    from pact.data_cache import cache_key
    m = {"train_images": "/d/train", "test_images": "/d/test",
         "train_csv": "/d/train.csv", "test_csv": "/d/test.csv"}
    k1 = cache_key(m)
    check("cache key stable", cache_key(dict(m)) == k1, k1)
    m2 = dict(m); m2["train_images"] = "/d/train2"
    check("cache key changes with image dir", cache_key(m2) != k1)
    m3 = dict(m); m3["test_csv"] = "/d/other.csv"
    check("cache key changes with test csv", cache_key(m3) != k1)


def test_data_cache_builder_cmd_safety():
    """Multi-size builder (rc4): mounts ONLY work dir (rw) + public dir
    (ro); env contract is CACHE_ROOT + CACHE_SIZES (never the data root);
    one docker run yields every requested size; idempotent hits."""
    import json as _json
    from pact.data_cache import ensure_image_caches, ensure_image_cache
    tmp, work, sub = _make_env()
    try:
        data = tmp / "data"
        pub = data / "prepared" / "public"
        pub.mkdir(parents=True)
        manifest = {
            "public_dir": str(pub),
            "train_images": str(pub / "train"),
            "test_images": str(pub / "test"),
            "train_csv": str(pub / "train.csv"),
            "test_csv": str(pub / "test.csv"),
        }
        calls = []

        def fake_run(cmd):
            joined = " ".join(cmd)
            calls.append(joined)
            root = key = sizes = None
            for part in cmd:
                if part.startswith("CACHE_ROOT="):
                    root = part.split("=", 1)[1]
                if part.startswith("CACHE_KEY="):
                    key = part.split("=", 1)[1]
                if part.startswith("CACHE_SIZES="):
                    sizes = _json.loads(part.split("=", 1)[1])
            for s in sizes or [192]:
                d = Path(root) / str(s)
                d.mkdir(parents=True, exist_ok=True)
                (d / "meta.json").write_text(
                    _json.dumps({"key": key, "rows_train": 3, "rows_test": 2}),
                    encoding="utf-8")
            return (0, "CACHE_BUILD_OK train=3 test=2", "")

        res = ensure_image_caches(work, manifest, sizes=[64, 128, 192],
                                  docker_bin="docker", exec_image="exec:v2",
                                  exec_python="/opt/python3", run_fn=fake_run)
        joined = calls[0]
        check("multi-size built", sorted(int(s) for s in res) == [64, 128, 192],
              sorted(res))
        check("cache work dir mounted rw",
              "-v %s:%s" % (str(work), str(work)) in joined, joined)
        check("cache public dir mounted ro",
              "-v %s:%s:ro" % (str(pub), str(pub)) in joined, joined)
        check("cache never mounts data root",
              "-v %s:%s" % (str(tmp), str(tmp)) not in joined
              and "private" not in joined, joined)
        check("cache entrypoint python + script",
              "--entrypoint /opt/python3" in joined
              and "_cache_build.py" in joined, joined)
        check("cache env contract multi-size",
              "CACHE_TRAIN_IMAGES=%s" % manifest["train_images"] in joined
              and "CACHE_SIZES=[64, 128, 192]" in joined
              and "CACHE_ROOT=" in joined, joined)
        # idempotent second call => no new docker run
        res2 = ensure_image_caches(work, manifest, sizes=[64, 128, 192],
                                   docker_bin="docker", exec_image="exec:v2",
                                   exec_python="/opt/python3", run_fn=fake_run)
        check("cache built rows reported",
              res[64].get("rows_train") == 3
              and res[64].get("rows_test") == 2, res[64])
        check("cache second call is a hit",
              all(v.get("status") == "hit" for v in res2.values())
              and len(calls) == 1, res2)
        check("cache hit rows reported from meta",
              res2[64].get("rows_train") == 3
              and res2[64].get("rows_test") == 2, res2[64])
        # legacy single-size wrapper still routes through the multi-size
        # builder and hits the already-built 192 dir
        info = ensure_image_cache(work, manifest, image_size=192,
                                  docker_bin="docker", exec_image="exec:v2",
                                  exec_python="/opt/python3", run_fn=fake_run)
        check("single-size wrapper hits multi-size tree",
              info.get("status") == "hit" and len(calls) == 1, info)
    finally:
        import shutil as _shutil
        _shutil.rmtree(tmp, ignore_errors=True)






def test_data_cache_stale_hit_rebuilds():
    """Stale-cache guard: the cache key is path-based; when the CSV under
    the same path gains/loses rows (prepared dir regenerated), a previous
    hit must be invalidated and rebuilt - never served stale."""
    import json as _json
    from pact.data_cache import ensure_image_caches
    tmp, work, sub = _make_env()
    try:
        data = tmp / "data"
        pub = data / "prepared" / "public"
        pub.mkdir(parents=True)
        train_csv = pub / "train.csv"
        test_csv = pub / "test.csv"
        train_csv.write_text("id,label\n1,0\n2,1\n3,0\n", encoding="utf-8")
        test_csv.write_text("id\n101\n102\n", encoding="utf-8")
        manifest = {
            "public_dir": str(pub),
            "train_images": str(pub / "train"),
            "test_images": str(pub / "test"),
            "train_csv": str(train_csv),
            "test_csv": str(test_csv),
        }
        calls = []

        def fake_run(cmd):
            calls.append(cmd)
            root = key = sizes = t_csv = None
            for part in cmd:
                if part.startswith("CACHE_ROOT="):
                    root = part.split("=", 1)[1]
                if part.startswith("CACHE_KEY="):
                    key = part.split("=", 1)[1]
                if part.startswith("CACHE_SIZES="):
                    sizes = _json.loads(part.split("=", 1)[1])
                if part.startswith("CACHE_TRAIN_CSV="):
                    t_csv = part.split("=", 1)[1]
            n_tr = sum(1 for _ in open(t_csv, encoding="utf-8")) - 1
            for s in sizes or [192]:
                d = Path(root) / str(s)
                d.mkdir(parents=True, exist_ok=True)
                (d / "meta.json").write_text(
                    _json.dumps({"key": key, "rows_train": n_tr,
                                 "rows_test": 2}),
                    encoding="utf-8")
            return (0, "CACHE_BUILD_OK", "")

        res = ensure_image_caches(work, manifest, sizes=[64],
                                  docker_bin="docker", exec_image="exec:v2",
                                  exec_python="/opt/python3", run_fn=fake_run)
        check("first build rows", res[64].get("rows_train") == 3, res[64])
        check("first build one run", len(calls) == 1, len(calls))
        # same paths, CSV content changed (regenerated prepared dir)
        train_csv.write_text("id,label\n1,0\n2,1\n3,0\n4,1\n",
                             encoding="utf-8")
        res2 = ensure_image_caches(work, manifest, sizes=[64],
                                   docker_bin="docker", exec_image="exec:v2",
                                   exec_python="/opt/python3", run_fn=fake_run)
        check("stale hit rebuilt", len(calls) == 2, len(calls))
        check("rebuilt rows match csv", res2[64].get("rows_train") == 4,
              res2[64])
        res3 = ensure_image_caches(work, manifest, sizes=[64],
                                   docker_bin="docker", exec_image="exec:v2",
                                   exec_python="/opt/python3", run_fn=fake_run)
        check("rebuilt cache hits", len(calls) == 2
              and res3[64].get("status") == "hit", res3)
    finally:
        import shutil as _shutil
        _shutil.rmtree(tmp, ignore_errors=True)


def test_image_harness_cache_mismatch_guard():
    """Image templates must never silently consume a stale cache: row/id
    mismatch with the CSV falls back to raw decode (correctness over
    speed) - generic for every MLE-Bench task."""
    import program_compiler as pc
    for name, tpl in (("embed", pc._IMAGE_EMBED_TEMPLATE),
                      ("finetune", pc._IMAGE_FINETUNE_TEMPLATE)):
        check(name + " cache mismatch guard",
              "cache mismatch" in tpl and "raw decode fallback" in tpl
              and 'kind + "_ids.json"' in tpl,
              tpl[:160])


def test_data_cache_container_names_unique():
    """rc=125 regression: 3 concurrent tasks started cache builds in the
    same second and collided on the time-only container name. Names must
    carry a random suffix so concurrent builds never conflict."""
    import json as _json
    from pact.data_cache import ensure_image_caches
    tmp, work, sub = _make_env()
    try:
        data = tmp / "data"
        pub = data / "prepared" / "public"
        pub.mkdir(parents=True)
        names = []
        calls = []

        def fake_run(cmd):
            calls.append(cmd)
            for i, part in enumerate(cmd):
                if part == "--name":
                    names.append(cmd[i + 1])
            for part in cmd:
                if part.startswith("CACHE_ROOT="):
                    root = part.split("=", 1)[1]
                if part.startswith("CACHE_KEY="):
                    key = part.split("=", 1)[1]
                if part.startswith("CACHE_SIZES="):
                    sizes = _json.loads(part.split("=", 1)[1])
            for s in sizes or [192]:
                d = Path(root) / str(s)
                d.mkdir(parents=True, exist_ok=True)
                (d / "meta.json").write_text(
                    _json.dumps({"key": key, "rows_train": 3, "rows_test": 2}),
                    encoding="utf-8")
            return (0, "CACHE_BUILD_OK train=3 test=2", "")

        # two DIFFERENT manifests (different image dirs) => two real builds
        for suffix in ("a", "b"):
            m = {
                "public_dir": str(pub),
                "train_images": str(pub / ("train_" + suffix)),
                "test_images": str(pub / ("test_" + suffix)),
                "train_csv": str(pub / ("train_" + suffix + ".csv")),
                "test_csv": str(pub / ("test_" + suffix + ".csv")),
            }
            ensure_image_caches(work, m, sizes=[64],
                                docker_bin="docker", exec_image="exec:v2",
                                exec_python="/opt/python3", run_fn=fake_run)
        check("two builds ran", len(calls) == 2, len(calls))
        check("container names carry random suffix",
              all(re.match(r"^v2_cache_\d+_[0-9a-f]{8}$", n) for n in names),
              names)
        check("concurrent-safe names differ", len(set(names)) == 2, names)
    finally:
        import shutil as _shutil
        _shutil.rmtree(tmp, ignore_errors=True)


def test_implementer_cache_rule_in_prompt():
    """Every prompt (any task type) must reference the shared zero-decode
    cache so trials stop re-decoding images 10-30min per child."""
    from pact.implementer import Implementer
    from v2_contracts import ResearchPlan
    tmp, work, sub = _make_env()
    try:
        data = tmp / "data"
        data.mkdir()
        (data / "train.csv").write_text("id,label\n1,a\n2,b\n", encoding="utf-8")
        (data / "test.csv").write_text("id\n3\n", encoding="utf-8")
        imp = Implementer(llm_call_fn=lambda p: "{}")
        p_acc = imp.build_code_prompt(
            ResearchPlan(hypothesis="H"), None, data, work, sub,
            metric_info={"metric_name": "accuracy",
                         "metric_direction": "higher_is_better",
                         "metric_label": "accuracy"})
        check("prompt has shared data cache rule",
              "SHARED DATA CACHE" in p_acc and "train_X.npy" in p_acc,
              p_acc[:2000])
        check("cache rule forbids decode when cache exists",
              "NEVER decode images" in p_acc, p_acc[:2000])
    finally:
        import shutil as _shutil
        _shutil.rmtree(tmp, ignore_errors=True)


def test_implementer_prompt_reference_asset():
    """Round-continuity: the verified best code of a previous round is
    offered to the implementer as a replaceable asset (never a mandated
    method), so the next round builds on success instead of rewriting from
    baseline."""
    from pact.implementer import Implementer
    from v2_contracts import ResearchPlan
    tmp, work, sub = _make_env()
    try:
        data = tmp / "data"
        data.mkdir()
        (data / "train.csv").write_text("id,label\n1,a\n2,b\n", encoding="utf-8")
        (data / "test.csv").write_text("id\n3\n", encoding="utf-8")
        imp = Implementer(llm_call_fn=lambda p: "{}")
        prev = "print('prev best')\n"
        prompt = imp.build_code_prompt(
            ResearchPlan(hypothesis="H"), None, data, work, sub,
            reference_code=prev,
            reference_meta={"round_num": 1, "metric": 0.9,
                            "branch_id": "baseline"})
        check("prompt offers previous best code",
              "PREVIOUS BEST CODE" in prompt and prev in prompt,
              prompt[:2000])
        check("prompt frames it as an asset, not a mandate",
              "asset" in prompt and "NOT a mandate" in prompt
              and "may also replace it" in prompt, prompt[:2000])
        check("prompt carries incumbent metadata",
              "round=1" in prompt and "metric=0.9" in prompt,
              prompt[:2000])
        plain = imp.build_code_prompt(ResearchPlan(hypothesis="H"), None,
                                      data, work, sub)
        check("no reference -> no section",
              "PREVIOUS BEST CODE" not in plain)
    finally:
        import shutil as _shutil
        _shutil.rmtree(tmp, ignore_errors=True)




if __name__ == "__main__":
    print("=== V2 PACT tests ===\n")
    test_implementer_stub_llm()
    test_implementer_fallback()
    test_executor_container_fallback()
    test_executor_docker_cmd_shape()
    test_executor_docker_cmd_torch_cache()
    test_implementer_prompt_sota_contract()
    test_executor_mount_fail_closed()
    test_executor_seal_immutability()
    test_evaluator_never_log_parse()
    test_host_fallback_artifacts_on_failure()
    test_executor_repair_loop()
    test_implementer_deterministic_baseline()
    test_executor_preflight_host_skipped()
    test_preflight_script_smoke()
    test_materialize_dataset()
    test_sanitize_test_csv()
    test_layout_gold_only_for_real_split()
    test_quality_gate()
    test_oof_insample_allowed()
    test_promotion_unverified_metric_rejected()
    test_deterministic_artifacts_atomic_overwrite()
    test_stale_proposal_quarantine()
    test_implementer_oof_contract_by_metric()
    test_executor_docker_cmd_thread_caps_override()
    test_host_trial_timeout_uses_plan_budget()
    test_cache_key_content_addressed()
    test_data_cache_builder_cmd_safety()
    test_data_cache_container_names_unique()
    test_data_cache_stale_hit_rebuilds()
    test_image_harness_cache_mismatch_guard()
    test_executor_cache_dirs_env()
    test_v2_llm_codegen_streaming_parse()
    test_v2_llm_codegen_single_json_fallback()
    test_v2_llm_codegen_total_cap_hard()
    test_v2_llm_role_latency_bounds()
    test_data_cache_builder_extension_probe()
    test_probe_script_extension_probe()
    test_implementer_cache_rule_in_prompt()
    test_implementer_prompt_reference_asset()
    test_host_quality_gate_rejects_before_run()
    test_success_path()
    test_failure_on_error()
    test_fail_closed_no_metric()
    test_timeout()
    test_verdict_comparison()
    test_budget_guard()
    test_budget_guard_three_limits()
    test_budget_guard_persistence_and_crash_recovery()
    test_budget_commit_idempotent()
    test_budget_receipt_authoritative_crash_window()
    test_budget_recover_uses_frozen_trial_budget()
    test_budget_recover_receipt_first()
    test_mode_guard()
    print("\nRESULT=" + ("PASS" if not FAILURES else "FAIL:" + ",".join(FAILURES)))
    sys.exit(0 if not FAILURES else 1)
