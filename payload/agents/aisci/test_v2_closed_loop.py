# -*- coding: utf-8 -*-
"""test_v2_closed_loop.py - End-to-end closed-loop tests (stub LLM)."""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from v2_closed_loop import ClosedLoop, build_parser, main  # noqa: E402
from pact import (FileBus, HostSupervisorService, PactLedger)  # noqa: E402
from v2_contracts import AnalysisProfile, ResearchPlan, TrialReceipt  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("[OK] " + name)
    else:
        print("[FAIL] " + name + (" | " + detail if detail else ""))
        FAILURES.append(name)


def _args(tmp, **overrides):
    base = {
        "competition": "stub_comp",
        "task_prompt": "Stub binary classification",
        "max_rounds": 2,
        "round_timeout": 60,
        "total_budget": 300,
        "data_dir": str(tmp / "data"),
        "work_dir": str(tmp / "work"),
        "submission_dir": str(tmp / "sub"),
        "state_dir": str(tmp / "state"),
        "sample_path": "",
        "host_daemon": False,
        "trial_budget": 3,
    }
    base.update(overrides)
    argv = ["--competition", base["competition"],
            "--task-prompt", base["task_prompt"],
            "--max-rounds", str(base["max_rounds"]),
            "--round-timeout", str(base["round_timeout"]),
            "--total-budget", str(base["total_budget"]),
            "--data-dir", base["data_dir"],
            "--work-dir", base["work_dir"],
            "--submission-dir", base["submission_dir"],
            "--state-dir", base["state_dir"]]
    argv.extend(["--trial-budget", str(base["trial_budget"])])
    if base.get("host_daemon"):
        argv.append("--host-daemon")
    if base["sample_path"]:
        argv.extend(["--sample-path", base["sample_path"]])
    return build_parser().parse_args(argv)


def _stub_llm(metric=0.85):
    def stub(prompt):
        if "Return a JSON plan" in prompt:
            return json.dumps({
                "hypothesis": "Stub hypothesis",
                "approach_type": "explore",
                "expected_improvement": "baseline",
                "risk": "Low",
                "method_detail": {"model": "random_forest", "features": "all"},
                "max_budget_seconds": 60,
            })
        if "Write a COMPLETE Python script" in prompt:
            return (
                "import csv\n"
                "with open('submission.csv', 'w', newline='') as f:\n"
                "    w = csv.writer(f)\n"
                "    w.writerow(['Id', 'Prediction'])\n"
                "    for i in range(10):\n"
                "        w.writerow([i, 0])\n"
                "with open('oof.csv', 'w', newline='') as f:\n"
                "    w = csv.writer(f)\n"
                "    w.writerow(['true', 'pred'])\n"
                "    for i in range(20):\n"
                "        w.writerow([1 if i < 17 else 0, 1])\n"
                "print('accuracy: %.4f')\n" % metric
            )
        return "{}"
    return stub


def _setup(tmp):
    data = tmp / "data"
    data.mkdir()
    (data / "train.csv").write_text("a,b,target\n1,0,1\n2,1,0\n", encoding="utf-8")
    (data / "test.csv").write_text("a,b\n3,1\n", encoding="utf-8")


def _setup_mlebench(tmp, nl=None):
    """Build a minimal mlebench-prepared layout. nl=None keeps the platform
    default newline (Path.write_text); nl="\n" forces LF so the byte-identity
    contract is tested the same way on Windows and Linux."""
    data = tmp / "data"
    public = data / "prepared" / "public"
    private = data / "prepared" / "private"
    public.mkdir(parents=True)
    private.mkdir(parents=True)

    def _w(path, text):
        if nl is None:
            path.write_text(text, encoding="utf-8")
        else:
            with path.open("w", encoding="utf-8", newline="") as fh:
                fh.write(text.replace("\n", nl))

    _w(public / "train.csv", "image,label\nimg_1.jpg,0\nimg_2.jpg,1\n")
    _w(public / "test.csv", "image\nimg_1.jpg\nimg_2.jpg\n")
    sample = public / "sample_submission.csv"
    _w(sample, "image,label\nimg_1.jpg,0\nimg_2.jpg,0\n")
    _w(private / "test.csv", "image,label\nimg_1.jpg,0\nimg_2.jpg,0\n")
    return data, sample


def test_closed_loop_one_round():
    tmp = Path(tempfile.mkdtemp(prefix="v2_loop_test_"))
    try:
        _setup(tmp)
        loop = ClosedLoop(_args(tmp, max_rounds=1), llm_call_fn=_stub_llm(0.85))
        result = loop.run()
        check("loop completed", result["status"] == "completed")
        check("one round", result["rounds_completed"] == 1, str(result["rounds_completed"]))
        check("best metric computed (v2.3 compiled baseline)",
              result["best_metric"] is not None, str(result["best_metric"]))
        check("submission produced", result["submission_exists"] is True)

        state = tmp / "state"
        ledger = state / "experiment_ledger.jsonl"
        check("ledger file exists", ledger.exists())
        lines = [l for l in ledger.read_text(encoding="utf-8").splitlines() if l.strip()]
        check("ledger has verified records", len(lines) >= 1, str(len(lines)))
        rec = json.loads(lines[0])
        check("record verdict success", rec.get("verdict") == "success", str(rec.get("verdict")))
        check("record code hash", str(rec.get("code_hash", "")).startswith("sha256:"))
        check("report written", (state / "run_report.json").exists())
        check("evidence nodes", (state / "evidence_nodes.json").exists())
        check("causal edges", (state / "causal_edges.json").exists())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_closed_loop_no_llm_fallback():
    tmp = Path(tempfile.mkdtemp(prefix="v2_loop_test_"))
    try:
        _setup(tmp)
        loop = ClosedLoop(_args(tmp, max_rounds=1), llm_call_fn=lambda p: "{}")
        result = loop.run()
        check("no-llm loop completed", result["status"] == "completed")
        check("no-llm submission produced", result["submission_exists"] is True)
        check("no-llm rounds", result["rounds_completed"] == 1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_closed_loop_mlebench_sample_fallback():
    tmp = Path(tempfile.mkdtemp(prefix="v2_loop_mlebench_test_"))
    try:
        data, sample = _setup_mlebench(tmp)
        loop = ClosedLoop(
            _args(tmp, data_dir=str(data), sample_path=str(sample), max_rounds=1),
            llm_call_fn=lambda p: "{}")
        result = loop.run()
        published = tmp / "sub" / "submission.csv"
        check("mlebench loop completed", result["status"] == "completed")
        check("mlebench submission produced", published.is_file())
        check("mlebench sample copied exactly",
              published.read_bytes() == sample.read_bytes())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_closed_loop_mlebench_sample_fallback_lf():
    """Linux-server regression: samples use LF line endings; the compiled
    harness / deterministic fallback must follow the sample's newline style
    so the published submission is byte-identical on any OS."""
    tmp = Path(tempfile.mkdtemp(prefix="v2_loop_mlebench_lf_"))
    try:
        data, sample = _setup_mlebench(tmp, nl="\n")
        loop = ClosedLoop(
            _args(tmp, data_dir=str(data), sample_path=str(sample),
                  max_rounds=1),
            llm_call_fn=lambda p: "{}")
        result = loop.run()
        published = tmp / "sub" / "submission.csv"
        check("lf loop completed", result["status"] == "completed")
        check("lf submission produced", published.is_file())
        check("lf sample copied exactly",
              published.read_bytes() == sample.read_bytes())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_feedback_driven_children():
    """FeedbackView: child N+1 is proposed only after child N's outcome, and
    its proposal carries the prior child's verified metric as evidence."""
    tmp = Path(tempfile.mkdtemp(prefix="v2_loop_feedback_"))
    try:
        _setup(tmp)
        loop = ClosedLoop(
            _args(tmp, max_rounds=1, trial_budget=2),
            llm_call_fn=_stub_llm(0.85))
        result = loop.run()
        check("feedback loop completed", result["status"] == "completed")
        check("feedback loop two children", result["total_trials"] == 2,
              str(result["total_trials"]))
        # proposals are claimed by the host; order by child_index
        claimed = []
        for p in (tmp / "state" / "protocol" / "outbox" / "claimed_host")\
                .glob("proposal_*.json"):
            claimed.append(json.loads(p.read_text(encoding="utf-8")))
        claimed.sort(key=lambda d: int(d.get("child_index") or 0))
        check("two proposals claimed", len(claimed) == 2, str(len(claimed)))
        check("child2 evidence carries child1 outcome",
              len(claimed) == 2 and "child 1" in claimed[1].get("evidence", ""),
              (claimed[1].get("evidence", "") if len(claimed) == 2 else "")[:200])
        _metric = result["best_metric"]
        check("child2 evidence carries child1 outcome",
              len(claimed) == 2 and "metric=" in claimed[1].get("evidence", ""),
              (claimed[1].get("evidence", "") if len(claimed) == 2 else "")[:200])
        check("child2 evidence carries actual metric",
              len(claimed) == 2 and _metric is not None
              and "metric=%s" % _metric in claimed[1].get("evidence", ""),
              (claimed[1].get("evidence", "") if len(claimed) == 2 else "")[:200])
        check("feedback best metric (v2.3 compiled baseline)",
              result["best_metric"] == _metric and _metric is not None,
              str(result["best_metric"]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_intent_authority_recomputes_children():
    """v2.2.1 acceptance probe: planner proposes confirmation/3, the
    prioritizer changes the intent to final_training -> the FROZEN grant
    must carry final_training with EXACTLY 1 child (recomputed from the
    FINAL intent), and the budget must reserve 1 trial (not 3)."""
    tmp = Path(tempfile.mkdtemp(prefix="v2_loop_intent_"))
    _saved = os.environ.get("STAGE_PROFILE")
    os.environ["STAGE_PROFILE"] = "S4_sprint"
    try:
        _setup(tmp)

        def stub(prompt):
            if "Return a JSON plan" in prompt:
                return json.dumps({
                    "hypothesis": "Stub",
                    "approach_type": "explore",
                    "expected_improvement": "x",
                    "risk": "Low",
                    "research_intent": "confirmation",
                    "children": 3,
                    "method_detail": {"model": "random_forest"},
                    "max_budget_seconds": 60,
                })
            if 'Return JSON: {"selected_branch_id"' in prompt:
                return json.dumps({
                    "selected_branch_id": "baseline",
                    "mutation_axis": "hyperparameter",
                    "research_intent": "final_training",
                    "reason": "probe",
                    "new_branches": [],
                })
            if "Write a COMPLETE Python script" in prompt:
                return ("import csv\n"
                        "with open('submission.csv', 'w', newline='') as f:\n"
                        "    w = csv.writer(f)\n"
                        "    w.writerow(['Id', 'Prediction'])\n"
                        "    for i in range(10):\n"
                        "        w.writerow([i, 0])\n"
                        "with open('oof.csv', 'w', newline='') as f:\n"
                        "    w = csv.writer(f)\n"
                        "    w.writerow(['true', 'pred'])\n"
                        "    for i in range(20):\n"
                        "        w.writerow([1 if i < 17 else 0, 1])\n"
                        "print('accuracy: 0.8500')\n")
            return "{}"

        loop = ClosedLoop(_args(tmp, max_rounds=1, trial_budget=3),
                          llm_call_fn=stub)
        result = loop.run()
        check("intent loop completed", result["status"] == "completed",
              str(result))
        frozen = list((tmp / "state" / "protocol" / "frozen_visible")
                      .glob("grant_*.json"))
        check("one grant frozen", len(frozen) == 1, str(len(frozen)))
        grant = json.loads(frozen[0].read_text(encoding="utf-8"))
        check("frozen intent = prioritizer final intent",
              grant.get("research_intent") == "final_training",
              str(grant.get("research_intent")))
        check("frozen children recomputed from FINAL intent",
              grant.get("trial_budget") == 1,
              str(grant.get("trial_budget")))
        check("ticket carries final intent/children",
              grant.get("ticket", {}).get("research_intent") == "final_training"
              and grant.get("ticket", {}).get("trial_budget") == 1,
              str(grant.get("ticket")))
        budget = json.loads((tmp / "state" / "budget_state.json")
                            .read_text(encoding="utf-8"))
        check("budget reserved exactly 1 trial",
              budget.get("trials_reserved") == 1
              and budget.get("grants_committed") == 1,
              str(budget))
        check("budget receipt exists",
              len(list((tmp / "state" / "budget_receipts").glob("receipt_*.json"))) == 1)
    finally:
        if _saved is None:
            os.environ.pop("STAGE_PROFILE", None)
        else:
            os.environ["STAGE_PROFILE"] = _saved
        shutil.rmtree(tmp, ignore_errors=True)


def test_closed_loop_daemon_mode():
    """True two-process loop: independent resident host daemon.

    The daemon process uses default_llm_call (env-configured), so in this
    offline unit test the proposer falls back to the deterministic compiled
    baseline (v2.3 template path, 0 LLM calls at execution). The point of
    the test is the architecture: daemon launched, children executed
    one-by-one, outcomes visible, stop marker honored.
    """
    tmp = Path(tempfile.mkdtemp(prefix="v2_loop_daemon_"))
    # The daemon is a child process that uses default_llm_call; on a host
    # with a live OPENAI_API_KEY it would call the real LLM and get 1.0.
    # Clear the LLM env so the child deterministically falls back to the
    # stdlib majority baseline (0.5 on the 2-row stub).
    _saved_env = {k: os.environ.get(k) for k in
                  ("OPENAI_API_KEY", "OPENAI_BASE_URL", "LLM_MODEL")}
    for _k in _saved_env:
        os.environ.pop(_k, None)
    try:
        _setup(tmp)
        loop = ClosedLoop(
            _args(tmp, max_rounds=1, trial_budget=2, host_daemon=True),
            llm_call_fn=_stub_llm(0.85))
        result = loop.run()
        check("daemon loop completed", result["status"] == "completed",
              str(result))
        check("daemon two children", result["total_trials"] == 2,
              str(result["total_trials"]))
        check("daemon best metric computed (compiled baseline)",
              result["best_metric"] is not None, str(result["best_metric"]))
        check("daemon submission produced", result["submission_exists"] is True)
        daemon_logs = list((tmp / "state").glob("host_daemon_*.log"))
        check("daemon log written", len(daemon_logs) == 1, str(daemon_logs))
        if daemon_logs:
            text = daemon_logs[0].read_text(encoding="utf-8")
            check("daemon served children",
                  "DAEMON_START" in text and "DAEMON_DONE receipts=2" in text,
                  text[-400:])
        # outcomes must be visible to the agent (2 verified outcomes)
        outcomes = list((tmp / "state" / "protocol" / "outcomes_visible")\
                        .glob("outcome_*.json"))
        check("daemon outcomes visible", len(outcomes) == 2, str(len(outcomes)))
    finally:
        for _k, _v in _saved_env.items():
            if _v is None:
                os.environ.pop(_k, None)
            else:
                os.environ[_k] = _v
        shutil.rmtree(tmp, ignore_errors=True)


def test_mode_guard_entry():
    import contextlib
    import io as _io
    old = os.environ.get("PACT_STAGE4_SELF_EVOLUTION")
    os.environ["PACT_STAGE4_SELF_EVOLUTION"] = "1"
    try:
        with contextlib.redirect_stderr(_io.StringIO()):
            rc = main(["--competition", "guard", "--max-rounds", "1"])
    finally:
        if old is None:
            os.environ.pop("PACT_STAGE4_SELF_EVOLUTION", None)
        else:
            os.environ["PACT_STAGE4_SELF_EVOLUTION"] = old
    check("entry rejects self-evolution=1", rc == 2, "rc=%s" % rc)

def test_absorb_receipt_ignores_rejected_metrics():
    """A rejected trial (rc!=0) must never move the
    loop's best pointer nor count as stagnation (fallback baseline metric)."""
    import types
    loop = object.__new__(ClosedLoop)
    loop.total_trials = 0
    loop.failed_trials = 0
    loop.best_metric = None
    loop.best_receipt_id = ""
    loop.stagnation_count = 0
    loop._log = lambda msg: None
    loop._is_better = (lambda c, b: b is None or c > b + 0.01)

    rejected = types.SimpleNamespace(
        receipt_id="r1", verdict="failure", metric=0.975, returncode=-3)
    loop._absorb_receipt(rejected)
    check("rejected trial does not set best",
          loop.best_metric is None, str(loop.best_metric))
    check("rejected trial does not count stagnation",
          loop.stagnation_count == 0, str(loop.stagnation_count))
    check("rejected trial counted as failed",
          loop.failed_trials == 1, str(loop.failed_trials))

    verified = types.SimpleNamespace(
        receipt_id="r2", verdict="success", metric=0.9, returncode=0)
    loop._absorb_receipt(verified)
    check("verified trial sets best",
          loop.best_metric == 0.9, str(loop.best_metric))
    check("verified trial resets stagnation",
          loop.stagnation_count == 0, str(loop.stagnation_count))

    stagnant = types.SimpleNamespace(
        receipt_id="r3", verdict="stagnant", metric=0.9, returncode=0)
    loop._absorb_receipt(stagnant)
    check("verified stagnant counts stagnation",
          loop.stagnation_count == 1, str(loop.stagnation_count))



def test_absorb_receipt_saves_incumbent_asset():
    """NEW BEST persists the verified code as a round-continuity asset:
    incumbent_best.json + incumbent/best_code_<round>.py, and the evidence
    summary exposes the asset path to HERA (HERA still chooses the method)."""
    import types
    tmp = Path(tempfile.mkdtemp(prefix="v2_loop_incumbent_"))
    try:
        state = tmp / "state"
        bus = FileBus(state)
        loop = object.__new__(ClosedLoop)
        loop.state_dir = state
        loop.bus = bus
        loop.competition = "stub_comp"
        loop.task_prompt = "Stub binary classification"
        loop.round_num = 1
        loop.total_trials = 0
        loop.failed_trials = 0
        loop.best_metric = None
        loop.best_receipt_id = ""
        loop.stagnation_count = 0
        loop._log = lambda msg: None
        loop._is_better = (lambda c, b: b is None or c > b + 0.01)
        loop.ledger = PactLedger(state)
        loop.memory = types.SimpleNamespace(
            relevant_strategies=lambda t, top_k=3: [],
            cross_task_knowledge=lambda c, top_k=5: [])
        loop.last_interpretation = None
        loop.metric_spec = {
            "metric_name": "accuracy",
            "metric_direction": "higher_is_better",
            "metric_alignment": "exact",
            "metric_label": "accuracy",
            "metric_params": {},
        }

        spec_id = "spec_abc123"
        code = "print('best model')\n"
        bus.ws_code.mkdir(parents=True, exist_ok=True)
        (bus.ws_code / ("trial_" + spec_id + ".py")).write_text(code, encoding="utf-8")
        receipt = TrialReceipt(
            receipt_id="r1", spec_id=spec_id, competition="stub_comp",
            round_num=1, returncode=0, stdout="", stderr="",
            metric=0.9, metric_name="accuracy", verdict="success",
            evidence="Improved", code_hash="sha256:x", verified=True)
        loop._absorb_receipt(receipt)
        check("incumbent json written",
              (state / "incumbent_best.json").is_file())
        data = json.loads((state / "incumbent_best.json").read_text(encoding="utf-8"))
        check("incumbent points at best code",
              data.get("metric") == 0.9
              and data.get("round_num") == 1
              and data.get("code_path")
              and Path(data["code_path"]).is_file(), str(data))
        saved = Path(data["code_path"])
        check("incumbent code content preserved",
              saved.read_text(encoding="utf-8") == code)
        evidence = loop._evidence_summary(None)
        check("evidence exposes incumbent asset",
              "Incumbent best code" in evidence and "best_code_01.py" in evidence,
              evidence[:300])
        # round 2 with a worse metric must NOT overwrite the incumbent
        loop.round_num = 2
        worse = TrialReceipt(
            receipt_id="r2", spec_id="spec_other", competition="stub_comp",
            round_num=2, returncode=0, stdout="", stderr="",
            metric=0.89, metric_name="accuracy", verdict="stagnant",
            evidence="metric", code_hash="sha256:y", verified=True)
        loop._absorb_receipt(worse)
        data2 = json.loads((state / "incumbent_best.json").read_text(encoding="utf-8"))
        check("worse round does not replace incumbent",
              data2.get("metric") == 0.9, str(data2))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_host_materialize_passes_reference_code():
    """HostSupervisorService hands the previous round's verified code to the
    implementer as a continuity asset (reference_code + meta), so the LLM can
    extend it instead of rewriting from baseline. The platform never forces a
    method - it only makes the asset available."""
    import types
    tmp = Path(tempfile.mkdtemp(prefix="v2_loop_refhost_"))
    try:
        state = tmp / "state"
        bus = FileBus(state)
        inc_dir = state / "incumbent"
        inc_dir.mkdir(parents=True)
        prev_code = "import numpy as np\nprint('prev best')\n"
        (inc_dir / "best_code_01.py").write_text(prev_code, encoding="utf-8")
        (state / "incumbent_best.json").write_text(json.dumps({
            "schema_version": "v2_incumbent_v1",
            "competition": "stub_comp", "round_num": 1,
            "metric": 0.9, "branch_id": "baseline",
            "code_path": str(inc_dir / "best_code_01.py"),
            "code_hash": "sha256:x",
        }, ensure_ascii=False), encoding="utf-8")

        calls = {}
        class _StubImplementer:
            def implement(self, *args, **kwargs):
                calls.update(kwargs)
                return "print('new code')\n"

        executor = types.SimpleNamespace(work_dir=str(tmp / "work"))
        host = HostSupervisorService(
            bus=bus, executor=executor, bundler=None, evaluator=None,
            promotion=None, implementer=_StubImplementer(), guards=None,
            ledger=None, competition="stub_comp",
            max_budget_seconds=60, data_dir=str(tmp / "data"),
            sample_path="", state_dir=state)
        plan = ResearchPlan(round_num=2, hypothesis="Improve",
                            approach_type="exploit")
        profile = AnalysisProfile(competition="stub_comp",
                                  task_type="classification")
        grant = {"grant_id": "g1", "selected_branch_id": "baseline",
                 "mutation_axis": "hyperparameter", "trial_budget": 2,
                 "competition": "stub_comp", "task_prompt": "t"}
        proposal = {"proposal_id": "p1", "grant_id": "g1",
                    "child_index": 1, "mutation_axis": "hyperparameter"}
        spec = host._materialize_spec(proposal, grant, profile, plan)
        check("implementer received reference code",
              calls.get("reference_code") == prev_code,
              str(calls.get("reference_code"))[:200])
        check("implementer received reference meta",
              calls.get("reference_meta", {}).get("metric") == 0.9,
              str(calls.get("reference_meta")))
        check("new code still sealed into spec",
              spec.code == "print('new code')\n"
              and spec.code_hash.startswith("sha256:"))
        # without state_dir the feature degrades off
        calls2 = {}
        class _StubImplementer2:
            def implement(self, *args, **kwargs):
                calls2.update(kwargs)
                return "print('new code')\n"
        host2 = HostSupervisorService(
            bus=bus, executor=executor, bundler=None, evaluator=None,
            promotion=None, implementer=_StubImplementer2(), guards=None,
            ledger=None, competition="stub_comp",
            max_budget_seconds=60, data_dir=str(tmp / "data"))
        host2._materialize_spec(proposal, grant, profile, plan)
        check("reference off without state_dir",
              "reference_code" not in calls2, str(calls2))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_round2_inherits_round1_best_code():
    """End-to-end continuity (v2.3 template path): round 1's NEW BEST is
    persisted as the incumbent asset; round 2 executes through the
    compiler (0 LLM codegen calls) and the final evidence still references
    the incumbent asset (the method decision stays with HERA)."""
    tmp = Path(tempfile.mkdtemp(prefix="v2_loop_inherit_"))
    try:
        _setup(tmp)
        calls = {"with_ref": 0}
        def stub(prompt):
            if "Return a JSON plan" in prompt:
                return json.dumps({
                    "hypothesis": "Stub hypothesis",
                    "approach_type": "explore",
                    "expected_improvement": "baseline",
                    "risk": "Low",
                    "method_detail": {"model": "random_forest", "features": "all"},
                    "max_budget_seconds": 60,
                })
            if "Write a COMPLETE Python script" in prompt:
                if "PREVIOUS BEST CODE" in prompt:
                    calls["with_ref"] += 1
                    metric = 0.95
                    hits = 19
                else:
                    metric = 0.85
                    hits = 17
                return ("import csv\n"
                        "with open('submission.csv', 'w', newline='') as f:\n"
                        "    w = csv.writer(f)\n"
                        "    w.writerow(['Id', 'Prediction'])\n"
                        "    for i in range(10):\n"
                        "        w.writerow([i, 0])\n"
                        "with open('oof.csv', 'w', newline='') as f:\n"
                        "    w = csv.writer(f)\n"
                        "    w.writerow(['true', 'pred'])\n"
                        "    for i in range(20):\n"
                        "        w.writerow([1 if i < %d else 0, 1])\n"
                        "print('accuracy: %.4f')\n" % (hits, metric))
            return "{}"
        loop = ClosedLoop(_args(tmp, max_rounds=2, trial_budget=1),
                          llm_call_fn=stub)
        result = loop.run()
        check("inherit loop completed", result["status"] == "completed")
        check("inherit loop best metric computed",
              result["best_metric"] is not None, str(result["best_metric"]))
        state = tmp / "state"
        inc_json = state / "incumbent_best.json"
        check("incumbent json exists", inc_json.is_file())
        data = json.loads(inc_json.read_text(encoding="utf-8"))
        check("incumbent metric matches best",
              data.get("metric") == result["best_metric"], str(data))
        check("incumbent code file exists",
              data.get("code_path") and Path(data["code_path"]).is_file())
        check("round2 template path: zero LLM codegen calls",
              calls["with_ref"] == 0, str(calls))
        evidence = loop._evidence_summary(None)
        check("final evidence references incumbent",
              "Incumbent best code" in evidence, evidence[:200])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)




def test_restart_scientific_state_recovery():
    """v2.2.1-rc2: a restarted loop restores best metric / round / trials
    from the certified promotion record + ledger + incumbent, so a
    regression after restart can never overwrite the best code."""
    tmp = Path(tempfile.mkdtemp(prefix="v2_restore_"))
    try:
        _setup(tmp)
        loop1 = ClosedLoop(_args(tmp, max_rounds=1), llm_call_fn=_stub_llm(0.85))
        result1 = loop1.run()
        _m1 = result1["best_metric"]
        check("first run best computed", _m1 is not None, str(_m1))
        # restart on the SAME state dir; the compiled baseline is
        # deterministic so the metric must be identical (never worse)
        loop2 = ClosedLoop(_args(tmp, max_rounds=2), llm_call_fn=_stub_llm(0.80))
        check("restart restores best from certified record",
              loop2.best_metric == _m1, str(loop2.best_metric))
        check("restart restores round", loop2.round_num >= 1,
              str(loop2.round_num))
        check("restart restores trials", loop2.total_trials >= 1,
              str(loop2.total_trials))
        result2 = loop2.run()
        check("regression does not replace best",
              result2["best_metric"] == _m1, str(result2["best_metric"]))
        inc = json.loads((tmp / "state" / "incumbent_best.json")
                         .read_text(encoding="utf-8"))
        check("incumbent asset still best", inc.get("metric") == _m1,
              str(inc))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_f0_requires_two_samples():
    """v2.2.1-rc2: F0 median is computed only after >=2 successful samples,
    accumulated across grants and persisted with receipt ids."""
    tmp = Path(tempfile.mkdtemp(prefix="v2_f0_"))
    _saved_gpu = os.environ.get("V2_GPU_MEM_MB")
    os.environ["V2_GPU_MEM_MB"] = "40960"
    try:
        _setup(tmp)
        loop = ClosedLoop(_args(tmp, max_rounds=1), llm_call_fn=_stub_llm(0.85))
        profile = loop.analyzer.profile(loop.competition)
        r1 = TrialReceipt(receipt_id="t1", spec_id="s1",
                          competition=loop.competition, round_num=1,
                          returncode=0, metric=0.8, verdict="success",
                          wall_clock_seconds=17.0)
        loop._maybe_calibrate_f0([r1], profile)
        f0 = loop.f0_calibration or {}
        check("single sample -> no median", f0.get("median_seconds") is None,
              str(f0))
        check("single sample persisted",
              len(f0.get("samples_seconds") or []) == 1, str(f0))
        r2 = TrialReceipt(receipt_id="t2", spec_id="s2",
                          competition=loop.competition, round_num=1,
                          returncode=0, metric=0.8, verdict="success",
                          wall_clock_seconds=25.0)
        loop._maybe_calibrate_f0([r2], profile)
        f0 = loop.f0_calibration or {}
        check("two samples -> median 21.0",
              f0.get("median_seconds") == 21.0, str(f0))
        check("samples recorded",
              len(f0.get("samples_seconds") or []) == 2, str(f0))
        check("receipt ids recorded",
              set(f0.get("sample_receipt_ids") or []) == {"t1", "t2"},
              str(f0.get("sample_receipt_ids")))
        # restart: calibration is loaded, no re-sampling
        loop2 = ClosedLoop(_args(tmp, max_rounds=1), llm_call_fn=_stub_llm(0.85))
        check("restart loads calibration",
              (loop2.f0_calibration or {}).get("median_seconds") == 21.0,
              str(loop2.f0_calibration))
    finally:
        if _saved_gpu is None:
            os.environ.pop("V2_GPU_MEM_MB", None)
        else:
            os.environ["V2_GPU_MEM_MB"] = _saved_gpu
        shutil.rmtree(tmp, ignore_errors=True)


def test_f0_cache_profile_mismatch_discards():
    """v2.2.1-rc2: an F0 saved with a different cached-weight whitelist is
    discarded once the current whitelist is known; an unknown whitelist
    (fresh start before preflight) still accepts it."""
    tmp = Path(tempfile.mkdtemp(prefix="v2_f0_cache_"))
    _saved_gpu = os.environ.get("V2_GPU_MEM_MB")
    os.environ["V2_GPU_MEM_MB"] = "40960"
    try:
        _setup(tmp)
        loop = ClosedLoop(_args(tmp, max_rounds=1), llm_call_fn=_stub_llm(0.85))
        profile = loop.analyzer.profile(loop.competition)
        loop._cached_weights_seen = ["resnet18"]
        r1 = TrialReceipt(receipt_id="t1", spec_id="s1",
                          competition=loop.competition, round_num=1,
                          returncode=0, metric=0.8, verdict="success",
                          wall_clock_seconds=17.0)
        r2 = TrialReceipt(receipt_id="t2", spec_id="s2",
                          competition=loop.competition, round_num=1,
                          returncode=0, metric=0.8, verdict="success",
                          wall_clock_seconds=21.0)
        loop._maybe_calibrate_f0([r1, r2], profile)
        f0 = loop.f0_calibration or {}
        check("f0 saved with cache profile",
              f0.get("median_seconds") == 19.0, str(f0))
        check("cache profile recorded",
              f0.get("cache_profile") == ["resnet18"],
              str(f0.get("cache_profile")))
        # restart with a DIFFERENT known whitelist -> discarded
        loop2 = ClosedLoop(_args(tmp, max_rounds=1), llm_call_fn=_stub_llm(0.85))
        loop2._cached_weights_seen = ["vit_base"]
        check("cache mismatch discards f0",
              loop2._load_f0(profile) == {},
              str(loop2._load_f0(profile)))
        # fresh start with unknown whitelist -> accepted
        loop3 = ClosedLoop(_args(tmp, max_rounds=1), llm_call_fn=_stub_llm(0.85))
        check("unknown cache accepts f0",
              (loop3._load_f0(profile) or {}).get("median_seconds") == 19.0,
              str(loop3._load_f0(profile)))
    finally:
        if _saved_gpu is None:
            os.environ.pop("V2_GPU_MEM_MB", None)
        else:
            os.environ["V2_GPU_MEM_MB"] = _saved_gpu
        shutil.rmtree(tmp, ignore_errors=True)



def test_lower_is_better_evidence_direction():
    """v2.2.1-rc3: evidence summary uses the direction-aware ledger helper,
    so a lower-is-better competition (logloss 0.50 -> 0.40) reports best=0.40
    instead of the max-oriented legacy helper."""
    import types
    tmp = Path(tempfile.mkdtemp(prefix="v2_logloss_ev_"))
    try:
        state = tmp / "state"
        state.mkdir(parents=True)
        ledger = PactLedger(state)
        plan = ResearchPlan(hypothesis="logloss probe")
        for rid, m in (("ll1", 0.50), ("ll2", 0.40)):
            ledger.append(TrialReceipt(
                receipt_id=rid, spec_id=rid, competition="stub_logloss",
                round_num=1, returncode=0, metric=m, metric_name="logloss",
                verdict="success", verified=True), plan)
        loop = object.__new__(ClosedLoop)
        loop.state_dir = state
        loop.competition = "stub_logloss"
        loop.task_prompt = "Logloss stub"
        loop.ledger = ledger
        loop.metric_spec = {
            "metric_name": "logloss",
            "metric_direction": "lower_is_better",
            "metric_alignment": "lower",
            "metric_label": "logloss",
            "metric_params": {},
        }
        loop.memory = types.SimpleNamespace(
            relevant_strategies=lambda t, top_k=3: [],
            cross_task_knowledge=lambda c, top_k=5: [])
        loop.last_interpretation = None
        loop.stage = None
        loop.guard = None
        loop.resource = {}
        check("ledger helper picks min for logloss",
              loop._ledger_best_metric() == 0.40,
              str(loop._ledger_best_metric()))
        evidence = loop._evidence_summary(None)
        check("evidence reports best=0.40",
              "Best verified metric so far: 0.4" in evidence,
              evidence[:300])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_best_child_direction_inline_loop():
    """v2.2.1-rc3: grant-internal best_child tracking is direction-aware:
    logloss 0.50 -> 0.40 -> 0.45 keeps 0.40 (never max())."""
    import types
    tmp = Path(tempfile.mkdtemp(prefix="v2_bestchild_"))
    try:
        state = tmp / "state"
        bus = FileBus(state)
        loop = object.__new__(ClosedLoop)
        loop.state_dir = state
        loop.bus = bus
        loop.competition = "stub_logloss"
        loop.task_prompt = "logloss stub"
        loop.round_num = 1
        loop.total_trials = 0
        loop.failed_trials = 0
        loop.best_metric = None
        loop.best_receipt_id = ""
        loop.stagnation_count = 0
        loop._log = lambda msg: None
        loop.metric_spec = {
            "metric_name": "logloss",
            "metric_direction": "lower_is_better",
            "metric_alignment": "lower",
            "metric_label": "logloss",
            "metric_params": {},
        }
        loop.ledger = PactLedger(state)
        loop.memory = types.SimpleNamespace(
            relevant_strategies=lambda t, top_k=3: [],
            cross_task_knowledge=lambda c, top_k=5: [])
        loop.last_interpretation = None
        loop.stage = None
        loop.guard = None
        loop.resource = {}
        loop.promotion = types.SimpleNamespace(
            certified_best=lambda: types.SimpleNamespace(
                certified_best_metric=None, certified_best_trial_id="",
                incumbent_metric=None))
        metrics = iter([0.50, 0.40, 0.45])
        seen = []

        class _Host:
            def supervise_once(self, grant, profile, plan, best_child):
                seen.append(best_child)
                m = next(metrics)
                return TrialReceipt(
                    receipt_id="c%d" % len(seen), spec_id="s%d" % len(seen),
                    competition="stub_logloss", round_num=1, returncode=0,
                    metric=m, metric_name="logloss", verdict="success",
                    verified=True)

        class _Agent:
            def feedback_view(self):
                return ""
            def propose_next(self, child_index, evidence):
                return types.SimpleNamespace(proposal_id="p%d" % child_index)

        loop.host = _Host()
        grant = types.SimpleNamespace(trial_budget=3, to_dict=lambda: {})
        ticket = types.SimpleNamespace(trial_budget=3)
        receipts = loop._run_grant_inline(
            _Agent(), grant, object(), object(), ticket)
        check("three children executed", len(receipts) == 3, str(len(receipts)))
        check("best_child handed to child2 is 0.50", seen[1] == 0.50, str(seen))
        check("best_child handed to child3 is 0.40", seen[2] == 0.40, str(seen))
        check("direction helper keeps min",
              loop._update_best_child(0.40, 0.45) == 0.40)
        check("direction helper accepts improvement",
              loop._update_best_child(0.50, 0.40) == 0.40)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print("=== V2 closed-loop tests ===\n")
    test_closed_loop_one_round()
    test_intent_authority_recomputes_children()
    test_closed_loop_no_llm_fallback()
    test_closed_loop_mlebench_sample_fallback()
    test_feedback_driven_children()
    test_closed_loop_daemon_mode()
    test_mode_guard_entry()
    test_absorb_receipt_ignores_rejected_metrics()
    test_absorb_receipt_saves_incumbent_asset()
    test_host_materialize_passes_reference_code()
    test_round2_inherits_round1_best_code()
    test_restart_scientific_state_recovery()
    test_f0_requires_two_samples()
    test_f0_cache_profile_mismatch_discards()
    test_lower_is_better_evidence_direction()
    test_best_child_direction_inline_loop()
    print("\nRESULT=" + ("PASS" if not FAILURES else "FAIL:" + ",".join(FAILURES)))
    sys.exit(0 if not FAILURES else 1)
