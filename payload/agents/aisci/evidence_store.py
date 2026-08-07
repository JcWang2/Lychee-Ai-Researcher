"""evidence_store.py - Experiment ledger."""
import json, os, time
from dataclasses import dataclass, field, asdict
from pathlib import Path

@dataclass
class ExperimentRecord:
    competition: str
    round_num: int
    trial_id: str
    hypothesis: str
    approach_type: str
    method_detail: dict
    code_path: str = ""
    code_hash: str = ""
    code_snippet: str = ""
    returncode: int = 0
    metric: float = None
    metric_name: str = "accuracy"
    verdict: str = "unknown"
    evidence: str = ""
    blind_spots: str = ""
    causal_attribution: str = ""
    next_suggestion: str = ""
    stderr_snippet: str = ""
    wall_clock_seconds: float = 0.0
    parent_trial_id: str = ""
    submission_exists: bool = False
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

class ExperimentLedger:
    def __init__(self, ledger_path=None):
        base = ledger_path or Path(os.environ.get("STATE_DIR", "/mnt/workspace"))
        self._path = base / "experiment_ledger.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._cache = []
    def append(self, record):
        self._cache.append(record)
        self._flush()
    def _flush(self):
        try:
            with self._path.open("a", encoding="utf-8") as f:
                for rec in self._cache:
                    d = asdict(rec)
                    d["method_detail"] = json.dumps(d.get("method_detail", {}), ensure_ascii=False)
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
            self._cache.clear()
        except OSError:
            pass
    def load_all(self):
        records = []
        if self._path.exists():
            for line in self._path.read_text(encoding="utf-8").strip().split("\n"):
                line = line.strip()
                if line:
                    try:
                        d = json.loads(line)
                        if isinstance(d.get("method_detail"), str):
                            try: d["method_detail"] = json.loads(d["method_detail"])
                            except: pass
                        records.append(d)
                    except:
                        pass
        return records
    def get_recent_rounds(self, competition, n=3):
        return [r for r in self.load_all() if r.get("competition") == competition][-n:]
    def count_stagnation(self, competition, window=3):
        recent = self.get_recent_rounds(competition, n=window)
        metrics = [r.get("metric") for r in recent
                   if r.get("metric") is not None and r.get("returncode") == 0]
        if len(metrics) < 2: return 0
        return len(metrics) if max(metrics) <= max(metrics[:-1]) else 0
    def get_best_metric(self, competition):
        vals = [r["metric"] for r in self.load_all()
                if r.get("competition") == competition
                and r.get("metric") is not None and r.get("returncode") == 0]
        return max(vals) if vals else None
    def get_trials_for_competition(self, competition):
        return [r for r in self.load_all() if r.get("competition") == competition]
