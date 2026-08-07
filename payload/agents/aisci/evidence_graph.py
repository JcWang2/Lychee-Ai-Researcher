"""evidence_graph.py - Cross-task causal knowledge accumulation."""
import json, os, time
from pathlib import Path
from typing import Optional, Any, Dict, List

class EvidenceGraph:
    def __init__(self, graph_dir=None):
        base = graph_dir or Path(os.environ.get("STATE_DIR", "/mnt/workspace"))
        self._nodes_path = base / "evidence_nodes.json"
        self._edges_path = base / "causal_edges.json"
        base.mkdir(parents=True, exist_ok=True)
        self._nodes = self._load_json(self._nodes_path, {})
        self._edges = self._load_json(self._edges_path, [])

    def _load_json(self, path, default):
        try:
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        except: pass
        return default

    def _save_nodes(self):
        # Convert sets to lists for JSON serialization
        clean = {}
        for k, v in self._nodes.items():
            clean[k] = dict(v)
            if isinstance(clean[k].get("tasks_applied"), set):
                clean[k]["tasks_applied"] = list(clean[k]["tasks_applied"])
        self._nodes_path.write_text(json.dumps(clean, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    def _save_edges(self):
        self._edges_path.write_text(json.dumps(self._edges, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    def record_trial_outcome(self, method_name, method_type, delta, success, task, round_num, trial_id):
        node_key = task + "::" + method_name
        if node_key not in self._nodes:
            self._nodes[node_key] = {
                "method_name": method_name, "method_type": method_type,
                "first_seen_task": task, "first_seen_round": round_num,
                "total_trials": 0, "success_count": 0, "cumulative_delta": 0.0,
                "tasks_applied": [], "trial_ids": [],
            }
        n = self._nodes[node_key]
        n["total_trials"] += 1
        n["trial_ids"].append(trial_id)
        if success:
            n["success_count"] += 1
        if delta is not None:
            n["cumulative_delta"] += delta
        if task not in n["tasks_applied"]:
            n["tasks_applied"].append(task)
        self._save_nodes()
        self._edges.append({
            "from": node_key, "to": task + "::round_" + str(round_num),
            "method_name": method_name, "task": task, "round_num": round_num,
            "trial_id": trial_id, "delta": delta, "success": success,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        self._save_edges()

    def query_methods(self, task, method_type=None, min_trials=1):
        results = []
        for key, n in self._nodes.items():
            if n["total_trials"] < min_trials: continue
            if method_type and n["method_type"] != method_type: continue
            if task not in n["tasks_applied"]: continue
            results.append({
                "key": key, "method_name": n["method_name"],
                "method_type": n["method_type"],
                "total_trials": n["total_trials"],
                "success_rate": n["success_count"] / max(n["total_trials"], 1),
                "avg_delta": n["cumulative_delta"] / max(n["total_trials"], 1),
            })
        results.sort(key=lambda x: x["success_rate"], reverse=True)
        return results

    def get_cross_task_knowledge(self, task, top_k=5):
        all_m = []
        for key, n in self._nodes.items():
            all_m.append({
                "method": n["method_name"], "type": n["method_type"],
                "total_trials": n["total_trials"],
                "success_rate": n["success_count"] / max(n["total_trials"], 1),
                "avg_delta": n["cumulative_delta"] / max(n["total_trials"], 1),
                "tasks": n["tasks_applied"],
            })
        all_m.sort(key=lambda x: x["success_rate"], reverse=True)
        lines = ["## Cross-Task Knowledge Evidence\n"]
        for m in all_m[:top_k]:
            lines.append("- **" + m["method"] + "** (" + m["type"] + "): success_rate=" +
                         "{:.0%}".format(m["success_rate"]) + " avg_delta=" +
                         "{:.4f}".format(m["avg_delta"]) + " trials=" + str(m["total_trials"]) +
                         " tasks=" + str(m["tasks"]))
        return "\n".join(lines) if len(lines) > 1 else "## Cross-Task Knowledge\n*No accumulated knowledge yet.*\n"
