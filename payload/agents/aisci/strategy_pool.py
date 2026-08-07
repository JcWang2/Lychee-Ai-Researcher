"""strategy_pool.py - Strategy discovery and retrieval."""
import json, os, time
from pathlib import Path
from typing import Optional, Any, Dict, List


class ResearchStrategy:
    """A reusable research strategy discovered from experiments."""
    def __init__(self, name: str, strategy_type: str, description: str,
                 conditions: str, template_prompt: str,
                 success_rate: float = 0.0, total_applications: int = 0,
                 source_task: str = "", discovered_round: int = 0):
        self.name = name
        self.strategy_type = strategy_type
        self.description = description
        self.conditions = conditions
        self.template_prompt = template_prompt
        self.success_rate = success_rate
        self.total_applications = total_applications
        self.source_task = source_task
        self.discovered_round = discovered_round

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__init__.__code__.co_varnames})


class StrategyPool:
    """Maintains and queries a portfolio of research strategies."""

    def __init__(self, pool_path=None):
        base = pool_path or Path(os.environ.get("STATE_DIR", "/mnt/workspace"))
        self._path = base / "strategy_pool.json"
        base.mkdir(parents=True, exist_ok=True)
        self._strategies: List[ResearchStrategy] = []
        self._load()

    def _load(self):
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text(encoding="utf-8"))
                self._strategies = [ResearchStrategy.from_dict(s) for s in data]
        except: pass

    def _save(self):
        self._path.write_text(
            json.dumps([s.to_dict() for s in self._strategies], indent=2, ensure_ascii=False),
            encoding="utf-8")

    def add_strategy(self, strategy: ResearchStrategy):
        existing = [s for s in self._strategies if s.name == strategy.name]
        if existing:
            existing[0].total_applications += 1
            existing[0].success_rate = (existing[0].success_rate + strategy.success_rate) / 2
        else:
            self._strategies.append(strategy)
        self._save()

    def get_relevant(self, task_description: str, top_k: int = 3) -> List[ResearchStrategy]:
        """Simple keyword-based retrieval. Can be upgraded to embedding-based."""
        keywords = set(task_description.lower().split())
        scored = []
        for s in self._strategies:
            desc_words = set(s.description.lower().split())
            overlap = len(keywords & desc_words)
            score = overlap * 0.7 + s.success_rate * 0.3
            scored.append((score, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:top_k]]

    def get_strategies_by_type(self, strategy_type: str) -> List[ResearchStrategy]:
        return [s for s in self._strategies if s.strategy_type == strategy_type]

    def extract_strategy(self, hypothesis: str, approach_type: str,
                          verdict: str, metric_delta: float,
                          task_description: str, round_num: int) -> Optional[ResearchStrategy]:
        """Extract and store a strategy from a successful round."""
        if verdict != "success" or metric_delta <= 0:
            return None
        strategy = ResearchStrategy(
            name=f"{approach_type}_{round_num}_{int(time.time())}",
            strategy_type=approach_type,
            description=f"{approach_type} on {task_description[:50]} improved by {metric_delta:.4f}: {hypothesis[:100]}",
            conditions=f"Metric improvement > 0, approach_type={approach_type}",
            template_prompt=hypothesis[:200],
            success_rate=1.0,
            total_applications=1,
            source_task=task_description[:50],
            discovered_round=round_num,
        )
        self.add_strategy(strategy)
        return strategy

    def get_all(self) -> List[ResearchStrategy]:
        return self._strategies
