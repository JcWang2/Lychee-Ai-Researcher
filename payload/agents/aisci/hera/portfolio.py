# -*- coding: utf-8 -*-
"""hera/portfolio.py - Method Portfolio: a HERA-owned, evolving method space.

The portfolio is the outer loop's method space. HERA (research decision
authority) selects ONE branch and ONE mutation axis per PrioritizationTicket
AND may WRITE new branches into the portfolio - the method space is not a
fixed design: it grows as HERA discovers directions worth trying.

Branches are validated (schema + allowed axes) before they are accepted and
persisted, so LLM creativity cannot corrupt the space. The portfolio is
persisted as portfolio.json under the state dir and reloaded on every round,
so exploration accumulates across rounds (and across restarts when the same
state dir / PORTFOLIO_FILE is reused).
"""
import hashlib
import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from v2_contracts import (IMAGE_FILE_MODALITY_THRESHOLD, canonical_hash,
                          now_iso)

ALLOWED_AXES = {
    "hyperparameter", "feature", "model", "preprocessing",
    "ensemble", "data", "architecture",
}
MAX_BRANCHES = 24

# ---- v2.2: competition-agnostic resource profiling ----
# Resource limits are DERIVED from measured profile signals (task modality,
# train rows, image dimensions, class count, feature dims) plus environment
# signals (available GPU memory, cached-weight whitelist, F0 runtime
# calibration). Competition NAMES are never used in resource derivation;
# they may appear only as optional experience priors (PRIORS), never in the
# control logic.

RESOURCE_KEYS = (
    "max_budget_seconds", "min_budget_seconds", "image_size_max",
    "epochs_min", "epochs_max",
    "max_folds", "train_rows_cap", "batch_hint", "model_scale_ceiling",
    "t_est_seconds", "pretrained_policy", "derived_from",
)

VALID_INTENTS = frozenset({
    "feasibility", "repair", "cheap_probe", "local_exploitation",
    "expensive_structural", "confirmation", "final_training",
})

# research intent -> allowed child-trial range (platform whitelist only;
# the intent itself is chosen by HERA)
INTENT_CHILD_RANGES = {
    "feasibility": (1, 2),
    "repair": (1, 2),
    "cheap_probe": (2, 4),
    "local_exploitation": (2, 3),
    "expensive_structural": (1, 2),
    "confirmation": (2, 3),
    "final_training": (1, 1),
}
_INTENT_CHILD_DEFAULT = {
    "feasibility": 1, "repair": 2, "cheap_probe": 3,
    "local_exploitation": 3, "expensive_structural": 1,
    "confirmation": 2, "final_training": 1,
}
INTENT_TIME_FACTOR = {
    "feasibility": 0.5, "repair": 0.5, "cheap_probe": 0.5,
    "local_exploitation": 1.0, "expensive_structural": 1.5,
    "confirmation": 1.0, "final_training": 2.0,
}

# Optional experience priors only (e.g. known leaderboard tops). NOT used
# by ResourceProfiler: they never control resources, budgets or methods.
PRIORS: Dict[str, Dict[str, float]] = {}


def resolve_children(intent: str, requested: Optional[int] = None) -> int:
    """Map a HERA-chosen research intent to an allowed child-trial count.

    The platform only WHITELISTS the range per intent; HERA may optionally
    request a concrete count, which is clamped into the intent's range.
    """
    intent = str(intent or "").strip()
    if intent not in INTENT_CHILD_RANGES:
        try:
            return max(1, int(requested))
        except (TypeError, ValueError):
            return 3
    lo, hi = INTENT_CHILD_RANGES[intent]
    default = _INTENT_CHILD_DEFAULT[intent]
    try:
        req = int(requested)
    except (TypeError, ValueError):
        req = 0
    if req < lo or req > hi:
        req = default
    return max(1, req)


def intent_time_factor(intent: str) -> float:
    return float(INTENT_TIME_FACTOR.get(str(intent or ""), 1.0))


def estimate_grant_cost(resource: dict, children: int, intent: str) -> float:
    """Best-effort wall-clock estimate for one grant (before committing).

    est = sum(children) * T_est(intent, profile) where T_est comes from the
    F0 calibration when available, else from the modality/rows/pixels
    derivation (see ResourceProfiler).
    """
    res = dict(resource or {})
    t_est = int(res.get("t_est_seconds") or res.get("max_budget_seconds") or 1800)
    return max(1.0, float(children)) * t_est * intent_time_factor(intent)


class ResourceProfiler:
    """Derive per-task trial resources from generic measured signals.

    Signals (all measured, never guessed):
      profile.train_rows / task_type / modality
      profile.image_width/height/channels, feature_dim, n_classes
      gpu_memory_mb        (nvidia-smi or V2_GPU_MEM_MB)
      cached_weights       (preflight whitelist)
      f0_calibration       (first-grant measured trial seconds)
    """

    def __init__(self, gpu_memory_mb: int = 0,
                 cached_weights: Optional[list] = None,
                 f0_calibration: Optional[dict] = None,
                 pretrained_policy: str = "cache"):
        self.gpu_memory_mb = max(0, int(gpu_memory_mb or 0))
        self.cached_weights = [str(w) for w in (cached_weights or [])]
        self.f0_calibration = dict(f0_calibration or {})
        policy = str(pretrained_policy or "cache").strip().lower()
        self.pretrained_policy = policy if policy in ("cache", "scratch", "auto") else "cache"

    _PROFILE_MODALITIES = ("image", "image_pixel", "text", "tabular",
                          "mixed", "unknown")

    @staticmethod
    def _modality(profile) -> str:
        m = str(getattr(profile, "modality", "") or "").strip().lower()
        if m in ResourceProfiler._PROFILE_MODALITIES:
            return m
        if int(getattr(profile, "image_width", 0) or 0) or int(
                getattr(profile, "image_height", 0) or 0):
            return "image"
        if int(getattr(profile, "image_file_count", 0) or 0) >= \
                IMAGE_FILE_MODALITY_THRESHOLD:
            return "image"
        cols = [str(c).lower() for c in (getattr(profile, "feature_columns", None) or [])]
        text_hint = ("text", "tweet", "sentence", "question", "answer",
                     "comment", "abstract", "title", "article", "review",
                     "message", "lyrics", "description")
        if any(any(k in c for k in text_hint) for c in cols):
            return "text"
        return "tabular"

    def _image_size_cap(self, rows: int) -> Optional[int]:
        """Memory-driven resize cap: sqrt(gpu_bytes / (rows * 32 bytes/px)).

        Rough small-CNN bound; clamped to a sane [96, 384] range. Unknown GPU
        -> conservative 192.
        """
        if self.gpu_memory_mb <= 0:
            return 192
        gpu_bytes = float(self.gpu_memory_mb) * 1024.0 * 1024.0
        est = int((gpu_bytes / (max(1, rows) * 32.0)) ** 0.5)
        return max(96, min(384, est))

    # v2.5.0 declarative modality-resource tables (data, not branches).
    # A modality value lives ONLY in these tables; derive() never compares a
    # modality/metric name inside a branch to pick a strategy.
    _IMAGE_MODALITIES = ("image", "image_pixel")
    _FEATURE_FACTOR_MODALITIES = ("tabular",)
    _PRETRAINED_BUDGET_MODALITIES = ("image",)
    _BATCH_FIXED = {"text": 32}
    _BATCH_GPU_TABLE = ((24000, 64), (12000, 32), (0, 16))
    _MODALITY_BASE_RESOURCE = {
        "image": (1500, 3, 8),
        "text": (1200, 2, 6),
    }
    _EPOCHS_BONUS_FIXED = {"text": (2, 30)}
    _EPOCHS_BONUS_GPU = {"image": 12}

    def derive(self, profile=None) -> dict:
        """Deterministic, competition-name-free resource profile."""
        rows = max(1, int(getattr(profile, "train_rows", 0) or 0))
        modality = self._modality(profile)
        is_image = modality in self._IMAGE_MODALITIES
        base_seconds, epochs_min, epochs_max = self._MODALITY_BASE_RESOURCE.get(
            modality, (900, 2, 6))

        rows_factor = max(0.4, min(4.0, (rows / 10000.0) ** 0.7))

        # Measured signals now actively consumed (v2.2.1):
        #   n_classes      -> budget / folds / model-scale ceiling
        #   feature_dim    -> t_est / budget (tabular)
        #   actual pixels  -> budget / t_est (image native resolution)
        #   cached weights -> budget / epochs / model-scale ceiling
        n_classes = max(0, int(getattr(profile, "n_classes", 0) or 0))
        feature_dim = max(0, int(getattr(profile, "feature_dim", 0) or 0))
        class_factor = 1.0
        if n_classes >= 2:
            class_factor = 1.0 + 0.12 * math.log2(
                max(1.0, float(n_classes) / 2.0))
        class_factor = max(0.75, min(1.5, class_factor))
        feature_factor = 1.0
        if modality in self._FEATURE_FACTOR_MODALITIES and feature_dim > 0:
            feature_factor = 1.0 + 0.1 * math.log2(
                max(1.0, float(feature_dim) / 24.0))
            feature_factor = max(0.8, min(1.5, feature_factor))
        cached_factor = 1.0
        if self.cached_weights:
            # v2.3.9: on a large GPU, cached timm/HF weights let image
            # fine-tunes scale much further; raise the factor cap for image
            # modality only (generic signal, never a task name).
            _cap = 1.45 if (is_image and self.gpu_memory_mb >= 24000) else 1.3
            cached_factor = min(1.0 + 0.05 * len(self.cached_weights), _cap)

        image_size_max = None
        pixels_factor = 1.0
        cur_pixels = 1
        cur_pixels_factor = 1.0
        if is_image:
            image_size_max = self._image_size_cap(rows)
            if image_size_max:
                pixels_factor = max(0.25, min(4.0,
                                              (image_size_max / 192.0) ** 2))
            if int(getattr(profile, "image_width", 0) or 0) and int(
                    getattr(profile, "image_height", 0) or 0):
                cur_pixels = max(1, int(profile.image_width)
                                 * int(profile.image_height))
            # native image resolution: larger source pixels need more budget
            cur_pixels_factor = max(0.5, min(3.0,
                                             (cur_pixels / 36864.0) ** 0.25))
        derive_factor = class_factor * feature_factor * cur_pixels_factor

        # F0 calibration: measured first-grant seconds scaled to current rows
        # (measured truth wins for t_est; derived factors still scale the
        # budget headroom, folds and model scale).
        f0_seconds = float(self.f0_calibration.get("f0_seconds") or 0)
        t_est = base_seconds * rows_factor * pixels_factor * derive_factor
        if f0_seconds > 0:
            cal_rows = max(1, int(self.f0_calibration.get("train_rows") or rows))
            t_est = f0_seconds * max(0.4, min(4.0,
                                              (rows / float(cal_rows)) ** 0.7))
        t_est = int(max(300, min(7200, t_est)))
        max_budget = int(max(300, min(7200,
                                      base_seconds * rows_factor * pixels_factor
                                      * derive_factor * cached_factor)))
        # v2.3.9 pretrained-aware image budget (modality-driven only):
        # cached timm/HF weights + a large GPU make REAL fine-tunes
        # affordable, so raise the per-trial budget ceiling generically
        # instead of capping every image grant at probe-sized runs.
        if modality in self._PRETRAINED_BUDGET_MODALITIES and \
                self.cached_weights and self.gpu_memory_mb >= 24000:
            max_budget = int(max(300, min(7200, max_budget * 1.35)))

        # folds: small data -> more folds; huge data -> fewer; many classes
        # -> less per-class signal per row, so cap the fold count
        if rows <= 2000:
            max_folds = 4
        elif rows <= 10000:
            max_folds = 3
        else:
            max_folds = 2
        if n_classes >= 100:
            max_folds = min(max_folds, 2)
        elif n_classes >= 50:
            max_folds = min(max_folds, 3)

        # row cap: keep trials fast on very large datasets
        if rows <= 20000:
            train_rows_cap = None
        else:
            train_rows_cap = 20000 if is_image else 50000

        # batch hint from GPU memory (conservative)
        if is_image:
            batch_hint = 64
            for _th, _b in self._BATCH_GPU_TABLE:
                if self.gpu_memory_mb >= _th:
                    batch_hint = _b
                    break
        elif modality in self._BATCH_FIXED:
            batch_hint = self._BATCH_FIXED[modality]
        else:
            batch_hint = 256 if rows <= 20000 else 512

        if self.gpu_memory_mb >= 24000:
            scale = "large"
        elif self.gpu_memory_mb >= 12000:
            scale = "medium"
        elif self.gpu_memory_mb > 0:
            scale = "small"
        else:
            scale = "medium" if is_image else "any"

        def _bump_scale(current: str) -> str:
            return {"any": "small", "small": "medium",
                    "medium": "large"}.get(current, "large")

        # class count: more classes need more capacity; cached weights make
        # deep pretrained models actually runnable -> raise the ceiling
        if n_classes >= 100:
            scale = _bump_scale(_bump_scale(scale))
        elif n_classes >= 20:
            scale = _bump_scale(scale)
        if self.cached_weights and scale in ("any", "small"):
            scale = "medium"
        if self.cached_weights and modality in self._EPOCHS_BONUS_FIXED:
            _bonus, _cap = self._EPOCHS_BONUS_FIXED[modality]
            epochs_max = min(_cap, epochs_max + _bonus)
        if self.cached_weights and modality in self._EPOCHS_BONUS_GPU:
            # image fine-tune templates (finetune v2 / ensemble) support
            # up to 12 epochs; cached weights make them runnable.
            _bump = 4 if self.gpu_memory_mb >= 24000 else 2
            epochs_max = min(self._EPOCHS_BONUS_GPU[modality],
                             epochs_max + _bump)

        # v2.5.4: trial-timeout floor = half the platform's own runtime
        # estimate (t_est), so an over-optimistic LLM budget can never
        # schedule a trial that is killed before the derived estimate says
        # it should finish (rc=-9 timeouts on large-row tasks). t_est
        # self-corrects via F0 calibration after the first success.
        min_budget = max(300, int(t_est * 0.5))
        min_budget = min(min_budget, max_budget)
        return {
            "max_budget_seconds": max_budget,
            "min_budget_seconds": min_budget,
            "image_size_max": image_size_max,
            "epochs_min": epochs_min,
            "epochs_max": epochs_max,
            "max_folds": max_folds,
            "train_rows_cap": train_rows_cap,
            "batch_hint": batch_hint,
            "model_scale_ceiling": scale,
            "t_est_seconds": t_est,
            "pretrained_policy": self.pretrained_policy,
            "derived_from": {
                "modality": modality,
                "rows": rows,
                "n_classes": n_classes,
                "feature_dim": feature_dim,
                "cur_pixels": cur_pixels if is_image else 0,
                "gpu_memory_mb": self.gpu_memory_mb,
                "cached_weights": len(self.cached_weights),
                "f0_seconds": round(f0_seconds, 1) if f0_seconds else 0,
                "factors": {
                    "rows": round(rows_factor, 3),
                    "pixels_cap": round(pixels_factor, 3),
                    "classes": round(class_factor, 3),
                    "feature_dim": round(feature_factor, 3),
                    "cur_pixels": round(cur_pixels_factor, 3),
                    "cached_weights": round(cached_factor, 3),
                },
            },
        }


def _clamp_resource(res: dict) -> dict:
    """Normalize a resource profile dict; missing keys get generic defaults."""
    base = ResourceProfiler().derive(None)
    out = dict(base)
    if isinstance(res, dict):
        for k, v in res.items():
            if k == "derived_from":
                continue
            if v is None and k in RESOURCE_KEYS:
                out[k] = None
            elif k in out:
                out[k] = v
    out["max_budget_seconds"] = max(300, min(7200, int(
        out.get("max_budget_seconds") or 1800)))
    out["image_size_max"] = (None if not out.get("image_size_max")
                             else max(64, min(512, int(out["image_size_max"]))))
    out["epochs_min"] = max(1, min(20, int(out.get("epochs_min") or 3)))
    out["epochs_max"] = max(out["epochs_min"],
                            min(30, int(out.get("epochs_max") or 8)))
    out["max_folds"] = max(1, min(5, int(out.get("max_folds") or 2)))
    out["train_rows_cap"] = (None if not out.get("train_rows_cap")
                             else max(100, int(out["train_rows_cap"])))
    out["batch_hint"] = (None if not out.get("batch_hint")
                         else max(8, int(out["batch_hint"])))
    out["t_est_seconds"] = max(300, min(7200, int(
        out.get("t_est_seconds") or out["max_budget_seconds"])))
    out["min_budget_seconds"] = max(300, min(
        int(out.get("max_budget_seconds") or 7200),
        int(out.get("min_budget_seconds") or 300)))
    out["pretrained_policy"] = str(out.get("pretrained_policy") or "cache")
    return out


def resource_profile_for(competition: str,
                         task_type: str = "classification",
                         profile=None, **profiler_kwargs) -> dict:
    """Competition-agnostic resource profile (v2.2).

    The competition NAME is deliberately NOT consulted: resources derive from
    generic signals only (rows/modality/dims + GPU/cache/F0). ``competition``
    stays in the signature for backward compatibility; it may only be used by
    optional PRIORS elsewhere, never by this derivation.
    """
    if profile is None:
        import types as _types
        profile = _types.SimpleNamespace(
            competition=competition, task_type=task_type,
            train_rows=0, test_rows=0, feature_columns=[], numeric_columns=[],
            image_width=0, image_height=0, image_channels=0,
            modality="tabular", n_classes=0)
    return ResourceProfiler(**profiler_kwargs).derive(profile)


@dataclass
class PortfolioBranch:
    branch_id: str = "baseline"
    model_family: str = "random_forest"
    description: str = ""
    allowed_mutation_axes: List[str] = field(
        default_factory=lambda: ["hyperparameter", "feature", "model"])
    defaults: Dict[str, Any] = field(default_factory=dict)
    # provenance: platform_safety_seed branches are NOT agent discoveries
    origin: str = "hera"               # platform_safety_seed | hera
    scientific_discovery: bool = True  # False only for platform seeds


def validate_branch(branch: Dict[str, Any]) -> Optional[str]:
    """Return an error string if the branch dict is not usable, else None."""
    if not isinstance(branch, dict):
        return "branch is not an object"
    branch_id = str(branch.get("branch_id") or "").strip()
    if not branch_id:
        return "branch_id missing"
    if not all(c.isalnum() or c == "_" for c in branch_id):
        return "branch_id must be alphanumeric/underscore: %r" % branch_id
    axes = branch.get("allowed_mutation_axes") or []
    if not isinstance(axes, list) or not axes:
        return "allowed_mutation_axes missing"
    bad = [a for a in axes if str(a) not in ALLOWED_AXES]
    if bad:
        return "invalid axes %s (allowed: %s)" % (bad, sorted(ALLOWED_AXES))
    defaults = branch.get("defaults") or {}
    if not isinstance(defaults, dict):
        return "defaults must be an object"
    return None


class MethodPortfolio:
    """Holds candidate branches and answers branch/axis queries.

    HERA may grow the branch set at runtime via add_branch(); the portfolio
    is persisted so exploration survives across rounds and restarts.
    """

    def __init__(self, branches: List[Dict[str, Any]] = None,
                 competition: str = "unknown",
                 path: Optional[Path] = None,
                 resource_profile: Optional[dict] = None):
        self.competition = competition
        self.path = Path(path) if path else None
        self.resource_profile = resource_profile or \
            resource_profile_for(competition, "classification")
        self.branches: List[PortfolioBranch] = []
        if branches:
            _branch_fields = set(getattr(
                PortfolioBranch, "__dataclass_fields__", {}).keys())
            for b in branches:
                if not isinstance(b, dict):
                    continue
                try:
                    self.branches.append(PortfolioBranch(**{
                        k: v for k, v in b.items() if k in _branch_fields}))
                except TypeError:
                    continue
        if not self.branches:
            self.branches = [PortfolioBranch()]  # baseline fallback

    @property
    def portfolio_hash(self) -> str:
        payload = {
            "competition": self.competition,
            "branches": [asdict(b) for b in self.branches],
        }
        return canonical_hash(payload)

    def branch_ids(self) -> List[str]:
        return [b.branch_id for b in self.branches]

    def get_branch(self, branch_id: str) -> PortfolioBranch:
        for b in self.branches:
            if b.branch_id == branch_id:
                return b
        return self.branches[0]

    def allowed_axes(self, branch_id: str) -> List[str]:
        return list(self.get_branch(branch_id).allowed_mutation_axes)

    def add_branch(self, branch: Dict[str, Any]) -> str:
        """HERA writes a new branch into the method space.

        Returns "" on success, or an error string (duplicate / invalid).
        """
        err = validate_branch(branch)
        if err:
            return err
        branch_id = str(branch.get("branch_id")).strip()
        if branch_id in self.branch_ids():
            return "duplicate branch_id %r" % branch_id
        if len(self.branches) >= MAX_BRANCHES:
            return "portfolio full (%d branches)" % MAX_BRANCHES
        self.branches.append(PortfolioBranch(
            branch_id=branch_id,
            model_family=str(branch.get("model_family") or "unknown"),
            description=str(branch.get("description") or "")[:300],
            allowed_mutation_axes=[str(a) for a in branch["allowed_mutation_axes"]],
            defaults=dict(branch.get("defaults") or {}),
            origin=str(branch.get("origin") or "hera"),
            scientific_discovery=bool(branch.get("scientific_discovery", True)),
        ))
        return ""

    def to_dict(self) -> dict:
        return {
            "competition": self.competition,
            "portfolio_hash": self.portfolio_hash,
            "resource_profile": self.resource_profile,
            "branches": [asdict(b) for b in self.branches],
        }

    # ---- persistence ----
    def save(self) -> bool:
        if self.path is None:
            return False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8")
            return True
        except OSError:
            return False

    @classmethod
    def load_or_default(cls, profile, path: Optional[Path] = None,
                        seed: Optional[list] = None) -> "MethodPortfolio":
        """Load a persisted portfolio or build the seed portfolio.

        Seed = deterministic starter branches (baseline + feature_engineering)
        based on the profile; any persisted branches are merged on top so a
        restarted run keeps its discovered method space.
        """
        seed = seed if seed is not None else _seed_branches(profile)
        if path is not None and Path(path).is_file():
            try:
                data = json.loads(Path(path).read_text(encoding="utf-8"))
                stored = data.get("branches") if isinstance(data, dict) else None
                if isinstance(stored, list) and stored:
                    merged = dict((b.get("branch_id"), b) for b in seed
                                  if isinstance(b, dict) and b.get("branch_id"))
                    for b in stored:
                        if isinstance(b, dict) and b.get("branch_id"):
                            merged[b["branch_id"]] = b
                    res = data.get("resource_profile") or {}
                    if not isinstance(res, dict) or not res:
                        res = resource_profile_for(
                            getattr(profile, "competition", "unknown"),
                            getattr(profile, "task_type", "classification"))
                    return cls(list(merged.values()),
                               competition=getattr(profile, "competition", "unknown"),
                               path=path,
                               resource_profile=_clamp_resource(res))
            except (OSError, ValueError):
                pass
        return cls(seed, competition=getattr(profile, "competition", "unknown"),
                   path=path)

    @classmethod
    def default_for(cls, profile) -> "MethodPortfolio":
        """Deterministic seed portfolio from an AnalysisProfile."""
        return cls(_seed_branches(profile),
                   competition=getattr(profile, "competition", "unknown"))


# v2.5.0 declarative safety-seed model table: task_type -> model family.
# The seed baseline is a platform fallback, never a research decision;
# HERA may (and usually does) pick a different candidate entirely.
_SEED_MODEL_BY_TASK = {
    "timeseries": "lightgbm",
    "regression": "xgboost",
}


def _seed_branches(profile) -> List[Dict[str, Any]]:
    task_type = getattr(profile, "task_type", "classification")
    model = _SEED_MODEL_BY_TASK.get(task_type, "random_forest")
    return [
        {
            "branch_id": "baseline",
            "model_family": model,
            "description": "Platform safety baseline (%s)" % model,
            "allowed_mutation_axes": ["hyperparameter", "feature"],
            "defaults": {"model": model, "features": "all"},
            "origin": "platform_safety_seed",
            "scientific_discovery": False,
        },
        {
            "branch_id": "feature_engineering",
            "model_family": model,
            "description": "Engineered features + scaling",
            "allowed_mutation_axes": ["feature", "preprocessing"],
            "defaults": {"model": model, "features": "engineered"},
            "origin": "platform_safety_seed",
            "scientific_discovery": False,
        },
    ]