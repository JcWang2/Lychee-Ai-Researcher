# -*- coding: utf-8 -*-
"""pact/implementer.py - Implementation step inside PACT.

Turns a HERA ResearchPlan into candidate Python code using the LLM.
PACT's execution and verification remain 100% deterministic - LLM-written
code is not trusted until it passes PACT verification. A deterministic
fallback keeps the loop moving when the LLM is unavailable.
v2.2.1 prompt hardening:
  - code-master persona + Arbor-style working method (plan silently,
    simplest correct version, self-review before returning, surgical
    root-cause repair) for high first-try success and code quality;
  - SOTA cadence contract (10-25 min per trial, pretrained weights from the
    verified cache only) so a 24h run can push leaderboard scores; the
    run script still offers a fast 256-round mode.
    submission.csv: sample header verbatim) so verification is meaningful;
  - branch=baseline can use a deterministic stdlib template
    (V2_DETERMINISTIC_BASELINE=1) that never needs the LLM, guaranteeing
    round 1 always produces a real metric when the data contract holds.
"""
import json as _json
import os
import re
from pathlib import Path
from typing import Callable, Optional

from data_layout import DatasetLayout, DatasetLayoutError, resolve_dataset_layout
from metrics_registry import OOF_GUIDE
from v2_contracts import AnalysisProfile, ResearchPlan
from v2_llm import default_llm_call

FALLBACK_CODE = (
    "import csv, os\n"
    "with open('submission.csv', 'w', newline='') as f:\n"
    "    w = csv.writer(f)\n"
    "    w.writerow(['Id', 'Prediction'])\n"
    "    for i in range(10):\n"
    "        w.writerow([i, 0])\n"
    "print('accuracy: 0.5000')\n"
)


_PROBABILITY_METRICS = {"logloss", "weighted_logloss",
                   "mean_auc_multilabel", "kl_div"}


def _is_probability_metric(metric_info) -> bool:
    """True when the official metric needs one pred_<class> column per class."""
    return (str((metric_info or {}).get("metric_name") or "accuracy")
            in _PROBABILITY_METRICS)


def _oof_format_rule(metric_info) -> str:
    """Requirement-5 text: oof.csv column contract for THIS metric."""
    policy = ("MLE-Bench only scores submission.csv on the hidden test "
              "set, so IN-SAMPLE predictions over the full training set "
              "are allowed (a validation split or KFold is also fine for "
              "a generalization signal).")
    if _is_probability_metric(metric_info):
        return ("5. Write oof.csv in the current working directory for the "
                "OFFICIAL METRIC above: a 'true' column plus ONE "
                "'pred_<class>' probability column per class (softmax over "
                "the full label space; e.g. header "
                "'true,pred_<c1>,pred_<c2>,...' and each pred_ column holds "
                "probabilities 0..1, rows sum to 1 - never class IDs). "
                + policy + "\n")
    return ("5. Write oof.csv in the current working directory with exactly "
            "two columns: true,pred. pred is the predicted CLASS ID "
            "(integer, same space as training labels) or numeric value - "
            "never probabilities. " + policy + "\n")


def _oof_fallback_full(metric_info) -> str:
    """Requirement-16 text: crash-safe artifact fallback for THIS metric."""
    if _is_probability_metric(metric_info):
        return ("still write submission.csv (sample ids + per-class "
                "probabilities following sample_submission.csv) and oof.csv "
                "(true + one pred_<class> probability column per class; if "
                "you have no split, use the constant distribution of the "
                "most frequent class (1.0 for that class, 0.0 others) per "
                "training row; in-sample predictions are allowed)")
    return ("still write submission.csv (sample ids + most frequent class "
            "prediction) and oof.csv (true,pred; if you have no split, "
            "use the CONSTANT most-frequent class per training row; "
            "in-sample predictions are allowed)")


def _oof_review_rule(metric_info) -> str:
    """Requirement-18(d) text: self-review contract for THIS metric."""
    if _is_probability_metric(metric_info):
        return ("(d) oof.csv has one true column plus one pred_<class> "
                "probability column per class with row count matching "
                "your protocol: the full training set (in-sample, allowed) "
                "or your declared split/CV rows")
    return ("(d) oof.csv has exactly two columns true,pred with "
            "raw label values AND row count matching your protocol: the "
            "full training set (in-sample, allowed) or your declared "
            "split/CV rows")


def _oof_skeleton_note(metric_info) -> str:
    """Skeleton caveat: the accuracy-style two-column skeleton does not fit
    probability metrics; point at the right artifact shape."""
    if _is_probability_metric(metric_info):
        return ("\nNOTE for logloss-style metrics: the skeleton above is "
                "for accuracy tasks. For THIS metric, write oof.csv with "
                "'true' + pred_<class> probability columns (softmax) and "
                "submission.csv with per-class probabilities following "
                "sample_submission.csv.")
    return ""


class Implementer:
    """Writes candidate code for a plan (LLM-assisted, with fallback)."""

    def __init__(self, llm_call_fn: Optional[Callable[[str], str]] = None):
        self.llm_call = llm_call_fn or default_llm_call

    def build_code_prompt(self, plan: ResearchPlan, profile: AnalysisProfile,
                          data_dir, code_dir, submission_dir,
                          sample_path: str = "", branch: str = "",
                          proposal: Optional[dict] = None,
                          pretrained_available: Optional[list] = None,
                          metric_info: Optional[dict] = None,
                          reference_code: str = "",
                          reference_meta: Optional[dict] = None) -> str:
        competition = profile.competition if profile is not None else "unknown"
        layout_text = "Dataset layout could not be resolved before implementation."
        manifest = {}
        try:
            layout = resolve_dataset_layout(data_dir, sample_path=sample_path)
            layout_text = layout.prompt_paths()
            manifest = _layout_manifest(layout, profile)
        except DatasetLayoutError as exc:
            layout_text = "Dataset layout warning: %s" % exc
        _img_max = ((plan.method_detail or {}).get("resource_profile") or {}).get("image_size_max") or 192
        _cache_hint = "<work_dir>/data_cache/<key>"
        _cache_dirs = {}
        try:
            from pact.data_cache import cache_dir as _cache_dir
            _cache_hint = str(_cache_dir(code_dir, manifest, _img_max))
            _env_dirs = os.environ.get("V2_CACHE_DIRS", "").strip()
            if _env_dirs:
                _parsed = _json.loads(_env_dirs)
                if isinstance(_parsed, dict):
                    _cache_dirs = {int(k): str(v) for k, v in _parsed.items()
                                   if str(v).strip()}
        except Exception:  # noqa: BLE001 - hint degrades to generic text
            pass
        if _cache_dirs:
            _cache_list = ", ".join(
                "%spx@%s" % (k, _cache_dirs[k]) for k in sorted(_cache_dirs))
        else:
            _cache_list = "%s (single default size)" % _cache_hint
        _cache_rule = (
            "6e. SHARED DATA CACHE (zero-recompute; applies to ANY task "
            "when present): a prebuilt cache shared by every trial lives "
            "at %s. Available cached sizes (container env V2_CACHE_DIRS = "
            "JSON {size: dir}, when set): %s. For image tasks each cache "
            "dir holds train_X.npy/test_X.npy (uint8 HxWx3; row order = "
            "train.csv/test.csv id order) and train_ids.json/test_ids.json "
            "(id strings, same order). If a cache dir exists, LOAD it with "
            "np.load (optionally mmap_mode='r') and NEVER decode images / "
            "repeat expensive prep - decoding runs ONCE per run; never "
            "write into the cache directory; prefer the SMALLEST cached "
            "size for cheap probes and larger sizes only when the method "
            "needs the resolution\n" % (_cache_hint, _cache_list))
        prompt = (
            "ROLE: YOU ARE A CODE MASTER. Respond with the code IMMEDIATELY: do NOT reason out loud, do NOT explain, do NOT use markdown fences. First token within seconds.\n"
            "You are the world's most meticulous senior ML engineer and Kaggle "
            "competition Grandmaster. One iron rule defines you: the code you "
            "write must run correctly on its FIRST execution, inside this "
            "offline container, on the exact data contract below. You never "
            "ship broken, partial, or guessed code. You are humble about data "
            "and ruthless about correctness. You write with master discipline: "
            "plan silently, keep it simple, verify every contract, refuse to "
            "guess.\n\n"
            "WORKING METHOD (Arbor-style discipline - do this in your head "
            "before typing any code):\n"
            "  A. Read the contract: identify columns, label space, image "
            "paths, and everything that can break (NaN, dtypes, missing "
            "files, label encoding).\n"
            "  B. Choose the model that fits the data size and the trial "
            "budget (10-25 minutes; small datasets can finish faster); "
            "improve by exactly ONE safe step per iteration (the research "
            "loop runs many rounds - verified steps beat big rewrites).\n"
            "  C. Write ONE complete script with small functions and a "
            "main() guard; no dead code, no TODO comments, no placeholders.\n"
            "  D. Before returning, mentally execute the script end-to-end "
            "against the contract and fix every bug you find.\n\n"
            "Write a COMPLETE Python script (runnable on the first try) for "
            + competition + ".\n\n"
            + "Branch direction: " + str(
              (plan.method_detail or {}).get("branch_id", branch)) + " - "
              + str((plan.method_detail or {}).get(
                  "branch_description", ""))[:160] + "\n"
            + "Hypothesis: " + plan.hypothesis + "\n"
            + "Approach: " + plan.approach_type + "\n"
            + "Method: " + _json.dumps(plan.method_detail, ensure_ascii=False)[:500] + "\n"
            + "Data dir: " + str(data_dir) + "\n"
            + "Code dir: " + str(code_dir) + "\n"
            + "Submission dir: " + str(submission_dir) + "\n\n"
            + _child_feedback_section(proposal)
            + _reference_code_section(reference_code, reference_meta)
            + "EXACT DATA CONTRACT (these absolute paths are mounted inside the "
              "execution container; use them verbatim):\n"
            + _json.dumps(manifest, ensure_ascii=False, indent=2) + "\n\n"
            + "The same contract is available as data_manifest.json in the "
              "working directory and as env vars DATA_DIR, TRAIN_CSV, TEST_CSV, "
              "SAMPLE_SUBMISSION, TRAIN_IMAGES, TEST_IMAGES, TARGET_COLUMN, "
              "TASK_TYPE.\n\n"
            + _metric_section(metric_info)
            + _pretrained_section(pretrained_available)
            + (_cpu_only_section() if os.environ.get("V2_CPU_ONLY") == "1"
               else _gpu_section())
            + _resource_section(plan)
            + "Requirements (all mandatory):\n"
            + "1. Read the exact resolved train/test CSV paths above; do not "
              "assume a flat layout and never invent file names\n"
            + "2. Implement the hypothesis with the simplest model that fits "
              "and improves on the baseline; ONE focused improvement per script\n"
            + "3. Write submission.csv in the current working directory with "
              "EXACTLY the header of sample_submission.csv and its id column "
              "values, same order, no extra rows\n"
            + "4. Print final validation metric as 'accuracy: 0.xxxx' (accuracy "
              "on your own validation split, 0..1)\n"
            + _oof_format_rule(metric_info)
            + "5b. OOF rows: you may predict the FULL training set with "
              "your final model (in-sample, ALLOWED - the host recomputes "
              "the metric for internal guidance and marks it as in-sample), "
              "or use KFold/StratifiedKFold (up to 5 folds: loop folds, "
              "train on train fold, predict val fold, fill oof[val_idx]) "
              "or a single train_test_split with only held-out rows when "
              "you want a generalization signal\n"
            + "6. For image tasks read images from TRAIN_IMAGES/TEST_IMAGES "
              "(absolute dirs); build paths with os.path.join and check "
              "os.path.isfile before reading; never guess relative paths\n"
            + "6b. IMAGE DATA PIPELINE (the #1 cause of GPU-idle rc=-9 "
              "timeouts): decode and resize EVERY image ONCE before "
              "training (PIL Image.open -> convert('RGB') -> resize to "
              "<=%dpx -> numpy uint8) and keep the small arrays in memory; "
              "NEVER decode full-size images inside the epoch loop and "
              "never pair num_workers=0 with full-size decoding. "
              "DataLoader num_workers=0 is fine ONLY on pre-cached small "
              "images\n" % _img_max
            + "6c. BUDGET SELF-CHECK before returning: estimate wall time = "
              "load_time + folds x epochs x ceil(train_rows/batch_size) x "
              "0.5s (shared-GPU estimate) and keep TOTAL <= max_budget_seconds "
              "x 0.7. If over, cut to 1 fold, 2-3 epochs, subsample rows, or "
              "raise batch_size; NEVER plan 3-fold x 8-epoch on 10k+ rows "
              "inside a 1500s budget (rc=-9)\n"
            + "6d. CUDA MUST ACTUALLY BE USED: device = torch.device('cuda'); "
              "model.to(device); move every batch with .to(device); right "
              "after model.to(device) add assert next(model.parameters())"
              ".is_cuda. A script that burns CPU while the GPU stays idle "
              "will time out (rc=-9); never let training run on cpu when a "
              "GPU is available\n"
            + _cache_rule
            + "7. If test_has_labels is true in the manifest, NEVER read the "
              "target column of the test CSV; use only its id column\n"
            + "8. Use sklearn/xgboost/lightgbm/pandas/numpy/torch/torchvision "
              "when needed; all are preinstalled\n"
            + "9. Handle missing values, categorical features, and image paths "
              "properly: fill/drop NaN explicitly, coerce dtypes explicitly "
              "(astype/float()), never let pandas silently mismatch columns\n"
            + "10. Use ONE preprocessing and label-encoding pipeline for train "
              "AND test: fit LabelEncoder/StandardScaler on train only, apply "
              "to both; submission values must be the ORIGINAL label strings/"
              "ids from the sample submission, never the encoded integers\n"
            + "11. Structure the script as: load data (env/manifest paths) -> "
              "train -> write submission.csv -> write oof.csv -> print "
              "accuracy; wrap the main flow in try/except BaseException "
              "(catch KeyboardInterrupt too) that PRINTS THE FULL TRACEBACK "
              "(traceback.print_exc()) and still writes both artifacts "
              "before exiting\n"
            + "12. Do NOT use pip install, subprocess, os.system, requests, "
              "urllib, socket, or any URL (even in comments) - the container "
              "is offline and the static gate rejects URLs; only preinstalled "
              "libraries\n"
            + "13. Never read the gold/private labelled test CSV and never use "
              "test labels for training or validation\n"
            + "14. Set a fixed random seed; TOTAL RUNTIME MUST BE 10-25 "
              "MINUTES, HARD CAP 30: past the cap you are SIGKILLed at "
              "the round timeout (rc=-9) and scored as the deterministic "
              "baseline. Training on the FULL training set is allowed "
              "when it fits the budget (small/medium data) - MLE-Bench "
              "only scores the hidden test set, so in-sample OOF over "
              "the full train set is fine. NEVER retrain on the full "
              "dataset after validation (doubles runtime - the top cause "
              "of rc=-9); the trained model IS final, predict test and "
              "oof.csv with it. Up to 5-fold KFold/StratifiedKFold is "
              "allowed against overfitting (each row predicted only "
              "by a model that never trained on it), but a final "
              "full-data retrain is NEVER allowed. DEFAULT for large "
              "data: train on a class-balanced subsample "
              "(e.g. 50-80%% of train rows, up to 10000 rows) - a "
              "partial-data model that finishes beats a full-data model "
              "that gets killed. If the dataset is large, also shrink "
              "images to <=160px, CNN epochs 5-12 with early stopping on "
              "the validation split - a converged model beats a rushed "
              "one, but a finished model beats a timed-out one\n"
            + "15. Pretrained weights are ALLOWED but ONLY from the "
              "PRETRAINED WEIGHT CACHE below (preflight-verified inside the "
              "container): load with timm/torchvision from those exact "
              "paths; NEVER download weights and NEVER reference a model "
              "that is not in the cache - otherwise train from scratch "
              "(still valid) or use a fast sklearn model; validation: ONE "
              "stratified split with early stopping, or up to 5-fold "
              "KFold/StratifiedKFold if you fear overfitting (fill "
              "oof[val_idx] per fold; NO full-data retrain); total "
              "runtime must stay inside the budget\n"
            + "16. WRITE THE ARTIFACTS NO MATTER WHAT: if training fails, "
              + _oof_fallback_full(metric_info) + ", then print "
              "accuracy; never exit before writing both files\n"
            + "17. Reference skeleton (adapt it; keep the artifact contract):\n"
            + "```python\n"
            + "import csv, os\n"
            + "import pandas as pd\n"
            + "TRAIN = os.environ.get('TRAIN_CSV', '')\n"
            + "TEST = os.environ.get('TEST_CSV', '')\n"
            + "SAMPLE = os.environ.get('SAMPLE_SUBMISSION', '')\n"
            + "TARGET = os.environ.get('TARGET_COLUMN', '')\n"
            + "df = pd.read_csv(TRAIN)\n"
            + "y = df[TARGET].astype(str)\n"
            + "pred = y.mode()[0]\n"
            + "test = pd.read_csv(SAMPLE) if SAMPLE else pd.read_csv(TEST)\n"
            + "test['Prediction'] = pred\n"
            + "test.to_csv('submission.csv', index=False)\n"
            + "with open('oof.csv', 'w', newline='') as f:\n"
            + "    w = csv.writer(f)\n"
            + "    w.writerow(['true', 'pred'])\n"
            + "    for v in y:\n"
            + "        w.writerow([v, pred])\n"
            + "acc = float((y == pred).mean())\n"
            + "print('accuracy: %.4f' % acc)\n"
            + "```\n"
            + _oof_skeleton_note(metric_info)
            + "18. Before returning, SELF-REVIEW the script like a code "
              "master: (a) valid Python with correct indentation; (b) every "
              "path comes from the contract; (c) submission.csv header "
              "EXACTLY matches sample_submission.csv and row order is "
              "preserved; " + _oof_review_rule(metric_info)
              + "; (e) it prints 'accuracy: ...'; (f) runtime "
              "stays under 30 minutes and inside the trial budget; (g) no "
              "bare except / no swallowed errors - every except prints the "
              "traceback\n"
            + "19. Common pitfalls to avoid: never assume column names or a "
              "flat layout; keep the id column values verbatim; predictions "
              "must be valid class ids from the training label space; for "
              "images always build paths from TRAIN_IMAGES/TEST_IMAGES; cast "
              "types explicitly to avoid CSV errors; drop or fill NaN; never "
              "reorder sample rows; never read test labels; never hide a bug "
              "behind except: pass\n"
            + "Write ONLY the Python code, no markdown.\n"
        )
        return prompt

    def implement(self, plan: ResearchPlan, profile: AnalysisProfile,
                  data_dir, code_dir, submission_dir,
                  sample_path: str = "", branch: str = "",
                  proposal: Optional[dict] = None,
                  reference_code: str = "",
                  reference_meta: Optional[dict] = None) -> str:
        manifest = _read_manifest(code_dir)
        metric_name = (manifest.get("metric_name") or
                       (manifest.get("metric_info") or {}).get("metric_name") or
                       "accuracy")
        if (branch == "baseline"
                and os.environ.get("V2_DETERMINISTIC_BASELINE", "0") == "1"):
            try:
                layout = resolve_dataset_layout(data_dir, sample_path=sample_path)
            except DatasetLayoutError:
                return FALLBACK_CODE
            return _build_baseline_code(layout, profile,
                                        metric_name=metric_name)
        pretrained_available = manifest.get("pretrained_available") or []
        prompt = self.build_code_prompt(plan, profile, data_dir, code_dir,
                                        submission_dir, sample_path=sample_path,
                                        branch=branch, proposal=proposal,
                                        pretrained_available=pretrained_available,
                                        metric_info=manifest,
                                        reference_code=reference_code,
                                        reference_meta=reference_meta)
        response = self.llm_call(prompt)
        code = clean_code(response)
        if not code or code.lstrip().startswith("{"):
            try:
                layout = resolve_dataset_layout(data_dir, sample_path=sample_path)
            except DatasetLayoutError:
                return FALLBACK_CODE
            return _build_layout_fallback(layout, metric_name=metric_name)
        return code

    def repair(self, code: str, error: str, plan: ResearchPlan,
               profile: Optional[AnalysisProfile] = None) -> str:
        """Arbor-style execution-feedback repair.

        The traceback from a failed run is fed back to the LLM, which returns
        a corrected complete script; it is then re-executed by PACT (bounded
        by max_repairs in the host supervisor). Empty/invalid responses fall
        through to the deterministic artifact fallback.
        """
        competition = profile.competition if profile is not None else "unknown"
        prompt = (
            "ROLE: YOU ARE A CODE MASTER DEBUGGING YOUR OWN SCRIPT.\n"
            "The script below failed inside the execution container. Debug "
            "like a master: read the traceback, identify the FIRST root "
            "cause (the exact failing line and the reason it fails: missing "
            "file, dtype/NaN, wrong path, label-space mismatch, or API "
            "misuse), then fix it surgically. Do not rewrite the whole "
            "script, do not wrap everything in a blanket try/except to hide "
            "the error, and do not make unrelated changes. Fix the bug "
            "precisely and completely - return a working script, not an "
            "explanation. The script below was executed and FAILED.\n\n"
            + "Competition: " + competition + "\n"
            + "Hypothesis: " + plan.hypothesis + "\n"
            + "Approach: " + plan.approach_type + "\n\n"
            + "--- FAILED SCRIPT ---\n"
            + code + "\n\n"
            + "--- EXECUTION ERROR (last lines) ---\n"
            + (error or "(empty)")[:2000] + "\n\n"
            + "Diagnosis guide (match the error to the fix):\n"
            + "  - FileNotFoundError/No such file -> use the exact contract "
              "paths (TRAIN_CSV/TEST_CSV/SAMPLE_SUBMISSION/TRAIN_IMAGES/"
              "TEST_IMAGES or data_manifest.json); never guess relative paths\n"
            + "  - KeyError/ValueError/NaN -> coerce dtypes explicitly "
              "(astype/float()), fill or drop NaN before modeling, verify "
              "column names from the contract\n"
            + "  - label-space mismatch -> keep predictions in the training "
              "label space (raw label ids/strings), fit encoders on train "
              "only and use the same pipeline for test\n"
            + "  - OOM/timeout -> shrink the workload: smaller images, fewer "
              "epochs, class-balanced subsample of train rows\n"
            + "Rules: keep the same data contract; write submission.csv + "
              "oof.csv no matter what; no pip install/subprocess/os.system/"
              "network/URLs; never read gold/private test labels; at most 2 "
              "folds; pretrained weights ONLY from the verified cache (never "
              "download); finish within the trial budget (10-25 minutes, "
              "hard cap 30); print the final validation metric as "
              "'accuracy: 0.xxxx'; return ONLY the Python code, no markdown.\n"
        )
        response = self.llm_call(prompt)
        fixed = clean_code(response)
        if not fixed or fixed.lstrip().startswith("{"):
            return ""
        return fixed


def _layout_manifest(layout: DatasetLayout, profile=None) -> dict:
    manifest = layout.manifest()
    if profile is not None:
        manifest["target_column"] = (
            getattr(profile, "target_column", "") or manifest["target_column"])
        manifest["train_rows"] = int(
            getattr(profile, "train_rows", 0) or manifest["train_rows"])
        manifest["test_rows"] = int(
            getattr(profile, "test_rows", 0) or manifest["test_rows"])
        manifest["task_type"] = (
            getattr(profile, "task_type", "") or manifest["task_type"])
    return manifest


def _read_manifest(code_dir) -> dict:
    """Read the director-written data_manifest.json (metric + data contract)."""
    try:
        path = Path(code_dir) / "data_manifest.json"
        if path.is_file():
            return _json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    return {}


def _read_pretrained_cache(code_dir) -> list:
    """Read the preflight-verified pretrained checkpoint list.

    Written by the director into work_dir/data_manifest.json after the
    container preflight (and mounted into every trial container via
    V2_TORCH_CACHE). Empty when unavailable (host mode / no cache).
    """
    try:
        path = Path(code_dir) / "data_manifest.json"
        if path.is_file():
            data = _json.loads(path.read_text(encoding="utf-8"))
            cached = data.get("pretrained_available") or []
            return [str(c) for c in cached if str(c).strip()]
    except (OSError, ValueError):
        pass
    return []



def _metric_section(metric_info) -> str:
    """Official-metric contract: name, direction and oof.csv column format."""
    info = metric_info or {}
    name = str(info.get("metric_name") or "accuracy")
    label = str(info.get("metric_label") or name)
    direction = str(info.get("metric_direction") or "higher_is_better")
    alignment = str(info.get("metric_alignment") or "exact")
    guide = OOF_GUIDE.get(name, "")
    lines = [
        "OFFICIAL METRIC: %s - %s (%s, alignment=%s)."
        % (label, "maximize" if direction == "higher_is_better" else "minimize",
           name, alignment),
    ]
    if guide:
        lines.append("OOF PREDICTION CONTRACT: %s." % guide)
    lines.append(
        "Your trial metric is recomputed from oof.csv (in-sample "
        "predictions over the full training set are allowed); "
        "submission.csv values follow sample_submission.csv semantics "
        "(probabilities for logloss/AUC competitions).")
    return "\n".join(lines) + "\n\n"


def _pretrained_section(pretrained_available) -> str:
    """SOTA contract: legal pretrained-weight sources (cache only)."""
    cached = [str(c) for c in (pretrained_available or []) if str(c).strip()]
    if cached:
        lines = [
            "PRETRAINED WEIGHT CACHE (preflight-verified inside the container; "
            "the ONLY legal source of pretrained weights):",
        ]
        lines += ["  - /root/.cache/torch/hub/checkpoints/%s" % c for c in cached]
        lines += [
            "Load pattern: timm.create_model('<name>', pretrained=False) then "
            "model.load_state_dict(torch.load('<path>', map_location='cpu')) - "
            "or set pretrained=True ONLY when the matching checkpoint is "
            "listed here. NEVER download weights.",
        ]
        lines += [
            "Checkpoint names may carry a suffix: e.g. "
            "'efficientnet_b0_rwightman-7f5810bc.pth' maps to the timm "
            "model 'efficientnet_b0' - when in doubt, torch.load + "
            "load_state_dict(map_location='cpu') and verify the keys "
            "match the model.",
        ]
        return "\n".join(lines) + "\n\n"
    return (
        "PRETRAINED WEIGHT CACHE: (empty - no cached checkpoints in this "
        "environment) train from scratch or use a fast sklearn model; never "
        "download weights.\n\n"
    )


def _resource_section(plan) -> str:
    """Per-competition RESOURCE PROFILE injected by the planner.

    Hard constraints from the competition portfolio (trial seconds, CV
    folds, image size, epochs, train rows). Empty when unavailable - the
    static requirements below remain the fallback.
    """
    try:
        resource = (plan.method_detail or {}).get("resource_profile") or {}
    except Exception:  # noqa: BLE001 - never break prompt building
        resource = {}
    if not resource:
        return ""
    parts = ["RESOURCE PROFILE (hard constraints for THIS competition):"]
    if resource.get("max_budget_seconds"):
        parts.append("- max trial seconds: %s" % resource["max_budget_seconds"])
    if resource.get("max_folds"):
        parts.append("- max CV folds: %s" % resource["max_folds"])
    if resource.get("image_size_max"):
        parts.append("- image size cap: <=%spx" % resource["image_size_max"])
    if resource.get("epochs_min") and resource.get("epochs_max"):
        parts.append("- epochs: %s-%s" % (resource["epochs_min"],
                                          resource["epochs_max"]))
    if resource.get("train_rows_cap"):
        parts.append("- train rows cap (class-balanced subsample): <=%s"
                     % resource["train_rows_cap"])
    if resource.get("batch_hint"):
        parts.append("- batch size hint: %s" % resource["batch_hint"])
    parts.append("Stay inside these limits - a finished model beats a "
                 "timed-out one (rc=-9).\n")
    return "\n".join(parts) + "\n"


def _gpu_section() -> str:
    """GPU execution contract: CUDA is available and mandatory for
    training; hardcoding cpu wastes the whole trial budget."""
    return (
        "GPU EXECUTION (CUDA IS AVAILABLE AND MANDATORY):\n"
        "  - device = torch.device(\"cuda\") - ALWAYS train on cuda\n"
        "  - NEVER use torch.device(\"cpu\") and NEVER fall back to cpu\n"
        "    via torch.cuda.is_available() - cuda IS available here\n"
        "  - move model AND every batch to device with .to(device)\n"
        "  - torch.load(..., map_location=\"cpu\") is allowed ONLY for\n"
        "    loading pretrained weights; then model.to(device)\n"
        "  - if memory is tight use torch.cuda.amp or batch_size 16-32,\n"
        "    never switch to cpu\n"
        "  - keep total training runtime 10-25 minutes\n\n"
    )


def _cpu_only_section() -> str:
    """CPU-only execution contract for trials that run without a GPU."""
    return (
        "CPU-ONLY EXECUTION (this trial runs on CPU, NO GPU):\n"
        "  - prefer sklearn / lightgbm / xgboost / pandas on tabular features\n"
        "  - for image tasks: tiny model (e.g. logistic regression on pixel\n"
        "    features or a small CNN), device=cpu, max 2 epochs, max 2000\n"
        "    training samples\n"
        "  - NEVER call .cuda() or torch.cuda.is_available() and never\n"
        "    set torch.cuda in any way\n"
        "  - set torch.device(\"cpu\") explicitly if using torch\n"
        "  - use DataLoader with num_workers=0 (CPU-safe)\n"
        "  - keep total runtime under 15 minutes on CPU\n\n"
    )


def _reference_code_section(reference_code: str,
                            reference_meta: Optional[dict] = None) -> str:
    """Round-continuity asset injection: the verified best code of a
    previous round/grant is offered to the implementer as a replaceable
    asset - NEVER as a platform-mandated method. The LLM may extend it,
    modify it surgically, or replace it entirely; the platform only
    guarantees the asset is a real, previously-verified implementation.
    """
    code = (reference_code or "").strip()
    if not code:
        return ""
    meta = reference_meta or {}
    lines = [
        "PREVIOUS BEST CODE (verified incumbent asset, NOT a mandate):",
        "  round=%s metric=%s branch=%s"
        % (meta.get("round_num", "?"), meta.get("metric", "?"),
           meta.get("branch_id") or "?"),
        "This is the strongest VERIFIED implementation so far. Prefer to",
        "build on it: start from this code and make ONE safe, focused",
        "improvement instead of rewriting from scratch, because it already",
        "runs and scores. You may also replace it entirely when your",
        "hypothesis demands a different method - the decision is yours.",
        "Keep its working parts (data loading, label encoding, artifact",
        "writing) intact when you extend it.",
        "```python",
        code[:6000],
        "```",
        "",
    ]
    return "\n".join(lines) + "\n"


def _child_feedback_section(proposal: Optional[dict]) -> str:
    """FeedbackView injection: the implementer sees the focused child
    hypothesis and the verified outcomes of the prior children in this
    grant, so the code it writes adapts to what already happened."""
    if not isinstance(proposal, dict):
        return ""
    child = proposal.get("child_index")
    hypothesis = str(proposal.get("hypothesis") or "").strip()
    evidence = str(proposal.get("evidence") or "").strip()
    if not hypothesis and not evidence:
        return ""
    lines = ["CHILD EXPERIMENT (this trial is child %s of the current grant):"
             % child]
    if hypothesis:
        lines.append("  Focused hypothesis: %s" % hypothesis[:400])
    if evidence:
        lines.append("  Prior-child feedback (verified outcomes):")
        for line in evidence.splitlines()[:8]:
            lines.append("    " + line[:160])
    lines.append("Implement exactly this focused experiment; do not wander "
                 "beyond it. If the feedback shows a dead end, fix it instead "
                 "of repeating it.")
    return "\n".join(lines) + "\n\n"


def clean_code(response: str) -> str:
    if not response:
        return ""
    code = re.sub(r"^```python\s*", "", response.strip())
    code = re.sub(r"^```\s*", "", code.strip())
    code = re.sub(r"\s*```$", "", code.strip())
    return code.strip()


def _build_layout_fallback(layout: DatasetLayout,
                           metric_name: Optional[str] = None) -> str:
    """LLM-empty/garbage fallback: sample-preserving submission + metric-aware OOF.

    This is the LAST-RESORT path when the LLM returns empty/garbage code. The
    submission contract here is deliberately conservative: copy the sample
    submission verbatim (or mirror the test CSV when no sample exists) so the
    published file is always structurally valid. The OOF columns still follow
    the official metric contract (probability families get true_<class> /
    pred_<class> columns) so the trusted evaluator can recompute a real metric
    instead of failing the trial.
    """
    train_path = repr(str(layout.train_path))
    test_path = repr(str(layout.test_path))
    sample_path = repr(str(layout.sample_submission_path or ""))
    mn = repr(metric_name or "accuracy")
    return (
        "# Deterministic fallback generated by HERA/PACT (stdlib only).\n"
        "import csv, os, shutil\n"
        "\n"
        "train_path = " + train_path + "\n"
        "test_path = " + test_path + "\n"
        "sample_path = " + sample_path + "\n"
        "metric_name = " + mn + "\n"
        "\n"
        "def _read(path):\n"
        "    with open(path, 'r', encoding='utf-8', errors='replace', newline='') as fh:\n"
        "        return list(csv.DictReader(fh))\n"
        "\n"
        "# submission: sample copy is the safe fallback contract\n"
        "if sample_path and os.path.isfile(sample_path):\n"
        "    shutil.copyfile(sample_path, 'submission.csv')\n"
        "else:\n"
        "    test_rows = _read(test_path) if os.path.isfile(test_path) else []\n"
        "    if test_rows:\n"
        "        header = list(test_rows[0].keys())\n"
        "        with open('submission.csv', 'w', encoding='utf-8', newline='') as fh:\n"
        "            w = csv.writer(fh)\n"
        "            w.writerow(header)\n"
        "            for r in test_rows:\n"
        "                w.writerow([r.get(c, '') for c in header])\n"
        "\n"
        "# oof: metric-aware columns so the trusted evaluator can recompute\n"
        "train_rows = _read(train_path) if os.path.isfile(train_path) else []\n"
        "values = []\n"
        "if train_rows:\n"
        "    if len(train_rows[0]) > 1:\n"
        "        values = [list(r.values())[-1].strip() for r in train_rows]\n"
        "    else:\n"
        "        values = [list(r.values())[0].strip() for r in train_rows]\n"
        "if values:\n"
        "    with open('oof.csv', 'w', encoding='utf-8', newline='') as fh:\n"
        "        w = csv.writer(fh)\n"
        "        if metric_name in ('logloss', 'weighted_logloss', 'kl_div', 'mean_auc_multilabel'):\n"
        "            classes = sorted(set(values))\n"
        "            n = float(len(values) or 1)\n"
        "            freqs = {}\n"
        "            for v in values:\n"
        "                freqs[v] = freqs.get(v, 0) + 1\n"
        "            freqs = {k: freqs[k] / n for k in classes}\n"
        "            w.writerow(['true'] + ['true_' + k for k in classes] + ['pred_' + k for k in classes])\n"
        "            for v in values:\n"
        "                w.writerow([v] + ['1' if v == k else '0' for k in classes]\n"
        "                           + [format(freqs[k], '.6f') for k in classes])\n"
        "        elif metric_name in ('binary_logloss', 'auc'):\n"
        "            n = float(len(values) or 1)\n"
        "            freqs = {}\n"
        "            for v in values:\n"
        "                freqs[v] = freqs.get(v, 0) + 1\n"
        "            freqs = {k: freqs[k] / n for k in freqs}\n"
        "            pos = '1' if '1' in freqs else ('True' if 'True' in freqs else max(freqs, key=freqs.get))\n"
        "            w.writerow(['true', 'pred'])\n"
        "            for v in values:\n"
        "                w.writerow([v, format(freqs.get(pos, 0.5), '.6f')])\n"
        "        else:\n"
        "            counts = {}\n"
        "            for v in values:\n"
        "                counts[v] = counts.get(v, 0) + 1\n"
        "            pred = max(counts, key=counts.get) if counts else ''\n"
        "            w.writerow(['true', 'pred'])\n"
        "            for v in values:\n"
        "                w.writerow([v, pred])\n"
        "print('accuracy: 0.0000')\n"
    )



def _build_baseline_code(layout: DatasetLayout, profile=None,
                         metric_name: Optional[str] = None) -> str:
    """Deterministic stdlib-only majority/mean baseline, metric-aware.

    Round 1 with branch=baseline uses this instead of LLM code, so a real
    metric is guaranteed as soon as the data contract resolves. The oof.csv
    columns follow metrics_registry.OOF_GUIDE:
      - probability families (logloss, weighted_logloss, kl_div,
        mean_auc_multilabel): one-hot true_<class> plus class-frequency
        pred_<class> columns
      - binary probability families (binary_logloss, auc): pred = the
        positive-class frequency (0..1)
      - everything else: true,pred with the majority class id / mean
    """
    train_path = repr(str(layout.train_path))
    test_path = repr(str(layout.test_path))
    sample_path = repr(str(layout.sample_submission_path or ""))
    target = ""
    task_type = "classification"
    if profile is not None:
        target = getattr(profile, "target_column", "") or ""
        task_type = getattr(profile, "task_type", "classification") or "classification"
    template = (
        "# Deterministic baseline generated by HERA/PACT (stdlib only).\n"
        "import csv, os\n"
        "\n"
        "train_path = @@TRAIN_PATH@@\n"
        "test_path = @@TEST_PATH@@\n"
        "sample_path = @@SAMPLE_PATH@@\n"
        "target_col = @@TARGET@@\n"
        "task_type = @@TASK_TYPE@@\n"
        "metric_name = @@METRIC_NAME@@\n"
        "\n"
        "def _read(path):\n"
        "    with open(path, 'r', encoding='utf-8', errors='replace', newline='') as fh:\n"
        "        return list(csv.DictReader(fh))\n"
        "\n"
        "def _is_float(v):\n"
        "    try:\n"
        "        float(v)\n"
        "        return True\n"
        "    except (TypeError, ValueError):\n"
        "        return False\n"        "\n"
        "def _sniff_newline(path):\n"
        "    try:\n"
        "        if path and os.path.isfile(path):\n"
        "            with open(path, 'rb') as fh:\n"
        "                chunk = fh.read(8192)\n"
        "            if b'\\r\\n' in chunk:\n"
        "                return '\\r\\n'\n"
        "            if b'\\n' in chunk:\n"
        "                return '\\n'\n"
        "            if b'\\r' in chunk:\n"
        "                return '\\r'\n"
        "    except OSError:\n"
        "        pass\n"
        "    return '\\r\\n'\n"
        "\n"
        "train_rows = _read(train_path) if os.path.isfile(train_path) else []\n"
        "test_rows = _read(test_path) if os.path.isfile(test_path) else []\n"
        "values = []\n"
        "if train_rows:\n"
        "    if target_col and target_col in train_rows[0]:\n"
        "        values = [(r.get(target_col) or '').strip() for r in train_rows]\n"
        "    elif len(train_rows[0]) > 1:\n"
        "        values = [list(r.values())[-1].strip() for r in train_rows]\n"
        "numeric = [v for v in values if _is_float(v)]\n"
        "if task_type == 'regression':\n"
        "    pred = repr(round(sum(float(v) for v in numeric) / len(numeric), 6)) if numeric else '0.0'\n"
        "    acc = 0.0\n"
        "else:\n"
        "    counts = {}\n"
        "    for v in values:\n"
        "        counts[v] = counts.get(v, 0) + 1\n"
        "    pred = max(counts, key=counts.get) if counts else ''\n"
        "    acc = (counts.get(pred, 0) / len(values)) if values else 0.0\n"
        "\n"
        "src_rows = None\n"
        "if sample_path and os.path.isfile(sample_path):\n"
        "    src_rows = _read(sample_path)\n"
        "elif test_rows:\n"
        "    src_rows = test_rows\n"
        "if src_rows:\n"
        "    header = list(src_rows[0].keys())\n"
        "    id_col = header[0]\n"
        "    pred_cols = header[1:]\n"
        "    with open('submission.csv', 'w', encoding='utf-8', newline='') as fh:\n"
        "        w = csv.writer(fh, lineterminator=_sniff_newline(sample_path))\n"
        "        w.writerow(header)\n"
        "        for r in src_rows:\n"
        "            out = []\n"
        "            for col in header:\n"
        "                if col == id_col:\n"
        "                    out.append(r.get(col, ''))\n"
        "                elif len(pred_cols) > 1:\n"
        "                    out.append(1.0 if col == str(pred) else 0.0)\n"
        "                else:\n"
        "                    out.append(pred)\n"
        "            w.writerow(out)\n"
        "\n"
        "with open('oof.csv', 'w', encoding='utf-8', newline='') as fh:\n"
        "    w = csv.writer(fh)\n"
        "    if metric_name in ('logloss', 'weighted_logloss', 'kl_div', 'mean_auc_multilabel'):\n"
        "        classes = sorted(set(values))\n"
        "        n = float(len(values) or 1)\n"
        "        freqs = {}\n"
        "        for v in values:\n"
        "            freqs[v] = freqs.get(v, 0) + 1\n"
        "        freqs = {k: freqs[k] / n for k in classes}\n"
        "        w.writerow(['true'] + ['true_%s' % k for k in classes] + ['pred_%s' % k for k in classes])\n"
        "        for v in values:\n"
        "            w.writerow([v] + ['1' if v == k else '0' for k in classes]\n"
        "                       + ['%.6f' % freqs[k] for k in classes])\n"
        "    elif metric_name in ('binary_logloss', 'auc'):\n"
        "        n = float(len(values) or 1)\n"
        "        freqs = {}\n"
        "        for v in values:\n"
        "            freqs[v] = freqs.get(v, 0) + 1\n"
        "        freqs = {k: freqs[k] / n for k in freqs}\n"
        "        pos = '1' if '1' in freqs else ('True' if 'True' in freqs else max(freqs, key=freqs.get))\n"
        "        w.writerow(['true', 'pred'])\n"
        "        for v in values:\n"
        "            w.writerow([v, '%.6f' % freqs.get(pos, 0.5)])\n"
        "    else:\n"
        "        w.writerow(['true', 'pred'])\n"
        "        for v in values:\n"
        "            w.writerow([v, pred])\n"
        "print('accuracy: {0:.4f}'.format(acc if values else 0.0))\n"
    )
    return (template
            .replace("@@TRAIN_PATH@@", train_path)
            .replace("@@TEST_PATH@@", test_path)
            .replace("@@SAMPLE_PATH@@", sample_path)
            .replace("@@TARGET@@", repr(target))
            .replace("@@TASK_TYPE@@", repr(task_type))
            .replace("@@METRIC_NAME@@", repr(metric_name or "accuracy")))
