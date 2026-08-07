# -*- coding: utf-8 -*-
"""pact/calibration_probe.py - startup F0 runtime calibration (generic).

Why: v2.2.1-rc3 calibrated F0 lazily from the first 2 successful trials,
so grant #1 always ran on GUESSED cost estimates (f0_seconds=0) and HERA
over-spent its cheap probes (29min "cheap" probes on 3 cases). rc4 runs a
tiny, generic measurement probe ONCE before grant #1 so the very first
grant already gets honest t_est/est_cost numbers.

Genericity (MLE-Bench-wide, never competition-specific):
  - image tasks: load the prebuilt multi-size cache (smallest size) when
    present, else decode a 64-image subsample and measure the decode rate;
    X = flattened uint8 arrays (or raw decode).
  - tabular tasks: numeric columns from TRAIN_CSV; X = first N rows.
  - text tasks: degrade to the tabular numeric path if any numeric column
    exists, else the probe is skipped (lazy calibration remains the
    fallback).
  - model: LogisticRegression(max_iter=50) for classification (any class
    count), Ridge() for regression - chosen ONLY from task_type, never
    from the competition name.
  - The probe is fail-open: any error => run_calibration_probe returns {}
    and the director keeps the lazy path.

Container contract: the probe script runs inside the same exec image with
the same mounts as a trial (work dir rw, public dir ro, torch cache), and
prints one line `F0_PROBE <json>` as the machine-readable result.
"""
import json
import os
import shlex
import subprocess
import time
import uuid
from pathlib import Path

PROBE_ROWS_DEFAULT = 512
PROBE_IMAGES_DEFAULT = 64
_PROBE_TIMEOUT = 1200


def probe_script() -> str:
    """Container-side probe (pure stdlib + numpy/sklearn/PIL, no network)."""
    return r'''
import json, os, time
from pathlib import Path
import numpy as np

def emit(data):
    print("F0_PROBE " + json.dumps(data, ensure_ascii=False), flush=True)

def fail(reason):
    emit({"error": str(reason)[:300]})
    raise SystemExit(1)

rows = int(os.environ.get("PROBE_ROWS", "512"))
n_img = int(os.environ.get("PROBE_IMAGES", "64"))
train_csv = os.environ.get("TRAIN_CSV", "")
train_images = os.environ.get("TRAIN_IMAGES", "")
target_col = os.environ.get("TARGET_COLUMN", "")
task_type = (os.environ.get("TASK_TYPE", "") or "").strip().lower()
cache_dirs = {}
try:
    cache_dirs = json.loads(os.environ.get("V2_CACHE_DIRS", "{}") or "{}")
except Exception:
    cache_dirs = {}

if not train_csv or not os.path.isfile(train_csv):
    fail("probe: TRAIN_CSV missing")

import csv as _csv
with open(train_csv, newline="", encoding="utf-8", errors="replace") as fh:
    all_rows = list(_csv.reader(fh))
if not all_rows:
    fail("probe: empty train csv")
header = all_rows[0]
data_rows = all_rows[1:]
n_total = len(data_rows)

# ---- labels ----
y = None
if target_col and target_col in header:
    ti = header.index(target_col)
    y = np.asarray([r[ti] for r in data_rows[:rows]], dtype=object)

# ---- features: cache first, then raw decode, then numeric CSV ----
X = None
mode = "none"
decode_sec_per_image = 0.0
use_cache = False
if train_images and os.path.isdir(train_images):
    sizes = sorted([int(s) for s in cache_dirs.keys()], reverse=True)
    if sizes:
        use_cache = True
        size = min(sizes)  # smallest cache = cheapest probe
        d = cache_dirs[str(size)]
        xp = Path(d) / "train_X.npy"
        ids_p = Path(d) / "train_ids.json"
        if xp.is_file() and ids_p.is_file():
            ids = json.loads(ids_p.read_text(encoding="utf-8"))
            want = ids[:rows]
            arr = np.load(xp, mmap_mode="r")
            t0 = time.time()
            idx = []
            id_set = {v: i for i, v in enumerate(ids)}
            for w in want:
                j = id_set.get(w)
                if j is not None:
                    idx.append(j)
            X = np.asarray(arr[idx[:rows]], dtype=np.float32)
            load_sec = time.time() - t0
            X = X.reshape(len(X), -1)
            mode = "cache_%d" % size
            decode_sec_per_image = load_sec / max(1, len(X))
        else:
            fail("probe: cache dir missing train_X.npy")
    if X is None:
        # raw decode subsample (no cache available yet)
        from PIL import Image
        from concurrent.futures import ThreadPoolExecutor
        # generic id -> path index: flat dirs, class subdirs, deeper
        # nesting; keys are full filename AND stem (bare vs extensioned ids)
        img_idx = {}
        _stack = [Path(train_images)]
        while _stack:
            _d = _stack.pop()
            try:
                _entries = list(_d.iterdir())
            except OSError:
                continue
            for _e in _entries:
                try:
                    if _e.is_dir():
                        _stack.append(_e)
                    elif _e.is_file() and _e.suffix.lower() in (
                            ".png", ".jpg", ".jpeg", ".bmp",
                            ".tif", ".tiff", ".webp"):
                        img_idx.setdefault(_e.name, str(_e))
                        img_idx.setdefault(_e.stem, str(_e))
                except OSError:
                    continue
        img_ids = [r[0] for r in data_rows[:n_img] if r]
        t0 = time.time()

        def dec(img_id):
            p = img_idx.get(str(img_id))
            if p is None:
                return np.zeros((64, 64, 3), dtype=np.uint8)
            try:
                return np.asarray(Image.open(p).convert("RGB").resize(
                    (64, 64)), dtype=np.uint8)
            except Exception:
                return np.zeros((64, 64, 3), dtype=np.uint8)
        with ThreadPoolExecutor(max_workers=8) as pool:
            imgs = list(pool.map(dec, img_ids))
        dt = time.time() - t0
        decode_sec_per_image = dt / max(1, len(imgs))
        X = np.asarray(imgs, dtype=np.float32).reshape(len(imgs), -1)
        mode = "raw_64"
else:
    # tabular/text fallback: numeric columns only (generic, cheap)
    num_idx = []
    for ci, col in enumerate(header):
        vals = []
        ok = True
        for r in data_rows[:64]:
            try:
                vals.append(float(r[ci]))
            except (ValueError, IndexError):
                ok = False
                break
        if ok:
            num_idx.append(ci)
    if num_idx:
        X = np.zeros((min(rows, n_total), len(num_idx)), dtype=np.float32)
        for ri in range(min(rows, n_total)):
            for kj, ci in enumerate(num_idx):
                try:
                    X[ri, kj] = float(data_rows[ri][ci])
                except (ValueError, IndexError):
                    X[ri, kj] = 0.0
        mode = "tabular_%d" % len(num_idx)

if X is None or len(X) == 0:
    fail("probe: no measurable features (image cache/raw or numeric cols)")

if y is None or len(y) == 0:
    fail("probe: no target column")

# numeric-encode y for sklearn (classification ids / regression floats)
from sklearn.preprocessing import LabelEncoder
try:
    yf = y.astype(float)
    is_cls = False
except (TypeError, ValueError):
    yf = LabelEncoder().fit_transform(y)
    is_cls = True
if task_type in ("regression", "regression_multioutput", "ordinal_regression") and not is_cls:
    is_cls = False
else:
    is_cls = is_cls or (task_type in ("classification", "multiclass_classification",
                                      "binary_classification", "multi_label_classification",
                                      "multilabel_classification"))

n_use = min(rows, len(X), len(yf))
X = X[:n_use]
yf = yf[:n_use]
if n_use < 32:
    fail("probe: too few rows (%d)" % n_use)

t0 = time.time()
if is_cls:
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=50)
else:
    from sklearn.linear_model import Ridge
    model = Ridge()
try:
    model.fit(X, yf)
except Exception as exc:  # noqa: BLE001 - degenerate features -> skip probe
    fail("probe: fit failed %r" % exc)
fit_sec = time.time() - t0

emit({
    "rows_measured": int(n_use),
    "total_rows": int(n_total),
    "fit_seconds": round(float(fit_sec), 4),
    "decode_sec_per_image": round(float(decode_sec_per_image), 6),
    "feature_mode": mode,
    "image_cache": use_cache,
    "n_classes": int(len(set(yf.tolist()))),
    "task_type": task_type,
    "ok": True,
})
'''


def _manifest_env(manifest: dict) -> dict:
    """Same data-contract env mapping as the executor (generic)."""
    env = {}
    for key, mkey in (("DATA_DIR", "dataset_root"),
                      ("TRAIN_CSV", "train_csv"),
                      ("TEST_CSV", "test_csv"),
                      ("SAMPLE_SUBMISSION", "sample_submission"),
                      ("TRAIN_IMAGES", "train_images"),
                      ("TEST_IMAGES", "test_images"),
                      ("TARGET_COLUMN", "target_column"),
                      ("TASK_TYPE", "task_type")):
        value = manifest.get(mkey)
        if value:
            # v2.3.7: pixel-level layouts measure F0 on the numeric pixel
            # CSV path (id/value), never the per-image cache: the cache id
            # order (image stems) cannot match pixel rows, so a cache probe
            # would fit garbage features against pixel targets.
            if mkey in ("train_images", "test_images") and \
                    manifest.get("pixel_level"):
                continue
            env[key] = str(value)
    return env


def run_calibration_probe(work_dir, manifest: dict, cache_dirs: dict,
                          docker_bin: str = "docker", exec_image: str = "",
                          exec_python: str = "", rows: int = PROBE_ROWS_DEFAULT,
                          images: int = PROBE_IMAGES_DEFAULT,
                          timeout: int = _PROBE_TIMEOUT,
                          run_fn=None) -> dict:
    """Run the F0 probe in the exec container; parse F0_PROBE json line.

    Fail-open: any error returns {} (director falls back to lazy F0).
    run_fn(cmd) -> (rc, stdout, stderr) is injectable for tests.
    """
    try:
        work_dir_str = str(Path(work_dir))
        probe_path = Path(work_dir) / "_f0_probe.py"
        probe_path.write_text(probe_script(), encoding="utf-8")
        public_dir = manifest.get("public_dir") or ""
        if not public_dir:
            return {}
        tokens = shlex.split(exec_python) or ["python3"]
        # unique name: concurrent tasks may probe F0 in the same second;
        # a time-only name collides (docker rc=125) and the loser keeps no
        # calibration. uuid suffix => no collision.
        cmd = [docker_bin, "run", "--rm", "--name",
               "v2_f0probe_%s_%s" % (int(time.time()), uuid.uuid4().hex[:8]),
               "-e", "PYTHONUNBUFFERED=1"]
        cmd += ["-v", "%s:%s" % (work_dir_str, work_dir_str)]
        cmd += ["-v", "%s:%s:ro" % (public_dir, public_dir)]
        env = _manifest_env(manifest)
        env["V2_CACHE_DIRS"] = json.dumps({str(k): str(v)
                                           for k, v in (cache_dirs or {}).items()})
        env["PROBE_ROWS"] = str(int(rows))
        env["PROBE_IMAGES"] = str(int(images))
        for k, v in env.items():
            cmd += ["-e", "%s=%s" % (k, v)]
        cmd += ["-w", work_dir_str, "--entrypoint", tokens[0], exec_image]
        cmd += tokens[1:] + [str(probe_path)]
        if run_fn is not None:
            rc, out, err = run_fn(cmd)
        else:
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                      encoding="utf-8", errors="replace",
                                      timeout=timeout)
                rc, out, err = proc.returncode, proc.stdout or "", proc.stderr or ""
            except subprocess.TimeoutExpired:
                return {}
        if rc != 0:
            return {}
        for line in (out or "").splitlines():
            if line.startswith("F0_PROBE "):
                data = json.loads(line[len("F0_PROBE "):].strip())
                if data.get("ok"):
                    return data
        return {}
    except Exception:  # noqa: BLE001 - fail-open
        return {}


def project_f0(probe: dict, train_rows: int) -> dict:
    """Project the measured probe to a full-run cheap-probe estimate.

    f0_seconds = fit_seconds * (rows/rows_measured)^0.7
                 + decode/load seconds per row * total rows
    Returns a f0_calibration-compatible payload (schema v2_f0_v4).
    """
    probe = dict(probe or {})
    rows_measured = max(1, int(probe.get("rows_measured") or 1))
    rows = max(1, int(train_rows or rows_measured))
    fit_sec = float(probe.get("fit_seconds") or 0)
    decode_sec = float(probe.get("decode_sec_per_image") or 0)
    if probe.get("image_cache"):
        load_sec_per_row = 0.01  # memmap/load overhead per row, cached path
    else:
        load_sec_per_row = decode_sec
    projected = (fit_sec * ((rows / float(rows_measured)) ** 0.7)
                 + load_sec_per_row * rows)
    projected = max(5.0, float(projected))
    return {
        "schema_version": "v2_f0_v4",
        "f0_seconds": round(projected, 1),
        "median_seconds": round(projected, 1),
        "samples_seconds": [round(projected, 1)],
        "sample_receipt_ids": ["f0_probe"],
        "probe": True,
        "probe_rows_measured": int(rows_measured),
        "probe_fit_seconds": round(fit_sec, 4),
        "probe_decode_sec_per_image": round(decode_sec, 6),
        "probe_feature_mode": str(probe.get("feature_mode") or ""),
        "probe_image_cache": bool(probe.get("image_cache")),
        "cache_profile": [],
        "train_rows": int(rows),
        "profile_hash": "",
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }