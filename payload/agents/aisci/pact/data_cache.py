# -*- coding: utf-8 -*-
"""pact/data_cache.py - shared per-competition image cache (zero-decode).

Root-cause fix for GPU-idle rc=-9: every trial re-decoded ALL images
inside its container (PIL open -> resize -> numpy), costing 10-30+
minutes per child over NFS, repeated for every round and every child.
This module decodes every image EXACTLY ONCE per run and every trial
loads prebuilt uint8 arrays instead.

v2.2.1-rc4 (multi-size): a single build decodes at the LARGEST requested
size once, then downscales in memory to every smaller cached size, so
cheap probes (64/128px) and expensive models (192/256px) all hit the
cache without re-decoding. Layout:

  work_dir/data_cache/<content_key>/<size>/train_X.npy|train_ids.json
                                         |test_X.npy|test_ids.json|meta.json

Generic rule for ALL MLE-Bench tasks: whenever the cache exists it must
be used; decode/prepare-once, reuse-always beats per-trial recompute.

Security: the builder container mounts ONLY the public dir (ro) plus the
work dir - never the data root - so private/gold labels are physically
unreachable from the cache build path too.
"""
import hashlib
import json
import os
import shlex
import subprocess
import time
import uuid
from pathlib import Path
from typing import Optional

CACHE_DIRNAME = "data_cache"
_DEFAULT_SIZE = 192
DEFAULT_CACHE_SIZES = (64, 128, 192, 256)
_BUILD_TIMEOUT = 7200
_CHUNK = 512


def cache_key(manifest: dict) -> str:
    """Content-addressed key (v2, size-free): any data/layout change =>
    fresh cache; the size dimension lives INSIDE the cache tree so one
    decode serves every requested resolution."""
    snapshot = {
        "schema": "v2_size_free",
        "train_images": manifest.get("train_images") or "",
        "test_images": manifest.get("test_images") or "",
        "train_csv": manifest.get("train_csv") or "",
        "test_csv": manifest.get("test_csv") or "",
        "sample_submission": manifest.get("sample_submission") or "",
    }
    blob = json.dumps(snapshot, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def cache_root(work_dir, manifest: dict) -> Path:
    return Path(work_dir) / CACHE_DIRNAME / cache_key(manifest)


def cache_dir(work_dir, manifest: dict, image_size: int) -> Path:
    return cache_root(work_dir, manifest) / str(int(image_size or _DEFAULT_SIZE))


def parse_sizes(value) -> tuple:
    """Normalize a size spec (env string / list / None) to a sorted tuple
    of positive ints capped at 384."""
    sizes = []
    if value is None:
        return DEFAULT_CACHE_SIZES
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:  # noqa: BLE001 - fall back to comma split
            value = [int(x) for x in str(value).replace(" ", "").split(",")
                     if str(x).strip().isdigit()]
    if isinstance(value, (list, tuple, set)):
        for v in value:
            try:
                v = int(v)
            except (TypeError, ValueError):
                continue
            if 16 <= v <= 384:
                sizes.append(v)
    elif isinstance(value, int) and 16 <= value <= 384:
        sizes.append(value)
    return tuple(sorted(set(sizes))) or DEFAULT_CACHE_SIZES


def _builder_script() -> str:
    """Decode-once multi-size builder, executed inside the verified exec
    image (PIL+numpy guaranteed). Decodes at the max requested size and
    downscales in memory for every smaller size (memmap-backed so huge
    datasets never need the full max-size array in RAM)."""
    return r'''
import json, os, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(os.environ["CACHE_ROOT"])
KEY = os.environ["CACHE_KEY"]
SIZES = sorted(json.loads(os.environ.get("CACHE_SIZES", "[192]")))
ID_COLS = json.loads(os.environ.get("CACHE_ID_COLS", "{}"))
CHUNK = 512


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def index_images(img_dir):
    """id -> path map for a (possibly nested) image dir. Keys are BOTH the
    full filename and the stem, so bare CSV ids (aptos) and extensioned ids
    (aerial) resolve identically. Generic for every MLE-Bench layout:
    flat dirs, class subdirs, deeper nesting."""
    idx = {}
    stack = [Path(img_dir)]
    while stack:
        d = stack.pop()
        try:
            entries = list(d.iterdir())
        except OSError:
            continue
        for e in entries:
            try:
                if e.is_dir():
                    stack.append(e)
                elif e.is_file() and e.suffix.lower() in IMG_EXTS:
                    idx.setdefault(e.name, str(e))
                    idx.setdefault(e.stem, str(e))
            except OSError:
                continue
    return idx


_IMG_INDEX = {}


def decode(entry):
    img_id, img_dir = entry
    p = _IMG_INDEX.get(str(img_id))
    if p is None:
        return None
    try:
        img = Image.open(p).convert("RGB")
    except Exception:
        return None
    return np.asarray(img, dtype=np.uint8)


def rows_and_ids(kind, csv_path, img_dir):
    import csv as _csv
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as fh:
        rows = list(_csv.reader(fh))
    header = rows[0] if rows else []
    id_col = ID_COLS.get(kind)
    if not id_col or id_col not in header:
        id_col = "id" if "id" in header else (header[0] if header else None)
    if id_col is None:
        raise SystemExit("CACHE_NO_ID_COLUMN train")
    idx = header.index(id_col)
    ids = [str(r[idx]) for r in rows[1:] if len(r) > idx]
    return ids


def build(kind, csv_path, img_dir):
    global _IMG_INDEX
    _IMG_INDEX = index_images(img_dir)
    ids = rows_and_ids(kind, csv_path, img_dir)
    n = len(ids)
    max_size = max(SIZES)
    targets = {}
    for s in SIZES:
        d = ROOT / str(s)
        d.mkdir(parents=True, exist_ok=True)
        meta = d / "meta.json"
        if meta.exists():
            try:
                m = json.loads(meta.read_text(encoding="utf-8"))
                m_rows = m.get("rows_" + kind) or m.get("rows")
                if m.get("key") == KEY and m_rows == n:
                    targets[s] = {"mmap": None, "done": True}
                    continue
            except Exception:
                pass
        arr = np.lib.format.open_memmap(
            str(d / ("%s_X.npy" % kind)), mode="w+",
            dtype=np.uint8, shape=(n, s, s, 3))
        targets[s] = {"mmap": arr, "done": False}
    if all(v["done"] for v in targets.values()):
        return n, 0, True
    missing = 0
    with ThreadPoolExecutor(max_workers=16) as pool:
        for start in range(0, n, CHUNK):
            chunk_ids = ids[start:start + CHUNK]
            decoded = list(pool.map(decode,
                                    ((iid, img_dir) for iid in chunk_ids)))
            for i, arr in enumerate(decoded):
                if arr is None:
                    missing += 1
                    arr = np.zeros((max_size, max_size, 3), dtype=np.uint8)
                else:
                    arr = np.asarray(Image.fromarray(arr).resize(
                        (max_size, max_size)), dtype=np.uint8)
                for s in SIZES:
                    t = targets[s]
                    if t["done"]:
                        continue
                    if s == max_size:
                        t["mmap"][start + i] = arr
                    else:
                        small = np.asarray(Image.fromarray(arr).resize(
                            (s, s)), dtype=np.uint8)
                        t["mmap"][start + i] = small
    for s in SIZES:
        t = targets[s]
        if t["done"]:
            continue
        t["mmap"].flush()
        del t["mmap"]
        (ROOT / str(s) / ("%s_ids.json" % kind)).write_text(
            json.dumps(ids), encoding="utf-8")
        (ROOT / str(s) / "meta.json").write_text(
            json.dumps({"key": KEY, "size": s,
                        "rows": n, "rows_" + kind: n,
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                    time.gmtime())}),
            encoding="utf-8")
    return n, missing, False


n_tr, m_tr, skip_tr = build("train", os.environ["CACHE_TRAIN_CSV"],
                            os.environ["CACHE_TRAIN_IMAGES"])
n_te, m_te, skip_te = build("test", os.environ["CACHE_TEST_CSV"],
                            os.environ["CACHE_TEST_IMAGES"])
for _s in SIZES:
    _mp = ROOT / str(_s) / "meta.json"
    if _mp.exists():
        try:
            _m = json.loads(_mp.read_text(encoding="utf-8"))
            _m["rows_train"] = n_tr
            _m["rows_test"] = n_te
            _m["rows"] = n_tr  # backward-compatible: train rows
            _mp.write_text(json.dumps(_m), encoding="utf-8")
        except Exception:
            pass
print("CACHE_BUILD_OK train=%d test=%d sizes=%s missing=%d/%d" %
      (n_tr, n_te, json.dumps(SIZES), m_tr, m_te))
'''


def _public_dir(manifest: dict) -> str:
    public_dir = manifest.get("public_dir") or ""
    if not public_dir:
        raise ValueError("data cache requires manifest public_dir")
    return public_dir


def _csv_data_rows(path) -> Optional[int]:
    """Data-row count of a CSV (minus header); None when unreadable.
    Used to validate cache hits against the CURRENT dataset so a replaced
    CSV under the same path cannot keep serving stale arrays."""
    try:
        n = 0
        with open(path, "r", encoding="utf-8", errors="replace",
                  newline="") as fh:
            for _ in fh:
                n += 1
        return max(0, n - 1)
    except (OSError, TypeError):
        return None


def _expected_csv_rows(manifest: dict) -> dict:
    return {"train": _csv_data_rows(manifest.get("train_csv") or ""),
            "test": _csv_data_rows(manifest.get("test_csv") or "")}


def cache_map(work_dir, manifest: dict, sizes=None) -> dict:
    """{size: dir} for cache sizes that already exist and are valid."""
    root = cache_root(work_dir, manifest)
    out = {}
    for s in parse_sizes(sizes):
        d = root / str(s)
        meta = d / "meta.json"
        if meta.is_file():
            try:
                m = json.loads(meta.read_text(encoding="utf-8"))
                if m.get("key") == cache_key(manifest):
                    out[int(s)] = str(d)
            except Exception:  # noqa: BLE001 - corrupt cache -> ignored
                pass
    return out


def ensure_image_caches(work_dir, manifest: dict, sizes=None,
                        docker_bin: str = "docker", exec_image: str = "",
                        exec_python: str = "", run_fn=None,
                        force: bool = False,
                        timeout: int = _BUILD_TIMEOUT) -> dict:
    """Idempotent multi-size build. Returns {size: {status, dir, rows}}.

    run_fn(cmd: list) -> (returncode, stdout, stderr) is injectable for
    tests; default shells out to docker. One docker run decodes at the
    max requested size and derives every smaller size from it.
    """
    sizes = parse_sizes(sizes)
    root = cache_root(work_dir, manifest)
    key = cache_key(manifest)
    existing = cache_map(work_dir, manifest, sizes)
    # v2.3.1 stale-cache guard: the cache key is PATH-based, so a dataset
    # regenerated under the same paths (prepared dir rebuilt / CSV replaced)
    # would otherwise serve wrong arrays forever. Validate every hit's meta
    # row counts against the CURRENT CSVs; ANY mismatch invalidates the
    # whole hit set and the builder self-heals per kind/size (rows it can
    # verify are kept, mismatched ones are rebuilt).
    expected = _expected_csv_rows(manifest)
    result = {}
    stale = False
    for s in sizes:
        if not force and s in existing:
            meta = {}
            try:
                meta = json.loads((Path(existing[s]) / "meta.json").read_text(
                    encoding="utf-8"))
            except Exception:  # noqa: BLE001 - meta is display-only here
                pass
            rows_tr = meta.get("rows_train") or meta.get("rows")
            rows_te = meta.get("rows_test") or meta.get("rows")
            exp_tr, exp_te = expected.get("train"), expected.get("test")
            if (exp_tr is not None and rows_tr != exp_tr) or (
                    exp_te is not None and rows_te != exp_te):
                stale = True
                break
            result[s] = {"status": "hit", "dir": existing[s], "key": key,
                         "rows_train": rows_tr, "rows_test": rows_te}
    if stale:
        result = {}
    if len(result) == len(sizes):
        return result
    root.mkdir(parents=True, exist_ok=True)
    script_path = root / "_cache_build.py"
    script_path.write_text(_builder_script(), encoding="utf-8")
    work_dir_str = str(Path(work_dir))
    env_map = {
        "CACHE_ROOT": str(root),
        "CACHE_KEY": key,
        "CACHE_SIZES": json.dumps([int(s) for s in sizes]),
        "CACHE_TRAIN_CSV": manifest.get("train_csv") or "",
        "CACHE_TEST_CSV": manifest.get("test_csv") or "",
        "CACHE_TRAIN_IMAGES": manifest.get("train_images") or "",
        "CACHE_TEST_IMAGES": manifest.get("test_images") or "",
        "CACHE_ID_COLS": json.dumps({}),
    }
    public_dir = _public_dir(manifest)
    tokens = shlex.split(exec_python) or ["python3"]
    # unique name: 3 concurrent tasks may build caches in the SAME
    # second; a time-only name collides (docker rc=125 Conflict) and the
    # loser silently loses its whole cache. uuid suffix => no collision.
    cmd = [docker_bin, "run", "--rm", "--name",
           "v2_cache_%s_%s" % (int(time.time()), uuid.uuid4().hex[:8]),
           "-e", "PYTHONUNBUFFERED=1"]
    # work dir rw (writes cache); ONLY the public dir ro (never data root)
    cmd += ["-v", "%s:%s" % (work_dir_str, work_dir_str)]
    cmd += ["-v", "%s:%s:ro" % (public_dir, public_dir)]
    for k, v in env_map.items():
        cmd += ["-e", "%s=%s" % (k, v)]
    cmd += ["-w", work_dir_str, "--entrypoint", tokens[0], exec_image]
    cmd += tokens[1:] + [str(script_path)]
    if run_fn is not None:
        rc, out, err = run_fn(cmd)
    else:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace",
                                  timeout=timeout)
            rc, out, err = proc.returncode, proc.stdout or "", proc.stderr or ""
        except subprocess.TimeoutExpired:
            raise RuntimeError("image cache build timed out after %ss" % timeout)
    built = cache_map(work_dir, manifest, sizes)
    if rc != 0 or not built:
        raise RuntimeError("image cache build failed rc=%s: %s"
                           % (rc, ((out or "") + (err or ""))[-800:]))
    for s, d in built.items():
        meta = json.loads((Path(d) / "meta.json").read_text(encoding="utf-8"))
        result[int(s)] = {"status": "built", "dir": d, "key": key,
                          "rows_train": meta.get("rows_train")
                          or meta.get("rows"),
                          "rows_test": meta.get("rows_test")
                          or meta.get("rows")}
    return result


def ensure_image_cache(work_dir, manifest: dict, image_size: int = _DEFAULT_SIZE,
                       docker_bin: str = "docker", exec_image: str = "",
                       exec_python: str = "", run_fn=None,
                       force: bool = False, timeout: int = _BUILD_TIMEOUT) -> dict:
    """Single-size wrapper (kept for tests / legacy callers)."""
    res = ensure_image_caches(work_dir, manifest, sizes=[int(image_size)],
                              docker_bin=docker_bin, exec_image=exec_image,
                              exec_python=exec_python, run_fn=run_fn,
                              force=force, timeout=timeout)
    info = res.get(int(image_size)) or {}
    return {"status": info.get("status", "built"),
            "dir": info.get("dir", ""),
            "key": info.get("key", ""),
            "rows_train": info.get("rows_train"),
            "rows_test": info.get("rows_test")}