# -*- coding: utf-8 -*-
"""deep_profile.py - v2.4 M1 deep diagnostics (stdlib only, measured).

Extends the shallow AnalysisProfile with MEASURED evidence so HERA's
planner/prioritizer can read data instead of guessing:

  target_diag:   class count / top-1 share / entropy / skew / unique ratio
  feature_diag:  missing rates, cardinality, constant + duplicate columns,
                 numeric share, high-cardinality columns
  order_diag:    monotonic row-id, id-target correlation, time range

Bound: at most sample_rows rows and max_cols columns; every helper is
O(n) or O(n log n); any failure degrades to a partial report - the analyzer
never crashes on diagnostics. Deterministic: no RNG, sorted outputs.

Run: python -c "import deep_profile; print(deep_profile.build_deep_diagnostics('train.csv'))"
"""
import csv
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from data_layout import table_delimiter

DEFAULT_SAMPLE_ROWS = 8000
DEFAULT_MAX_COLS = 50
_ID_NAME_HINTS = ("id", "Id", "ID", "index", "row_id", "rowid", "uid", "key",
                  "row", "sample_id", "sampleid")
_TIME_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
                 "%Y/%m/%d", "%Y%m%d", "%d/%m/%Y", "%m/%d/%Y")


def _read_sample(path, sample_rows: int, max_cols: int):
    header: List[str] = []
    rows: List[List[str]] = []
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh, delimiter=table_delimiter(path))
        for i, row in enumerate(reader):
            if i == 0:
                header = [str(c) for c in row][:max_cols]
                continue
            if not row or not any(str(c).strip() for c in row):
                continue
            rows.append(row[:len(header)])
            if len(rows) >= sample_rows:
                break
    return header, rows


def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx <= 0 or sy <= 0:
        return None
    return cov / (sx * sy)


def _target_diag(values) -> dict:
    nonempty = [str(v).strip() for v in values if str(v).strip()]
    n = len(nonempty)
    if n == 0:
        return {"n_classes": 0, "top1_share": None, "entropy_bits": None,
                "skew": None, "unique_ratio": None, "numeric": False}
    nums = []
    numeric = True
    for v in nonempty:
        try:
            nums.append(float(v))
        except ValueError:
            numeric = False
            break
    if numeric:
        mean = sum(nums) / n
        srt = sorted(nums)
        median = srt[n // 2] if n % 2 else (srt[n // 2 - 1] + srt[n // 2]) / 2.0
        std = math.sqrt(sum((x - mean) ** 2 for x in nums) / n) or 1e-9
        return {"n_classes": len(set(nonempty)),
                "top1_share": None, "entropy_bits": None,
                "skew": round((mean - median) / std, 4),
                "unique_ratio": round(len(set(nonempty)) / n, 4),
                "numeric": True}
    counts: Dict[str, int] = {}
    for v in nonempty:
        counts[v] = counts.get(v, 0) + 1
    top1 = max(counts.values()) / n
    entropy = -sum((c / n) * math.log2(c / n) for c in counts.values())
    return {"n_classes": len(counts),
            "top1_share": round(top1, 4),
            "entropy_bits": round(entropy, 4),
            "skew": None,
            "unique_ratio": round(len(counts) / n, 4),
            "numeric": False}


def _feature_diag(header, rows, target_col, id_col) -> dict:
    n = len(rows)
    out = {"n_columns": 0, "numeric_share": None, "constant_cols": [],
           "duplicate_cols": [], "high_card_cols": [], "missing_rates": {}}
    if n == 0 or not header:
        return out
    skip = {str(target_col or ""), str(id_col or "")}
    cols = [c for c in header if c not in skip]
    out["n_columns"] = len(cols)
    if not cols:
        return out
    idx = {c: i for i, c in enumerate(header)}
    numeric_cols = 0
    card: Dict[str, int] = {}
    missing: Dict[str, float] = {}
    value_tuples: Dict[Tuple[str, ...], List[str]] = {}
    for cname in cols:
        ci = idx[cname]
        vals = [str(r[ci]).strip() if ci < len(r) else "" for r in rows]
        nonempty = [v for v in vals if v]
        missing[cname] = round(1.0 - len(nonempty) / n, 4)
        card[cname] = len(set(nonempty))
        if len(nonempty) >= 10:
            ok = 0
            for v in nonempty[:200]:
                try:
                    float(v)
                    ok += 1
                except ValueError:
                    pass
            if ok / len(nonempty) >= 0.8:
                numeric_cols += 1
        if len(nonempty) > 1:
            value_tuples.setdefault(tuple(vals), []).append(cname)
    out["numeric_share"] = round(numeric_cols / len(cols), 4)
    out["constant_cols"] = [c for c in cols if card[c] <= 1]
    out["missing_rates"] = dict(
        sorted(missing.items(), key=lambda kv: -kv[1])[:10])
    hi = sorted(((c, card[c]) for c in cols), key=lambda kv: -kv[1])
    out["high_card_cols"] = [[c, k] for c, k in hi[:10]
                             if k >= max(2, int(n * 0.5))]
    dup = [g for g in value_tuples.values() if len(g) > 1]
    dup.sort(key=len, reverse=True)
    out["duplicate_cols"] = dup[:5]
    return out


def _guess_id_col(header) -> str:
    for name in header:
        low = str(name).strip().lower()
        if low in _ID_NAME_HINTS or low.endswith("_id") or low.endswith("-id"):
            return str(name)
    return ""


def _parse_time(v: str):
    try:
        f = float(v)
        if 1e8 <= f <= 2e9:
            return datetime.utcfromtimestamp(f)
    except ValueError:
        pass
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(v, fmt)
        except ValueError:
            continue
    return None


def _order_diag(header, rows, target_col, id_col, time_col) -> dict:
    out = {"id_monotonic": None, "id_target_corr": None,
           "time_present": False, "time_min": None, "time_max": None}
    idx = {c: i for i, c in enumerate(header)}
    if id_col and id_col in idx:
        ints = []
        bad = False
        for r in rows:
            ci = idx[id_col]
            try:
                ints.append(int(str(r[ci]).strip()))
            except (ValueError, IndexError):
                bad = True
                break
        if not bad and len(ints) > 1:
            out["id_monotonic"] = all(ints[i] < ints[i + 1]
                                      for i in range(len(ints) - 1))
    if target_col and target_col in idx:
        tci = idx[target_col]
        pairs = []
        for i, r in enumerate(rows):
            try:
                pairs.append((i, float(str(r[tci]).strip())))
            except (ValueError, IndexError):
                pass
        if len(pairs) >= 10:
            corr = _pearson([p[0] for p in pairs], [p[1] for p in pairs])
            out["id_target_corr"] = round(corr, 4) if corr is not None else None
    if time_col and time_col in idx:
        tci = idx[time_col]
        stamps = []
        for r in rows:
            v = str(r[tci]).strip() if tci < len(r) else ""
            if not v:
                continue
            ts = _parse_time(v)
            if ts is not None:
                stamps.append(ts)
        if stamps and len(stamps) >= max(2, len(rows) // 2):
            out["time_present"] = True
            out["time_min"] = min(stamps).strftime("%Y-%m-%d")
            out["time_max"] = max(stamps).strftime("%Y-%m-%d")
    return out


def build_deep_diagnostics(train_path, target_column: str = "",
                           time_column: str = "", id_column: str = "",
                           sample_rows: int = DEFAULT_SAMPLE_ROWS,
                           max_cols: int = DEFAULT_MAX_COLS) -> dict:
    """Measured deep diagnostics for one train table (fail-open)."""
    try:
        header, rows = _read_sample(train_path, sample_rows, max_cols)
        if not header:
            return {"sampled_rows": 0, "error": "no header"}
        tcol = target_column or (header[-1] if header else "")
        icol = id_column or _guess_id_col(header)
        tvals = []
        if tcol and tcol in header:
            tci = header.index(tcol)
            tvals = [str(r[tci]) if tci < len(r) else "" for r in rows]
        report = {
            "sampled_rows": len(rows),
            "target_diag": _target_diag(tvals),
            "feature_diag": _feature_diag(header, rows, tcol, icol),
            "order_diag": _order_diag(header, rows, tcol, icol, time_column),
        }
        return report
    except Exception as exc:  # fail-open by contract
        return {"sampled_rows": 0,
                "error": "%s: %s" % (type(exc).__name__, str(exc)[:200])}
