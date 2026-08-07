# -*- coding: utf-8 -*-
"""pact/evaluator.py - TrustedEvaluator: independent metric recomputation.

Deterministic, auditable, dependency-free metric recomputation from the
CandidateBundle's OOF predictions. The evaluator does NOT trust metric
strings printed by candidate code and never parses execution logs as a
metric source. A trial without a well-formed oof.csv has no metric
(metric=None), which fails the trial at the verdict layer.

OOF contract v2 (oof.csv):
  - primary columns: `true` (aliases y_true/target) and `pred` (aliases
    prediction/y_pred); values are numeric, class labels or strings
    depending on the metric family (see metrics_registry.OOF_GUIDE)
  - probability families (logloss, weighted_logloss, kl_div,
    mean_auc_multilabel): one `pred_<class>` column per class holding
    probabilities; `true` holds the class label; kl_div additionally uses
    `true_<class>` columns for the target distribution
  - multi-output families (spearman, rmsle, mae, mean_angular_error):
    optional `true_<name>` / `pred_<name>` columns; each column is scored
    and the mean is returned
  - grouped families (map_at_k, label_ranking_ap): a `query` column groups
    rows; `true` marks positive candidates (0/1), `pred` is a score
  - flat families (dice, iou_mean): one row per pixel, `true`/`pred` 0/1
"""
import csv
import math
from pathlib import Path
from typing import Optional, Tuple

from v2_contracts import CandidateBundle, EvaluatorReceipt, new_id, now_iso


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
def _read_pairs(path: Path) -> Tuple[list, list]:
    """Backward-compatible: primary (true, pred) numeric pairs."""
    data = _read_columns(path)
    t, p = data["primary"]
    out_t, out_p = [], []
    for tv, pv in zip(t, p):
        tn, pn = _num(tv), _num(pv)
        if tn is not None and pn is not None:
            out_t.append(tn)
            out_p.append(pn)
    return out_t, out_p


def _read_columns(path: Path) -> dict:
    """Parse oof.csv into primary/columns/query structures (values as str).

    Returns {"primary": (true_list, pred_list),
             "columns": {name: (true_list, pred_list)},   # pred_<x>/true_<x>
             "query": list or None}
    """
    primary_t, primary_p = [], []
    columns = {}
    query = None
    try:
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh)
            fieldnames = list(reader.fieldnames or [])
            pred_cols = [f for f in fieldnames if f.startswith("pred_")]
            true_cols = [f for f in fieldnames if f.startswith("true_")]
            qcol = next((f for f in fieldnames if str(f).strip().lower() == "query"),
                        None)
            if qcol:
                query = []
            for row in reader:
                t = (row.get("true") or row.get("y_true") or row.get("target") or "").strip()
                p = (row.get("pred") or row.get("prediction") or row.get("y_pred") or "").strip()
                primary_t.append(t)
                primary_p.append(p)
                for name in pred_cols:
                    columns.setdefault(name, [[], []])[1].append(
                        (row.get(name) or "").strip())
                for name in true_cols:
                    columns.setdefault(name, [[], []])[0].append(
                        (row.get(name) or "").strip())
                if qcol:
                    query.append((row.get(qcol) or "").strip())
    except OSError:
        pass
    return {"primary": (primary_t, primary_p), "columns": columns, "query": query}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _label_key(x) -> str:
    return str(x).strip()


def _label_eq(t, p) -> bool:
    tn, pn = _num(t), _num(p)
    if tn is not None and pn is not None:
        return abs(tn - pn) < 0.5
    return _label_key(t) == _label_key(p)


def _numeric_pairs(true_vals, pred_vals):
    out_t, out_p = [], []
    for t, p in zip(true_vals, pred_vals):
        tn, pn = _num(t), _num(p)
        if tn is not None and pn is not None:
            out_t.append(tn)
            out_p.append(pn)
    return out_t, out_p


def _confusion(true_vals, pred_vals):
    keys = sorted({_label_key(t) for t in true_vals}
                  | {_label_key(p) for p in pred_vals})
    idx = {k: i for i, k in enumerate(keys)}
    k = len(keys)
    mat = [[0] * k for _ in range(k)]
    for t, p in zip(true_vals, pred_vals):
        mat[idx[_label_key(t)]][idx[_label_key(p)]] += 1
    return mat, keys


def _binary_counts(true_vals, pred_vals):
    """TP/FP/FN/TN with True/1 treated as positive."""
    def pos(x):
        n = _num(x)
        if n is not None:
            return n > 0.5
        return _label_key(x).lower() in ("true", "1", "yes", "positive")
    tp = fp = fn = tn = 0
    for t, p in zip(true_vals, pred_vals):
        pt, pp = pos(t), pos(p)
        if pt and pp: tp += 1
        elif pt and not pp: fn += 1
        elif not pt and pp: fp += 1
        else: tn += 1
    return tp, fp, fn, tn


def _fbeta_from_counts(tp, fp, fn, beta):
    if tp == 0:
        return 0.0
    b2 = beta * beta
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    if prec + rec == 0:
        return 0.0
    return (1 + b2) * prec * rec / (b2 * prec + rec)


# --------------------------------------------------------------------------
# single-pair metrics (values may be numeric or strings)
# --------------------------------------------------------------------------
def _accuracy(true_vals, pred_vals):
    if not true_vals:
        return None
    return sum(1 for t, p in zip(true_vals, pred_vals) if _label_eq(t, p)) / len(true_vals)


def _per_class_f1(true_vals, pred_vals):
    mat, keys = _confusion(true_vals, pred_vals)
    k = len(keys)
    out = []
    for i in range(k):
        tp = mat[i][i]
        fp = sum(mat[j][i] for j in range(k)) - tp
        fn = sum(mat[i][j] for j in range(k)) - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        out.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return out


def _f1_macro(true_vals, pred_vals):
    f1s = _per_class_f1(true_vals, pred_vals)
    if not f1s:
        return None
    return sum(f1s) / len(f1s)


def _f1_micro(true_vals, pred_vals):
    mat, keys = _confusion(true_vals, pred_vals)
    k = len(keys)
    tp = sum(mat[i][i] for i in range(k))
    fp = sum(sum(mat[j][i] for j in range(k)) - mat[i][i] for i in range(k))
    fn = sum(sum(mat[i][j] for j in range(k)) - mat[i][i] for i in range(k))
    if tp + fp + fn == 0:
        return 0.0
    return 2 * tp / (2 * tp + fp + fn)


def _f1_binary(true_vals, pred_vals):
    tp, fp, fn, _ = _binary_counts(true_vals, pred_vals)
    return _fbeta_from_counts(tp, fp, fn, 1.0)


def _f0_5(true_vals, pred_vals):
    tp, fp, fn, _ = _binary_counts(true_vals, pred_vals)
    return _fbeta_from_counts(tp, fp, fn, 0.5)


def _mcc(true_vals, pred_vals):
    tp, fp, fn, tn = _binary_counts(true_vals, pred_vals)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return 0.0
    return (tp * tn - fp * fn) / denom


def _rmse(true_vals, pred_vals):
    t, p = _numeric_pairs(true_vals, pred_vals)
    if not t:
        return None
    sq = sum((a - b) ** 2 for a, b in zip(t, p))
    return math.sqrt(sq / len(t))


def _mae(true_vals, pred_vals):
    t, p = _numeric_pairs(true_vals, pred_vals)
    if not t:
        return None
    return sum(abs(a - b) for a, b in zip(t, p)) / len(t)


def _log_mae(true_vals, pred_vals):
    t, p = _numeric_pairs(true_vals, pred_vals)
    if not t:
        return None
    return sum(abs(math.log1p(abs(a)) - math.log1p(abs(b)))
               for a, b in zip(t, p)) / len(t)


def _rmsle(true_vals, pred_vals):
    t, p = _numeric_pairs(true_vals, pred_vals)
    if not t:
        return None
    sq = sum((math.log1p(max(0.0, a)) - math.log1p(max(0.0, b))) ** 2
             for a, b in zip(t, p))
    return math.sqrt(sq / len(t))


def _pearson(true_vals, pred_vals):
    t, p = _numeric_pairs(true_vals, pred_vals)
    n = len(t)
    if n < 2:
        return None
    mt = sum(t) / n
    mp = sum(p) / n
    cov = sum((a - mt) * (b - mp) for a, b in zip(t, p))
    var_t = sum((a - mt) ** 2 for a in t)
    var_p = sum((b - mp) ** 2 for b in p)
    if var_t == 0 or var_p == 0:
        return None
    return cov / math.sqrt(var_t * var_p)


def _rank_average(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(true_vals, pred_vals):
    t, p = _numeric_pairs(true_vals, pred_vals)
    if len(t) < 2:
        return None
    rt = _rank_average(t)
    rp = _rank_average(p)
    return _pearson(rt, rp)


def _auc(true_vals, pred_vals):
    t, p = _numeric_pairs(true_vals, pred_vals)
    pairs = [(b, a) for a, b in zip(t, p) if a in (0.0, 1.0)]
    if len(pairs) < 2:
        return None
    n_pos = sum(1 for _, a in pairs if a > 0.5)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    pairs.sort(key=lambda x: x[0])
    n = len(pairs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rank_sum_pos = sum(r for r, (_, a) in zip(ranks, pairs) if a > 0.5)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _qwk(true_vals, pred_vals):
    t, p = _numeric_pairs(true_vals, pred_vals)
    pairs = [(round(a), round(b)) for a, b in zip(t, p)]
    if not pairs:
        return None
    classes = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    if len(classes) < 2:
        return None
    n = len(pairs)
    k = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    mat = [[0.0] * k for _ in range(k)]
    hist_t = [0] * k
    hist_p = [0] * k
    for a, b in pairs:
        i, j = idx[a], idx[b]
        mat[i][j] += 1
        hist_t[i] += 1
        hist_p[j] += 1

    def w(i, j):
        return (i - j) ** 2 / (k - 1) ** 2

    w_obs = sum(w(i, j) * mat[i][j] for i in range(k) for j in range(k)) / n
    w_exp = sum(w(i, j) * hist_t[i] * hist_p[j] / n
                for i in range(k) for j in range(k)) / n
    if w_exp == 1.0:
        return 0.0
    return 1.0 - w_obs / w_exp


def _kendall_tau(true_vals, pred_vals):
    """Kendall tau matching the official AI4Code grader.

    Official formula (grade.py): ranks = [gt.index(x) for x in pred],
    tau = 1 - 4 * inversions / (n * (n - 1)), where inversions counts
    strict inversions (equal ranks are not inversions).

    OOF proxy: each row is one item with (true_score, pred_score); pred
    score is the predicted position in the order (smaller = earlier), so the
    predicted order is pred scores ascending (ties broken by true score
    ascending, so ties are never penalized).
    """
    t, p = _numeric_pairs(true_vals, pred_vals)
    n = len(t)
    if n < 2:
        return None

    def merge_inversions(arr):
        # strict inversions (arr[i] > arr[j] for i<j)
        tmp = [0] * len(arr)

        def sort_count(lo, hi):
            if hi - lo <= 1:
                return 0
            mid = (lo + hi) // 2
            inv = sort_count(lo, mid) + sort_count(mid, hi)
            i, j, k = lo, mid, lo
            while i < mid and j < hi:
                if arr[i] <= arr[j]:
                    tmp[k] = arr[i]; i += 1
                else:
                    tmp[k] = arr[j]; j += 1
                    inv += mid - i
                k += 1
            while i < mid:
                tmp[k] = arr[i]; i += 1; k += 1
            while j < hi:
                tmp[k] = arr[j]; j += 1; k += 1
            arr[lo:hi] = tmp[lo:hi]
            return inv

        return sort_count(0, len(arr))

    order = sorted(range(n), key=lambda i: (p[i], t[i]))
    ts = [t[i] for i in order]
    inv = merge_inversions(ts)
    return 1.0 - 4.0 * inv / (n * (n - 1))


def _mean_angular_error(true_vals, pred_vals):
    t, p = _numeric_pairs(true_vals, pred_vals)
    if not t:
        return None
    return sum(abs(((a - b + 180.0) % 360.0) - 180.0)
               for a, b in zip(t, p)) / len(t)


def _haversine(true_vals, pred_vals):
    R = 6371.0
    dists = []
    for t, p in zip(true_vals, pred_vals):
        def parse(v):
            parts = str(v).replace(";", ",").replace(" ", ",").split(",")
            if len(parts) < 2:
                return None
            try:
                return float(parts[0]), float(parts[1])
            except ValueError:
                return None
        a = parse(t)
        b = parse(p)
        if a is None or b is None:
            continue
        lat1, lon1 = math.radians(a[0]), math.radians(a[1])
        lat2, lon2 = math.radians(b[0]), math.radians(b[1])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        h = (math.sin(dlat / 2) ** 2
             + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
        dists.append(2 * R * math.asin(math.sqrt(min(1.0, h))))
    if not dists:
        return None
    return sum(dists) / len(dists)


def _levenshtein(true_vals, pred_vals):
    dists = []
    for t, p in zip(true_vals, pred_vals):
        a, b = _label_key(t), _label_key(p)
        if not a and not b:
            dists.append(0.0)
            continue
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i] + [0] * len(b)
            for j, cb in enumerate(b, 1):
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                             prev[j - 1] + (0 if ca == cb else 1))
            prev = cur
        dists.append(float(prev[-1]))
    if not dists:
        return None
    return sum(dists) / len(dists)


def _jaccard(true_vals, pred_vals):
    scores = []
    for t, p in zip(true_vals, pred_vals):
        a = set(_label_key(t).split())
        b = set(_label_key(p).split())
        if not a and not b:
            scores.append(1.0)
            continue
        inter = len(a & b)
        scores.append(inter / (len(a | b)) if (len(a) + len(b)) else 0.0)
    if not scores:
        return None
    return sum(scores) / len(scores)


def _dice(true_vals, pred_vals):
    tp = fp = fn = 0
    for t, p in zip(true_vals, pred_vals):
        tn, pn = _num(t), _num(p)
        a = (tn if tn is not None else (1.0 if _label_key(t).lower() in ("true", "1") else 0.0))
        b = (pn if pn is not None else (1.0 if _label_key(p).lower() in ("true", "1") else 0.0))
        if a > 0.5 and b > 0.5: tp += 1
        elif a > 0.5: fn += 1
        elif b > 0.5: fp += 1
    if tp + fp + fn == 0:
        return 1.0 if true_vals else None
    return 2 * tp / (2 * tp + fp + fn)


def _dice_global(true_vals, pred_vals):
    """Global dice over all flattened rows (contrails official semantics)."""
    tp = fp = fn = 0
    for t, p in zip(true_vals, pred_vals):
        tn, pn = _num(t), _num(p)
        a = (tn if tn is not None else (1.0 if _label_key(t).lower() in ("true", "1") else 0.0))
        b = (pn if pn is not None else (1.0 if _label_key(p).lower() in ("true", "1") else 0.0))
        if a > 0.5 and b > 0.5: tp += 1
        elif a > 0.5: fn += 1
        elif b > 0.5: fp += 1
    if tp + fp + fn == 0:
        return 1.0 if true_vals else None
    return 2 * tp / (2 * tp + fp + fn)


def _iou_mean(true_vals, pred_vals):
    t, p = _numeric_pairs(true_vals, pred_vals)
    if not t:
        return None
    scores = []
    for thr in [x / 100.0 for x in range(50, 100, 5)]:
        inter = union = 0
        for a, b in zip(t, p):
            ba = 1.0 if a > 0.5 else 0.0
            bb = 1.0 if b > thr else 0.0
            inter += ba * bb
            union += ba + bb - ba * bb
        scores.append(inter / union if union else 1.0)
    return sum(scores) / len(scores)


def _map_at_k(query, true_vals, pred_vals, k):
    groups = {}
    for q, t, p in zip(query, true_vals, pred_vals):
        groups.setdefault(q, []).append((_num(p) if _num(p) is not None else 0.0, _label_key(t)))
    aps = []
    for q, rows in groups.items():
        rows.sort(key=lambda x: -x[0])
        pos_total = sum(1 for _, t in rows if _label_key(t).lower() in ("1", "true", "yes"))
        if pos_total == 0:
            continue
        hits = 0
        ap = 0.0
        for rank, (_, t) in enumerate(rows[:k], start=1):
            if _label_key(t).lower() in ("1", "true", "yes"):
                hits += 1
                ap += hits / rank
        aps.append(ap / min(k, pos_total))
    if not aps:
        return None
    return sum(aps) / len(aps)


def _label_ranking_ap(query, true_vals, pred_vals):
    groups = {}
    for q, t, p in zip(query, true_vals, pred_vals):
        groups.setdefault(q, []).append((_num(p) if _num(p) is not None else 0.0, _label_key(t)))
    aps = []
    for q, rows in groups.items():
        rows.sort(key=lambda x: -x[0])
        pos_idx = [i for i, (_, t) in enumerate(rows)
                   if _label_key(t).lower() in ("1", "true", "yes")]
        if not pos_idx:
            continue
        total = 0.0
        for i in pos_idx:
            # standard LRAP: (positives with score >= current) / (rank of current)
            cur_score = rows[i][0]
            pos_ge = sum(1 for j in pos_idx if rows[j][0] >= cur_score)
            ge = sum(1 for s, _ in rows if s >= cur_score)
            if ge:
                total += pos_ge / ge
        aps.append(total / len(pos_idx))
    if not aps:
        return None
    return sum(aps) / len(aps)


def _label_ranking_ap_multilabel(rows, true_cols, pred_cols):
    """Weighted multilabel LRAP (freesound official lwlrap semantics).

    rows: list of dicts (one per sample); true_cols/pred_cols: column names.
    Per-sample LRAP over the relevant labels, then weighted average by the
    number of relevant labels (sklearn sample_weight behavior).
    """
    weighted = 0.0
    total_weight = 0.0
    for row in rows:
        tv = []
        pv = []
        for tc, pc in zip(true_cols, pred_cols):
            tt = _num(row.get(tc) or "")
            pp = _num(row.get(pc) or "")
            tv.append(tt if tt is not None else 0.0)
            pv.append(pp if pp is not None else 0.0)
        pos = [i for i, v in enumerate(tv) if v > 0.5]
        if not pos:
            continue
        total = 0.0
        for i in pos:
            cur = pv[i]
            pos_ge = sum(1 for j in pos if pv[j] >= cur)
            ge = sum(1 for s in pv if s >= cur)
            if ge:
                total += pos_ge / ge
        lrap = total / len(pos)
        weighted += lrap * len(pos)
        total_weight += len(pos)
    if total_weight <= 0:
        return None
    return weighted / total_weight


# --------------------------------------------------------------------------
# probability metrics (pred_<class> columns)
# --------------------------------------------------------------------------
def _logloss(true_vals, prob_by_class, classes):
    eps = 1e-15
    losses = []
    for i, t in enumerate(true_vals):
        key = _label_key(t)
        if key not in prob_by_class:
            continue
        p = _num(prob_by_class[key][i]) if i < len(prob_by_class[key]) else None
        if p is None:
            continue
        p = min(max(p, eps), 1.0 - eps)
        losses.append(-math.log(p))
    if not losses:
        return None
    return sum(losses) / len(losses)


def _binary_logloss(true_vals, pred_vals):
    eps = 1e-15
    t, p = _numeric_pairs(true_vals, pred_vals)
    losses = []
    for a, b in zip(t, p):
        if a not in (0.0, 1.0):
            continue
        pb = min(max(b, eps), 1.0 - eps)
        losses.append(-math.log(pb if a > 0.5 else 1.0 - pb))
    if not losses:
        return None
    return sum(losses) / len(losses)


def _weighted_logloss(true_vals, prob_by_class, classes):
    eps = 1e-15
    per_class = []
    for cls in classes:
        key = _label_key(cls)
        losses = []
        for i, t in enumerate(true_vals):
            if _label_key(t) != key:
                continue
            p = _num(prob_by_class[key][i]) if i < len(prob_by_class[key]) else None
            if p is None:
                continue
            p = min(max(p, eps), 1.0 - eps)
            losses.append(-math.log(p))
        if losses:
            per_class.append(sum(losses) / len(losses))
    if not per_class:
        return None
    return sum(per_class) / len(per_class)


def _kl_div(true_by_class, pred_by_class, classes):
    eps = 1e-15
    n = 0
    total = 0.0
    for i in range(len(next(iter(pred_by_class.values())))):
        tvals = []
        pvals = []
        for cls in classes:
            tk = _label_key(cls)
            tv = _num(true_by_class[tk][i]) if i < len(true_by_class[tk]) else None
            pv = _num(pred_by_class[tk][i]) if i < len(pred_by_class[tk]) else None
            tvals.append(tv if tv is not None else 0.0)
            pvals.append(pv if pv is not None else 0.0)
        st = sum(tvals)
        sp = sum(pvals)
        if st <= 0 or sp <= 0:
            continue
        tvals = [v / st for v in tvals]
        pvals = [min(max(v / sp, eps), 1.0) for v in pvals]
        total += sum(tv * math.log(tv / pv)
                     for tv, pv in zip(tvals, pvals) if tv > 0)
        n += 1
    if not n:
        return None
    return total / n


def _mean_auc_multilabel(true_vals, prob_by_class, classes, true_by_class=None):
    aucs = []
    for cls in classes:
        key = _label_key(cls)
        pv = prob_by_class[key]
        if true_by_class and ("true_" + key) in true_by_class:
            tv = true_by_class["true_" + key]
        else:
            tv = [1.0 if _label_eq(t, cls) else 0.0 for t in true_vals]
        a = _auc(tv, pv)
        if a is not None:
            aucs.append(a)
    if not aucs:
        return None
    return sum(aucs) / len(aucs)


# --------------------------------------------------------------------------
# TrustedEvaluator
# --------------------------------------------------------------------------
_SINGLE_PAIR = {
    "accuracy": _accuracy,
    "f1_macro": _f1_macro,
    "f1_micro": _f1_micro,
    "f1_binary": _f1_binary,
    "f0_5": _f0_5,
    "mcc": _mcc,
    "auc": _auc,
    "qwk": _qwk,
    "rmse": _rmse,
    "mae": _mae,
    "log_mae": _log_mae,
    "rmsle": _rmsle,
    "spearman": _spearman,
    "pearson": _pearson,
    "kendall_tau": _kendall_tau,
    "mean_angular_error": _mean_angular_error,
    "haversine": _haversine,
    "levenshtein": _levenshtein,
    "jaccard": _jaccard,
    "dice": _dice,
    "iou_mean": _iou_mean,
}

_MULTI_OUTPUT = {"spearman", "rmsle", "mae", "mean_angular_error"}


# v2.5.0 declarative metric dispatch: EVERY non-pair metric has a handler
# row below (uniform signature). Adding a metric = adding one row; there is
# no if/elif chain naming metric families anywhere in evaluation.
def _handle_binary_logloss(self, data):
    t, p = data["primary"]
    return _binary_logloss(t, p)


def _handle_probability(self, data):
    return self._probability_metric(data)


def _handle_map_at_k(self, data):
    t, p = data["primary"]
    if not data["query"]:
        return None
    return _map_at_k(data["query"], t, p,
                     int(self.metric_params.get("k", 5)))


def _handle_label_ranking_ap(self, data):
    t, p = data["primary"]
    if not data["query"]:
        return None
    pred_cols = sorted(k for k in data["columns"] if k.startswith("pred_"))
    true_cols = sorted(k for k in data["columns"] if k.startswith("true_"))
    if pred_cols:
        if not true_cols:
            return None
        groups = {}
        for i, q in enumerate(data["query"]):
            groups.setdefault(q, []).append(
                {k: data["columns"][k][0][i] if k.startswith("true_")
                 else data["columns"][k][1][i] for k in data["columns"]})
        vals = [_label_ranking_ap_multilabel(rows, true_cols, pred_cols)
                for rows in groups.values()]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None
    return _label_ranking_ap(data["query"], t, p)


def _handle_dice_global(self, data):
    t, p = data["primary"]
    return _dice_global(t, p)


_COMPUTE_HANDLERS = {
    "binary_logloss": _handle_binary_logloss,
    "logloss": _handle_probability,
    "weighted_logloss": _handle_probability,
    "kl_div": _handle_probability,
    "mean_auc_multilabel": _handle_probability,
    "map_at_k": _handle_map_at_k,
    "label_ranking_ap": _handle_label_ranking_ap,
    "dice_global": _handle_dice_global,
}


def _prob_logloss(t, prob_by_class, classes, true_by_class):
    return _logloss(t, prob_by_class, classes)


def _prob_weighted(t, prob_by_class, classes, true_by_class):
    return _weighted_logloss(t, prob_by_class, classes)


def _prob_kl(t, prob_by_class, classes, true_by_class):
    return _kl_div(true_by_class, prob_by_class, classes)


def _prob_auc(t, prob_by_class, classes, true_by_class):
    return _mean_auc_multilabel(t, prob_by_class, classes, true_by_class)


_PROBABILITY_HANDLERS = {
    "logloss": _prob_logloss,
    "weighted_logloss": _prob_weighted,
    "kl_div": _prob_kl,
    "mean_auc_multilabel": _prob_auc,
}


class TrustedEvaluator:
    """Recomputes metrics from bundle artifacts (deterministic, auditable).

    metric_name / metric_direction / metric_alignment / metric_label /
    metric_params come from metrics_registry.get_metric_spec(competition).
    """

    def __init__(self, metric_name: str = "accuracy",
                 metric_direction: str = "higher_is_better",
                 metric_alignment: str = "exact",
                 metric_label: str = "accuracy",
                 metric_params: Optional[dict] = None):
        self.metric_name = metric_name
        self.metric_direction = metric_direction
        self.metric_alignment = metric_alignment
        self.metric_label = metric_label or metric_name
        self.metric_params = dict(metric_params or {})

    def evaluate(self, bundle: Optional[CandidateBundle],
                 stdout: str = "", stderr: str = "", returncode: int = 0,
                 submission_path: str = "") -> EvaluatorReceipt:
        metric = None
        evidence = ""
        evaluator = "none"

        if bundle is not None and bundle.oof_path and Path(bundle.oof_path).is_file():
            data = _read_columns(Path(bundle.oof_path))
            t, p = data["primary"]
            if not t:
                evidence = "OOF file empty or unparseable (true,pred required)"
            else:
                metric = self._compute(data)
                evaluator = "trusted_recompute_%s" % self.metric_name
                if metric is None:
                    evidence = ("OOF columns/values incompatible with metric "
                                "'%s' (see metrics_registry.OOF_GUIDE)" % self.metric_name)
                else:
                    evidence = "recomputed from OOF %s rows" % len(t)
        else:
            evidence = "no OOF predictions (log-parse is never a trusted metric)"

        if metric is None and returncode != 0:
            evidence = "execution failed rc=%s" % returncode

        return EvaluatorReceipt(
            receipt_id=new_id("eval"),
            trial_id=(bundle.trial_id if bundle else ""),
            metric=metric,
            metric_name=self.metric_name,
            metric_direction=self.metric_direction,
            metric_alignment=self.metric_alignment,
            metric_label=self.metric_label,
            evaluator=evaluator,
            evidence=evidence,
            artifact_hash=(bundle.bundle_hash if bundle else ""),
        )

    def _compute(self, data: dict):
        name = self.metric_name
        t, p = data["primary"]
        if name in _SINGLE_PAIR:
            if name in _MULTI_OUTPUT and data["columns"]:
                vals = []
                for cname, (ct, cp) in data["columns"].items():
                    if cname.startswith("pred_") or cname.startswith("true_"):
                        v = _SINGLE_PAIR[name](ct, cp)
                        if v is not None:
                            vals.append(v)
                if vals:
                    return sum(vals) / len(vals)
            return _SINGLE_PAIR[name](t, p)
        handler = _COMPUTE_HANDLERS.get(name)
        if handler is not None:
            return handler(self, data)
        return None

    def _probability_metric(self, data: dict):
        t, _ = data["primary"]
        cols = data["columns"]
        pred_cols = sorted(c for c in cols if c.startswith("pred_"))
        if not pred_cols:
            return None
        classes = [c[len("pred_"):] for c in pred_cols]
        prob_by_class = {cls: cols["pred_" + cls][1] for cls in classes}
        true_by_class = {c[len("true_"):]: cols[c][0]
                         for c in cols if c.startswith("true_")}
        handler = _PROBABILITY_HANDLERS.get(self.metric_name)
        if handler is None:
            return None
        return handler(t, prob_by_class, classes, true_by_class)
