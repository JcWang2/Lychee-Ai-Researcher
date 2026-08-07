# -*- coding: utf-8 -*-
"""v2.4.0 offline tests: M1 deep diagnostics (analysis capability, stdlib).

Covers:
  1) target diagnostics: class count / imbalance / entropy / skew /
     unique ratio (numeric + categorical targets);
  2) feature diagnostics: missing rates, cardinality, constant columns,
     duplicate columns, numeric share, high-cardinality columns;
  3) order diagnostics: monotonic row-id, id-target correlation, time
     range (epoch + ISO + date formats);
  4) fail-open contract: missing/unreadable train file and empty tables
     degrade to a report dict, never an exception;
  5) Analyzer integration: profile() populates deep_diagnostics and the
     data_notes summary; to_dict/from_dict round-trip keeps the new
     v2.4 fields (deep_diagnostics, difficulty_ladder);
  6) prompt injection: the HERA prioritizer prompt carries the measured
     diagnostics and the difficulty ladder (when present), and marks the
     ladder as not-yet-measured otherwise; the planner prompt carries the
     compact DEEP DIAGNOSTICS line.

Run: python test_v2_240.py   (from the aisci payload dir)
"""
import csv
import json
import os
import shutil
import tempfile
from pathlib import Path

from deep_profile import build_deep_diagnostics
from hera.analyzer import Analyzer, _deep_notes
from v2_contracts import AnalysisProfile

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("[OK] " + name)
    else:
        print("[FAIL] " + name + (" | " + str(detail)[:300] if detail else ""))
        FAILURES.append(name)


def _write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def test_target_diagnostics():
    tmp = Path(tempfile.mkdtemp(prefix="v240_target_"))
    try:
        p = tmp / "t.csv"
        # 9:1 imbalanced, 5 classes, numeric col, constant col, dup col
        def _label(i):
            if i < 180:
                return "a"
            if i < 188:
                return "b"
            if i < 194:
                return "c"
            if i < 198:
                return "d"
            return "e"
        _write_csv(p, ["id", "label", "num", "const", "dup"],
                   [[i, _label(i), i * 1.0, 7.0, i * 1.0]
                    for i in range(200)])
        d = build_deep_diagnostics(str(p), target_column="label",
                                   id_column="id")
        t = d.get("target_diag") or {}
        check("target: n_classes=5", t.get("n_classes") == 5, t)
        check("target: top1_share~0.9", t.get("top1_share") is not None
              and 0.85 <= t["top1_share"] <= 0.95, t)
        check("target: entropy low", t.get("entropy_bits") is not None
              and t["entropy_bits"] < 2.0, t)
        check("target: categorical (not numeric)", t.get("numeric") is False,
              t)
        f = d.get("feature_diag") or {}
        check("feature: constant col detected",
              "const" in f.get("constant_cols", []), f)
        check("feature: duplicate col pair detected",
              any("dup" in g for g in f.get("duplicate_cols", [])), f)
        check("feature: numeric_share>=0.5",
              (f.get("numeric_share") or 0) >= 0.5, f)
        o = d.get("order_diag") or {}
        check("order: id_monotonic True", o.get("id_monotonic") is True, o)
        # numeric target -> skew measured
        d2 = build_deep_diagnostics(str(p), target_column="num",
                                    id_column="id")
        t2 = d2.get("target_diag") or {}
        check("target: numeric skew measured",
              t2.get("skew") is not None and t2.get("numeric") is True, t2)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_order_and_time():
    tmp = Path(tempfile.mkdtemp(prefix="v240_time_"))
    try:
        p = tmp / "t.csv"
        rows = [["id", "y", "ts"]]
        import random
        rnd = random.Random(7)
        base = 1500000000
        y = 100.0
        for i in range(500):
            y += rnd.uniform(-1, 1)
            rows.append([i + 1, round(y, 3), base + i * 3600])
        _write_csv(p, ["id", "y", "ts"], rows[1:])
        d = build_deep_diagnostics(str(p), target_column="y",
                                   id_column="id", time_column="ts")
        o = d.get("order_diag") or {}
        check("order: epoch time detected", o.get("time_present") is True, o)
        check("order: time range min/max", bool(o.get("time_min"))
              and bool(o.get("time_max")), o)
        check("order: id monotonic", o.get("id_monotonic") is True, o)
        check("order: id_target_corr measured",
              o.get("id_target_corr") is not None, o)
        # ISO date format
        rows2 = [["id", "y", "d"]]
        for i in range(100):
            rows2.append([i, i * 0.1, "2020-%02d-15" % (i % 12 + 1)])
        _write_csv(tmp / "t2.csv", ["id", "y", "d"], rows2[1:])
        d2 = build_deep_diagnostics(str(tmp / "t2.csv"),
                                    target_column="y", id_column="id",
                                    time_column="d")
        check("order: ISO date detected",
              (d2.get("order_diag") or {}).get("time_present") is True, d2)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_fail_open_and_edges():
    tmp = Path(tempfile.mkdtemp(prefix="v240_fail_"))
    try:
        d = build_deep_diagnostics(str(tmp / "missing.csv"))
        check("fail-open: missing file -> report dict", isinstance(d, dict)
              and ("error" in d or "target_diag" in d), d)
        empty = tmp / "empty.csv"
        empty.write_text("id,x,y\n", encoding="utf-8")
        d2 = build_deep_diagnostics(str(empty), target_column="y")
        check("fail-open: empty table -> no crash", isinstance(d2, dict), d2)
        one = tmp / "one.csv"
        _write_csv(one, ["a", "b"], [["1", "x"]])
        d3 = build_deep_diagnostics(str(one), target_column="b")
        check("fail-open: single row -> no crash", isinstance(d3, dict), d3)
        rle = tmp / "rle.csv"
        _write_csv(rle, ["id", "mask"], [["i1", "1 3 5 2"], ["i2", ""],
                                         ["i3", "2 4"]])
        d4 = build_deep_diagnostics(str(rle), target_column="mask")
        t4 = (d4.get("target_diag") or {})
        check("rle strings not treated as numeric",
              t4.get("numeric") is False, t4)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _make_layout(tmp: Path):
    pub = tmp / "prepared" / "public"
    priv = tmp / "prepared" / "private"
    pub.mkdir(parents=True)
    priv.mkdir(parents=True)
    rows = [["id", "x1", "x2", "label"]]
    for i in range(60):
        rows.append([i, i % 7, float(i % 5), "pos" if i % 3 == 0 else "neg"])
    _write_csv(pub / "train.csv", ["id", "x1", "x2", "label"], rows[1:])
    _write_csv(pub / "test.csv", ["id", "x1", "x2"], rows[1:11])
    _write_csv(pub / "sample_submission.csv", ["id", "label"],
               [["%d" % i, "neg"] for i in range(10)])
    _write_csv(priv / "test.csv", ["id", "x1", "x2", "label"], rows[1:11])
    return pub


def test_analyzer_integration():
    tmp = Path(tempfile.mkdtemp(prefix="v240_analyzer_"))
    try:
        data = _make_layout(tmp)
        prof = Analyzer(str(tmp), "binary classification").profile("demo")
        dd = prof.deep_diagnostics or {}
        check("analyzer: deep_diagnostics populated",
              bool(dd.get("target_diag")) and bool(dd.get("feature_diag")),
              dd)
        check("analyzer: data_notes carries deep summary",
              "deep diagnostics:" in prof.data_notes, prof.data_notes[:400])
        # round-trip keeps new fields
        d = prof.to_dict()
        prof2 = AnalysisProfile.from_dict(d)
        check("round-trip keeps deep_diagnostics",
              prof2.deep_diagnostics == prof.deep_diagnostics, prof2)
        check("round-trip keeps difficulty_ladder",
              prof2.difficulty_ladder == {}, prof2)
        # deep notes helper fail-open on garbage
        check("deep notes fail-open", _deep_notes({"junk": 1}) == "", "")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_prompt_injection():
    from hera.portfolio import MethodPortfolio
    from hera.prioritization import Prioritizer
    from hera.planner import Planner, _deep_prompt_line
    from v2_contracts import ResearchPlan
    tmp = Path(tempfile.mkdtemp(prefix="v240_prompt_"))
    try:
        data = _make_layout(tmp)
        prof = Analyzer(str(tmp), "binary classification").profile("demo")
        port = MethodPortfolio.load_or_default(prof, tmp / "p.json")
        prio = Prioritizer(llm_call_fn=lambda prompt: "{}")
        prompt = prio.build_ticket_prompt(
            prof, port, ResearchPlan(hypothesis="H"), 3,
            research_intent="cheap_probe", stage="S1_baseline")
        check("prioritizer prompt: deep block present",
              "Measured deep diagnostics" in prompt, prompt[:1500])
        check("prioritizer prompt: ladder not-yet-measured",
              "not yet measured" in prompt, prompt[:1500])
        prof.difficulty_ladder = {"constant": 0.75, "linear": 0.70,
                                  "gbdt": 0.62, "headroom": 0.13}
        prompt2 = prio.build_ticket_prompt(
            prof, port, ResearchPlan(hypothesis="H"), 3,
            research_intent="cheap_probe", stage="S1_baseline")
        check("prioritizer prompt: ladder values injected",
              "constant=0.75 linear=0.7 gbdt=0.62" in prompt2
              or ("0.75" in prompt2 and "0.62" in prompt2), prompt2[:2000])
        check("prioritizer prompt: headroom injected",
              "0.13" in prompt2, prompt2[:2000])
        line = _deep_prompt_line(prof)
        check("planner deep line: ladder present", "ladder_c=0.75" in line,
              line)
        prof2 = AnalysisProfile.from_dict(prof.to_dict())
        line2 = _deep_prompt_line(prof2)
        check("planner deep line survives round-trip", "ladder_c" in line2,
              line2)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    test_target_diagnostics()
    test_order_and_time()
    test_fail_open_and_edges()
    test_analyzer_integration()
    test_prompt_injection()
    if FAILURES:
        print("RESULT=FAIL:%d" % len(FAILURES))
        raise SystemExit(1)
    print("RESULT=PASS")


if __name__ == "__main__":
    main()
