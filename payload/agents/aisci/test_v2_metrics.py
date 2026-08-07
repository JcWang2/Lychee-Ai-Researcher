# -*- coding: utf-8 -*-
"""test_v2_metrics.py - metric registry + TrustedEvaluator family tests.

Covers the full MLE-Bench metric registry (82 competitions) and the
dependency-free metric implementations used for direction-aware trusted
recomputation. All values are hand-computed or textbook definitions.
"""
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metrics_registry import (  # noqa: E402
    MLEBENCH_METRICS, SUPPORTED_METRICS, OOF_GUIDE, get_metric_spec,
    infer_metric_spec)
from pact.evaluator import (  # noqa: E402
    _accuracy, _f1_macro, _f1_micro, _f1_binary, _f0_5, _mcc, _auc, _qwk,
    _rmse, _mae, _log_mae, _rmsle, _spearman, _pearson, _kendall_tau,
    _mean_angular_error, _haversine, _levenshtein, _jaccard, _dice,
    _iou_mean, _binary_logloss, _logloss, _weighted_logloss, _kl_div,
    _mean_auc_multilabel, _map_at_k, _label_ranking_ap,
    _label_ranking_ap_multilabel, _dice_global)
from pact import TrustedEvaluator, HostSupervisorService, PromotionManager
from v2_contracts import CandidateBundle

FAILURES = []


def check(name, cond, detail=""):
    if not cond:
        FAILURES.append(name + ((": " + str(detail)) if detail else ""))
    print("[%s] %s" % ("OK" if cond else "FAIL", name))


def close(a, b, tol=1e-6):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) < tol


def test_registry_full_coverage():
    from metrics_registry import MLEBENCH_METRICS
    ids = sorted(MLEBENCH_METRICS)
    check("registry has 82 competitions", len(ids) == 82, len(ids))
    known = {
        "aerial-cactus-identification", "aptos2019-blindness-detection",
        "dog-breed-identification", "spaceship-titanic",
        "ventilator-pressure-prediction", "h-and-m-personalized-fashion-recommendations",
    }
    check("known ids present", known <= set(ids))
    bad_dir = [i for i in ids if MLEBENCH_METRICS[i][1] not in
               ("higher_is_better", "lower_is_better")]
    bad_met = [i for i in ids if MLEBENCH_METRICS[i][0] not in SUPPORTED_METRICS]
    bad_align = [i for i in ids if MLEBENCH_METRICS[i][2] not in
                 ("exact", "proxy")]
    check("directions valid", not bad_dir, bad_dir)
    check("metrics supported", not bad_met, bad_met)
    check("alignments valid", not bad_align, bad_align)
    for i in ids:
        m = MLEBENCH_METRICS[i][0]
        if m not in OOF_GUIDE:
            FAILURES.append("missing OOF_GUIDE for %s (%s)" % (m, i))
    check("every registry metric has OOF_GUIDE",
          all(MLEBENCH_METRICS[i][0] in OOF_GUIDE for i in ids))
    spec = get_metric_spec("aerial-cactus-identification")
    check("cactus -> auc", spec["metric_name"] == "auc")
    check("cactus direction higher", spec["metric_direction"] == "higher_is_better")
    check("cactus exact", spec["metric_alignment"] == "exact")
    spec = get_metric_spec("aptos2019-blindness-detection")
    check("aptos -> qwk", spec["metric_name"] == "qwk")
    spec = get_metric_spec("dog-breed-identification")
    check("dog-breed -> logloss lower",
          spec["metric_name"] == "logloss"
          and spec["metric_direction"] == "lower_is_better")
    spec = get_metric_spec("ventilator-pressure-prediction")
    check("ventilator -> mae (grade.py real impl)",
          spec["metric_name"] == "mae"
          and spec["metric_direction"] == "lower_is_better")
    inf = infer_metric_spec("regression")
    check("infer regression -> rmse lower",
          inf["metric_name"] == "rmse"
          and inf["metric_direction"] == "lower_is_better"
          and inf["metric_alignment"] == "inferred")
    inf = infer_metric_spec("classification")
    check("infer classification -> accuracy",
          inf["metric_name"] == "accuracy"
          and inf["metric_alignment"] == "inferred")


def test_label_metrics():
    check("accuracy", close(_accuracy(["1", "0", "1"], ["1", "0", "0"]), 2 / 3))
    check("accuracy strings", close(_accuracy(["True", "False"], ["True", "True"]), 0.5))
    t = ["a", "a", "b", "b"]
    p = ["a", "b", "a", "b"]
    check("f1_macro", close(_f1_macro(t, p), 0.5))
    check("f1_micro", close(_f1_micro(t, p), 0.5))
    t2 = [1, 1, 0, 0]
    p2 = [1, 0, 1, 0]
    check("f1_binary", close(_f1_binary(t2, p2), 0.5))
    check("f0_5", close(_f0_5(t2, p2), 0.5))
    check("mcc", close(_mcc(t2, p2), 0.0))
    check("mcc perfect", close(_mcc([1, 1, 0, 0], [1, 1, 0, 0]), 1.0))
    check("qwk perfect", close(_qwk([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]), 1.0))
    check("qwk majority", close(_qwk([0, 0, 0, 0, 0], [0, 0, 0, 0, 1]), 0.0))


def test_ranking_metrics():
    check("auc perfect", close(_auc([1, 1, 0, 0], [0.9, 0.8, 0.3, 0.2]), 1.0))
    check("auc reverse", close(_auc([1, 0], [0.1, 0.9]), 0.0))
    check("auc tie", close(_auc([1, 0], [0.5, 0.5]), 0.5))
    check("spearman perfect", close(_spearman([1, 2, 3, 4], [1, 2, 3, 4]), 1.0))
    check("spearman reverse", close(_spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0))
    check("pearson perfect", close(_pearson([1, 2, 3, 4], [2, 4, 6, 8]), 1.0))
    check("kendall perfect", close(_kendall_tau([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]), 1.0))
    check("kendall reverse", close(_kendall_tau([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]), -1.0))
    check("kendall ties", close(_kendall_tau([1, 1, 2, 2], [1, 1, 2, 2]), 1.0))
    check("angular", close(_mean_angular_error([0, 10], [350, 20]), 10.0))
    check("haversine same point", close(_haversine(["0,0"], ["0,0"]), 0.0))
    # one degree of longitude at the equator ~= 111.19 km
    check("haversine 1deg", close(_haversine(["0,0"], ["0,1"]), 111.1949, tol=0.5))


def test_regression_metrics():
    check("rmse", close(_rmse([1, 2, 3], [1, 2, 5]), math.sqrt(4 / 3)))
    check("mae", close(_mae([1, 2, 3], [1, 2, 5]), 2 / 3))
    check("log_mae", close(_log_mae([1, 2], [1, 2]), 0.0))
    check("rmsle", close(_rmsle([1, 2, 3], [1, 2, 3]), 0.0))
    check("rmse empty -> None", _rmse([], []) is None)


def test_string_metrics():
    check("levenshtein kitten/sitting",
          close(_levenshtein(["kitten"], ["sitting"]), 3.0))
    check("levenshtein mean",
          close(_levenshtein(["kitten", "saturday"], ["sitting", "sunday"]), 3.0))
    check("jaccard", close(_jaccard(["hello world foo"], ["hello world bar"]), 0.5))
    check("jaccard empty both", close(_jaccard([""], [""]), 1.0))


def test_flat_metrics():
    check("dice", close(_dice([1, 1, 0, 0], [1, 0, 1, 0]), 0.5))
    check("dice perfect", close(_dice([1, 1, 0, 0], [1, 1, 0, 0]), 1.0))
    # pixel threshold-scan proxy: with binary preds every thr<1 keeps
    # pred=1 pixels, so the scan degenerates to the fixed pixel IoU = 1/3
    check("iou_mean binary", close(_iou_mean([1, 1, 0, 0], [1, 0, 1, 0]), 1 / 3))


def test_probability_metrics():
    check("binary_logloss", close(
        _binary_logloss([1, 0], [0.9, 0.1]), -math.log(0.9)))
    check("binary_logloss perfect", close(_binary_logloss([1, 0], [1.0, 0.0]), 0.0))
    ll = _logloss(["a", "b"], {"a": ["0.9", "0.1"], "b": ["0.1", "0.9"]}, ["a", "b"])
    check("logloss", close(ll, -math.log(0.9)))
    wl = _weighted_logloss(["a", "b"], {"a": ["0.9", "0.1"], "b": ["0.1", "0.9"]},
                           ["a", "b"])
    check("weighted_logloss", close(wl, -math.log(0.9)))
    kl = _kl_div({"a": ["0.5", "0.5"], "b": ["0.5", "0.5"]},
                 {"a": ["0.5", "0.5"], "b": ["0.5", "0.5"]}, ["a", "b"])
    check("kl_div identical", close(kl, 0.0))
    ma = _mean_auc_multilabel(["a", "b", "a"], {"a": ["0.9", "0.2", "0.8"],
                                                "b": ["0.1", "0.8", "0.2"]},
                              ["a", "b"])
    check("mean_auc_multilabel", close(ma, 1.0))


def test_grouped_metrics():
    ap = _map_at_k(["q1", "q1", "q1"], ["1", "0", "1"], ["0.9", "0.5", "0.8"], 2)
    check("map_at_k", close(ap, 1.0))
    ap2 = _map_at_k(["q1", "q1"], ["1", "0"], ["0.5", "0.9"], 2)
    check("map_at_k miss first", close(ap2, 0.5))
    lrap = _label_ranking_ap(["q1", "q1", "q1", "q1"],
                             ["1", "0", "0", "1"],
                             ["0.9", "0.8", "0.7", "0.6"])
    # positives at ranks 1 and 4 (global): 1/4 + 2/4 over 2 positives
    check("label_ranking_ap", close(lrap, 0.75))


def test_official_semantics_extra():
    # AI4Code official formula: tau = 1 - 4*inv/(n*(n-1)), ties never
    # counted as inversions (grade.py bisect_right behavior).
    check("kendall official ties", close(_kendall_tau([1, 1, 2, 2], [1, 1, 2, 2]), 1.0))
    check("kendall official partial",
          close(_kendall_tau([1, 2, 3, 4], [1, 3, 2, 4]), 1 - 4 * 1.0 / (4 * 3)))
    # freesound lwlrap: weighted multilabel LRAP
    rows = [
        {"true_a": "1", "true_b": "0", "true_c": "0", "true_d": "1",
         "pred_a": "0.9", "pred_b": "0.8", "pred_c": "0.7", "pred_d": "0.6"},
        {"true_a": "1", "true_b": "0", "true_c": "0", "true_d": "0",
         "pred_a": "0.9", "pred_b": "0.4", "pred_c": "0.3", "pred_d": "0.2"},
    ]
    tcols = ["true_a", "true_b", "true_c", "true_d"]
    pcols = ["pred_a", "pred_b", "pred_c", "pred_d"]
    # sample1 lrap = (1/1 + 2/4)/2 = 0.75 (weight 2); sample2 lrap = 1/1 = 1.0 (weight 1)
    check("lrap multilabel weighted",
          close(_label_ranking_ap_multilabel(rows, tcols, pcols), (0.75 * 2 + 1.0) / 3))
    # contrails global dice over flattened rows
    check("dice global",
          close(_dice_global([1, 1, 0, 0, 1], [1, 0, 1, 0, 1]),
                2 * 2.0 / (2 * 2 + 1 + 1)))
    check("dice global both empty", close(_dice_global([0, 0], [0, 0]), 1.0))


def test_evaluator_integration():
    tmp = Path(tempfile.mkdtemp(prefix="v2_metrics_"))
    try:
        work = tmp / "work"
        work.mkdir()
        oof = work / "oof.csv"
        # multi-class log loss OOF with pred_* columns
        oof.write_text(
            "true,pred_a,pred_b,pred_c\n"
            "a,0.8,0.1,0.1\n"
            "b,0.1,0.8,0.1\n"
            "c,0.1,0.1,0.8\n", encoding="utf-8")
        bundle = CandidateBundle(trial_id="t1", oof_path=str(oof),
                                 submission_path="", bundle_hash="h")
        ev = TrustedEvaluator(metric_name="logloss",
                              metric_direction="lower_is_better",
                              metric_label="multi-class log loss")
        r = ev.evaluate(bundle)
        check("integration logloss value", close(r.metric, -math.log(0.8)))
        check("integration evaluator name",
              r.evaluator == "trusted_recompute_logloss")
        check("integration direction carried",
              r.metric_direction == "lower_is_better")
        check("integration label carried",
              r.metric_label == "multi-class log loss")
        # missing probability columns -> None
        oof2 = work / "oof2.csv"
        oof2.write_text("true,pred\na,0.5\n", encoding="utf-8")
        b2 = CandidateBundle(trial_id="t2", oof_path=str(oof2),
                             submission_path="", bundle_hash="h2")
        r2 = ev.evaluate(b2)
        check("integration no prob cols -> None", r2.metric is None)
        # auc OOF
        oof3 = work / "oof3.csv"
        oof3.write_text("true,pred\n1,0.9\n1,0.8\n0,0.3\n0,0.2\n",
                        encoding="utf-8")
        ev3 = TrustedEvaluator(metric_name="auc")
        r3 = ev3.evaluate(CandidateBundle(trial_id="t3", oof_path=str(oof3),
                                          submission_path="", bundle_hash="h3"))
        check("integration auc value", close(r3.metric, 1.0))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_direction_aware_host():
    from pact.file_bus import FileBus
    tmp = Path(tempfile.mkdtemp(prefix="v2_dir_"))
    try:
        bus = FileBus(tmp / "state")
        dummy_exec = type("DummyExec", (), {"work_dir": str(tmp / "work")})()
        low = HostSupervisorService(
            bus=bus,
            executor=dummy_exec,
            bundler=None,
            evaluator=TrustedEvaluator(metric_name="rmse",
                                       metric_direction="lower_is_better"),
            promotion=PromotionManager(bus, metric_direction="lower_is_better"),
            implementer=None,
            competition="demo")
        check("host lower better smaller", low._better(1.0, 2.0))
        check("host lower better larger is worse", not low._better(2.0, 1.0))
        check("host lower verdict success",
              low._verdict(0.5, 1.0, 0) == "success")
        check("host lower verdict regression",
              low._verdict(2.0, 1.0, 0) == "regression")
        check("host lower verdict stagnant",
              low._verdict(1.005, 1.0, 0) == "stagnant")
        high = HostSupervisorService(
            bus=bus, executor=dummy_exec, bundler=None,
            evaluator=TrustedEvaluator(metric_name="accuracy"),
            promotion=PromotionManager(bus),
            implementer=None,
            competition="demo")
        check("host higher verdict success",
              high._verdict(0.9, 0.8, 0) == "success")
        check("host higher verdict regression",
              high._verdict(0.7, 0.8, 0) == "regression")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print("=== V2 METRICS tests ===\n")
    test_registry_full_coverage()
    test_label_metrics()
    test_ranking_metrics()
    test_regression_metrics()
    test_string_metrics()
    test_flat_metrics()
    test_probability_metrics()
    test_grouped_metrics()
    test_evaluator_integration()
    test_official_semantics_extra()
    test_direction_aware_host()
    print("\nRESULT=" + ("PASS" if not FAILURES else "FAIL:" + ",".join(FAILURES)))
    sys.exit(0 if not FAILURES else 1)
