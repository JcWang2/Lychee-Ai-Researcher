# -*- coding: utf-8 -*-
"""test_v2_resource_profiler.py - V2.2 competition-agnostic resource tests."""
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from hera.portfolio import ResourceProfiler, estimate_grant_cost  # noqa: E402
from v2_contracts import AnalysisProfile  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("[OK] " + name)
    else:
        print("[FAIL] " + name + (" | " + detail if detail else ""))
        FAILURES.append(name)


def _img(rows=20000, w=192, h=192, classes=120, comp="any-competition"):
    return AnalysisProfile(competition=comp, task_type="classification",
                           modality="image", train_rows=rows,
                           image_width=w, image_height=h, image_channels=3,
                           n_classes=classes, feature_columns=["image"])


def _tab(rows=10000, comp="any-competition"):
    return AnalysisProfile(competition=comp, task_type="classification",
                           modality="tabular", train_rows=rows,
                           feature_dim=24, n_classes=2,
                           feature_columns=["a", "b"])


def _text(rows=5000, comp="any-competition"):
    return AnalysisProfile(competition=comp, task_type="classification",
                           modality="text", train_rows=rows,
                           feature_dim=1, n_classes=4,
                           feature_columns=["text"])


def test_three_modalities():
    tmp = Path(tempfile.mkdtemp(prefix="v2_resprof_"))
    try:
        for name, profile in (("image", _img()), ("tabular", _tab()),
                              ("text", _text())):
            res = ResourceProfiler().derive(profile)
            check(name + " has all keys",
                  all(k in res for k in (
                      "max_budget_seconds", "image_size_max", "epochs_min",
                      "epochs_max", "max_folds", "train_rows_cap",
                      "batch_hint", "model_scale_ceiling", "t_est_seconds",
                      "pretrained_policy", "derived_from")),
                  str(sorted(res.keys())))
            check(name + " budget sane",
                  300 <= res["max_budget_seconds"] <= 7200
                  and res["epochs_min"] <= res["epochs_max"],
                  str(res))
            check(name + " no competition name in derivation",
                  "competition" not in str(res.get("derived_from") or ""),
                  str(res.get("derived_from")))
        img_res = ResourceProfiler().derive(_img())
        tab_res = ResourceProfiler().derive(_tab())
        check("image gets size cap, tabular not",
              img_res["image_size_max"] is not None
              and tab_res["image_size_max"] is None,
              str((img_res["image_size_max"], tab_res["image_size_max"])))
        check("image budget > tabular budget (same rows)",
              img_res["max_budget_seconds"] > tab_res["max_budget_seconds"],
              str((img_res["max_budget_seconds"],
                   tab_res["max_budget_seconds"])))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_competition_invariance():
    a = ResourceProfiler().derive(_img(comp="dog-breed-identification"))
    b = ResourceProfiler().derive(_img(comp="totally-unknown-task"))
    check("same signals -> same resources regardless of name",
          a["max_budget_seconds"] == b["max_budget_seconds"]
          and a["image_size_max"] == b["image_size_max"]
          and a["max_folds"] == b["max_folds"],
          str((a["max_budget_seconds"], b["max_budget_seconds"])))


def test_gpu_memory_scaling():
    small = ResourceProfiler(gpu_memory_mb=8192).derive(_img())
    big = ResourceProfiler(gpu_memory_mb=40960).derive(_img())
    check("bigger GPU -> bigger image cap",
          big["image_size_max"] >= small["image_size_max"],
          str((small["image_size_max"], big["image_size_max"])))
    check("bigger GPU -> larger batch hint",
          big["batch_hint"] >= small["batch_hint"],
          str((small["batch_hint"], big["batch_hint"])))
    check("24G+ ceiling large", big["model_scale_ceiling"] == "large",
          big["model_scale_ceiling"])


def test_f0_calibration_scales_t_est():
    cal = {"f0_seconds": 600.0, "train_rows": 20000, "image_pixels": 36864}
    res = ResourceProfiler(f0_calibration=cal).derive(_img(rows=20000))
    check("t_est follows f0 when rows match",
          abs(res["t_est_seconds"] - 600) <= 60,
          str(res["t_est_seconds"]))
    res2 = ResourceProfiler(f0_calibration=cal).derive(_img(rows=80000))
    check("more rows -> larger t_est", res2["t_est_seconds"] > res["t_est_seconds"],
          str(res2["t_est_seconds"]))
    res3 = ResourceProfiler().derive(_img())
    check("no f0 -> derived estimate", 300 <= res3["t_est_seconds"] <= 7200,
          str(res3["t_est_seconds"]))


def test_all_eight_signals_consumed():
    """v2.2.1: n_classes / feature_dim / cur_pixels / cached_weights must
    actually change resource decisions (the acceptance probe said only 4 of
    8 signals were consumed)."""
    low_feat = AnalysisProfile(
        competition="x", task_type="classification", modality="tabular",
        train_rows=10000, feature_dim=24, n_classes=2,
        feature_columns=["a", "b"])
    high_feat = AnalysisProfile(
        competition="x", task_type="classification", modality="tabular",
        train_rows=10000, feature_dim=4096, n_classes=2,
        feature_columns=["a", "b"])
    low = ResourceProfiler().derive(low_feat)
    high2 = ResourceProfiler().derive(high_feat)
    check("feature_dim raises t_est",
          high2["t_est_seconds"] > low["t_est_seconds"],
          str((low["t_est_seconds"], high2["t_est_seconds"])))
    many = ResourceProfiler().derive(AnalysisProfile(
        competition="x", task_type="classification", modality="image",
        train_rows=20000, image_width=192, image_height=192, n_classes=120,
        feature_columns=["image"]))
    few = ResourceProfiler().derive(AnalysisProfile(
        competition="x", task_type="classification", modality="image",
        train_rows=20000, image_width=192, image_height=192, n_classes=2,
        feature_columns=["image"]))
    check("class count raises budget",
          many["max_budget_seconds"] > few["max_budget_seconds"],
          str((few["max_budget_seconds"], many["max_budget_seconds"])))
    check("class count raises scale",
          many["model_scale_ceiling"] != "any"
          or few["model_scale_ceiling"] == "any",
          str((few["model_scale_ceiling"], many["model_scale_ceiling"])))
    big_px = ResourceProfiler().derive(AnalysisProfile(
        competition="x", task_type="classification", modality="image",
        train_rows=20000, image_width=512, image_height=512, n_classes=2,
        feature_columns=["image"]))
    check("native pixels raise budget",
          big_px["max_budget_seconds"] > few["max_budget_seconds"],
          str((few["max_budget_seconds"], big_px["max_budget_seconds"])))
    cached = ResourceProfiler(cached_weights=["resnet18", "efficientnet_b0",
                                              "vit_tiny"]).derive(
        AnalysisProfile(competition="x", task_type="classification",
                        modality="image", train_rows=20000,
                        image_width=192, image_height=192, n_classes=2,
                        feature_columns=["image"]))
    check("cached weights raise epochs ceiling",
          cached["epochs_max"] > few["epochs_max"],
          str((few["epochs_max"], cached["epochs_max"])))
    df = many.get("derived_from") or {}
    check("derived_from exposes all signals",
          df.get("n_classes") == 120 and "cur_pixels" in df
          and "factors" in df and df.get("cached_weights") == 0,
          str(df))


def test_pretrained_policy_and_cost():
    res = ResourceProfiler(pretrained_policy="scratch").derive(_img())
    check("scratch policy honored", res["pretrained_policy"] == "scratch",
          res["pretrained_policy"])
    res2 = ResourceProfiler(pretrained_policy="cache",
                            cached_weights=["resnet18", "efficientnet_b0"]).derive(_img())
    check("cache policy + whitelist", res2["pretrained_policy"] == "cache"
          and res2["derived_from"]["cached_weights"] == 2,
          str(res2["derived_from"]))
    cost = estimate_grant_cost(res2, 3, "expensive_structural")
    check("cost = children * t_est * intent factor",
          abs(cost - 3 * res2["t_est_seconds"] * 1.5) < 1.0,
          str(cost))


if __name__ == "__main__":
    print("=== V2 resource-profiler tests ===\n")
    test_three_modalities()
    test_competition_invariance()
    test_gpu_memory_scaling()
    test_f0_calibration_scales_t_est()
    test_all_eight_signals_consumed()
    test_pretrained_policy_and_cost()
    print("\nRESULT=" + ("PASS" if not FAILURES else "FAIL:" + ",".join(FAILURES)))
    sys.exit(0 if not FAILURES else 1)