# -*- coding: utf-8 -*-
"""metrics_registry.py - MLE-Bench official metric registry (82 competitions).

Audit trail (public information only; see docs/V2_ARCHITECTURE_ALIGNMENT.md):
  - grader names: openai/mle-bench mlebench/competitions/<id>/config.yaml
    (the same public rules printed on every Kaggle competition page)
  - direction: official leaderboard.csv first-vs-last score
    (Grader.is_lower_better; public leaderboard)
  - ventilator-pressure-prediction: config grader name is a copy-paste error
    (dice-hausdorff-combo) but the shipped grade.py uses mean absolute error;
    we map to the real implementation (mae).
  No private labels, gold answers, checksums or data-preparation logic is used.

Every competition maps to (metric, direction, alignment, label, params):
  - metric: canonical family implemented by TrustedEvaluator
  - direction: "higher_is_better" | "lower_is_better" - comparison direction
    of the metric family itself (a proxy metric carries its own direction;
    the official direction is preserved in the human label)
  - alignment: "exact" | "proxy" | "inferred"
  - label: human-readable official metric (for prompts / audit)
  - params: metric parameters (e.g. k for map_at_k)
"""

# metric family -> human label
METRIC_LABELS = {
    "accuracy": "accuracy",
    "f1_macro": "macro F1",
    "f1_micro": "micro F1",
    "f1_binary": "binary F1",
    "f0_5": "F0.5",
    "mcc": "Matthews correlation coefficient",
    "auc": "ROC AUC (binary)",
    "mean_auc_multilabel": "mean column-wise ROC AUC",
    "qwk": "quadratic weighted kappa",
    "logloss": "multi-class log loss",
    "binary_logloss": "binary log loss",
    "weighted_logloss": "weighted multi-label log loss",
    "kl_div": "KL divergence",
    "rmse": "root mean squared error",
    "mae": "mean absolute error",
    "log_mae": "log mean absolute error",
    "rmsle": "root mean squared log error",
    "spearman": "Spearman correlation",
    "pearson": "Pearson correlation",
    "kendall_tau": "Kendall tau",
    "mean_angular_error": "mean angular error",
    "haversine": "average haversine distance",
    "levenshtein": "Levenshtein distance",
    "jaccard": "word-level Jaccard",
    "dice": "Dice coefficient",
    "iou_mean": "mean IoU @ thresholds",
    "map_at_k": "mean average precision @k",
    "label_ranking_ap": "label ranking average precision",
}

# metric family -> oof.csv pred-column contract (for the implementer prompt)
OOF_GUIDE = {
    "accuracy": "pred = predicted class label, same space as true",
    "f1_macro": "pred = predicted class label (macro F1 over classes)",
    "f1_micro": "pred = predicted class label",
    "f1_binary": "pred = predicted 0/1 (or True/False)",
    "f0_5": "pred = predicted 0/1 per pixel/row",
    "mcc": "pred = predicted 0/1 (True/False)",
    "auc": "pred = P(class=1) probability (0..1) or a confidence score",
    "mean_auc_multilabel": "one pred_<class> probability column per class",
    "qwk": "pred = predicted ordinal value (e.g. 0-4)",
    "logloss": "one pred_<class> probability column per class (softmax)",
    "binary_logloss": "pred = P(class=1) probability (0..1)",
    "weighted_logloss": "one pred_<class> probability column per class",
    "kl_div": "true_<class> = target distribution, pred_<class> = predicted probability",
    "rmse": "pred = numeric prediction",
    "mae": "pred = numeric prediction",
    "log_mae": "pred = numeric prediction (log-space error)",
    "rmsle": "pred = numeric prediction",
    "spearman": "pred = numeric score",
    "pearson": "pred = numeric prediction",
    "kendall_tau": "pred = numeric score",
    "mean_angular_error": "pred = numeric angle",
    "haversine": "pred = latitude,longitude (comma separated)",
    "levenshtein": "pred = output string",
    "jaccard": "pred = output string (word-level Jaccard)",
    "dice": "pred = 0/1 per pixel/row (flattened); global aggregation when metric_params.aggregation=global (contrails)",
    "iou_mean": "pred = 0/1 per pixel/row (flattened; threshold-scan proxy of component-level official)",
    "map_at_k": "query column groups rows; true = 0/1 positive candidate; pred = score",
    "label_ranking_ap": "single-label: query groups rows, true = 0/1, pred = score; multi-label (freesound): true_<class> 0/1 + pred_<class> probability columns per row",
}

# metric family -> minimum meaningful absolute improvement.
#
# v2.3.6 root-cause fix: a single global min_delta=0.01 blocked real
# improvements forever for bounded / score metrics (AUC 0.9997 vs 0.9972 is
# +0.0025 < 0.01, so it could never be promoted). The threshold is a
# property of the metric family, not of the run: bounded score metrics in
# [0,1] use 1e-4; probability-space losses (logloss) use 1e-4; unbounded
# error metrics use 1e-3. Unknown families keep the legacy 0.01 default.
DEFAULT_MIN_DELTA = 0.01

METRIC_MIN_DELTA = {
    # bounded / score metrics (higher-is-better, roughly [0,1])
    "accuracy": 1e-4,
    "auc": 1e-4,
    "mean_auc_multilabel": 1e-4,
    "f1_macro": 1e-4,
    "f1_micro": 1e-4,
    "f1_binary": 1e-4,
    "f0_5": 1e-4,
    "mcc": 1e-4,
    "qwk": 1e-4,
    "spearman": 1e-4,
    "pearson": 1e-4,
    "kendall_tau": 1e-4,
    "dice": 1e-4,
    "iou_mean": 1e-4,
    "map_at_k": 1e-4,
    "label_ranking_ap": 1e-4,
    "jaccard": 1e-4,
    "mean_angular_error": 1e-4,
    # probability-space losses (logloss ~0.3..2.0, KL in nats)
    "logloss": 1e-4,
    "binary_logloss": 1e-4,
    "weighted_logloss": 1e-4,
    "kl_div": 1e-4,
    # unbounded error metrics (typical magnitudes 0.1..10+)
    "rmse": 1e-3,
    "mae": 1e-3,
    "log_mae": 1e-3,
    "rmsle": 1e-3,
    "haversine": 1e-3,
    "levenshtein": 1e-3,
}


def metric_min_delta(metric_name: str) -> float:
    """Per-metric minimum meaningful improvement (absolute units)."""
    return float(METRIC_MIN_DELTA.get(str(metric_name or "").strip().lower(),
                                       DEFAULT_MIN_DELTA))


# competition -> (metric, direction, alignment, label, params)
MLEBENCH_METRICS = {
    "3d-object-detection-for-autonomous-vehicles": ("map_at_k", "higher_is_better", "proxy", "mean average precision (object detection; ranking proxy)", {"k": 5}),
    "aerial-cactus-identification": ("auc", "higher_is_better", "exact", "ROC AUC", {}),
    "AI4Code": ("kendall_tau", "higher_is_better", "proxy", "Kendall tau (token-ordering proxy)", {}),
    "alaska2-image-steganalysis": ("mean_auc_multilabel", "higher_is_better", "proxy", "weighted AUROC (unweighted proxy)", {}),
    "aptos2019-blindness-detection": ("qwk", "higher_is_better", "exact", "quadratic weighted kappa", {}),
    "billion-word-imputation": ("levenshtein", "lower_is_better", "exact", "Levenshtein distance", {}),
    "bms-molecular-translation": ("levenshtein", "lower_is_better", "exact", "Levenshtein distance", {}),
    "cassava-leaf-disease-classification": ("accuracy", "higher_is_better", "exact", "accuracy", {}),
    "cdiscount-image-classification-challenge": ("accuracy", "higher_is_better", "exact", "accuracy", {}),
    "chaii-hindi-and-tamil-question-answering": ("jaccard", "higher_is_better", "exact", "word-level Jaccard", {}),
    "champs-scalar-coupling": ("log_mae", "lower_is_better", "exact", "log mean absolute error", {}),
    "denoising-dirty-documents": ("rmse", "lower_is_better", "exact", "root mean squared error (pixel-level; flatten pixels)", {}),
    "detecting-insults-in-social-commentary": ("auc", "higher_is_better", "exact", "ROC AUC", {}),
    "dog-breed-identification": ("logloss", "lower_is_better", "exact", "multi-class log loss", {}),
    "dogs-vs-cats-redux-kernels-edition": ("binary_logloss", "lower_is_better", "exact", "binary log loss", {}),
    "facebook-recruiting-iii-keyword-extraction": ("f1_micro", "higher_is_better", "exact", "micro F1", {}),
    "freesound-audio-tagging-2019": ("label_ranking_ap", "higher_is_better", "exact", "label ranking average precision", {}),
    "google-quest-challenge": ("spearman", "higher_is_better", "proxy", "column-wise Spearman (mean over columns)", {}),
    "google-research-identify-contrails-reduce-global-warming": ("dice", "higher_is_better", "exact", "global Dice (flatten pixels)", {"aggregation": "global"}),
    "h-and-m-personalized-fashion-recommendations": ("map_at_k", "higher_is_better", "proxy", "MAP@12 (recommendation proxy)", {"k": 12}),
    "herbarium-2020-fgvc7": ("f1_macro", "higher_is_better", "exact", "macro F1", {}),
    "herbarium-2021-fgvc8": ("f1_macro", "higher_is_better", "exact", "macro F1", {}),
    "herbarium-2022-fgvc9": ("f1_macro", "higher_is_better", "exact", "macro F1", {}),
    "histopathologic-cancer-detection": ("auc", "higher_is_better", "exact", "ROC AUC", {}),
    "hms-harmful-brain-activity-classification": ("kl_div", "lower_is_better", "proxy", "KL divergence (voting-distribution proxy)", {}),
    "hotel-id-2021-fgvc8": ("map_at_k", "higher_is_better", "proxy", "map@5 (retrieval proxy)", {"k": 5}),
    "hubmap-kidney-segmentation": ("dice", "higher_is_better", "exact", "Dice coefficient (flatten pixels)", {}),
    "icecube-neutrinos-in-deep-ice": ("mean_angular_error", "lower_is_better", "proxy", "mean angular error (scalar-angle proxy)", {}),
    "imet-2020-fgvc7": ("f1_micro", "higher_is_better", "exact", "micro F1", {}),
    "inaturalist-2019-fgvc6": ("accuracy", "higher_is_better", "exact", "official top-1 classification error (lower); OOF reports accuracy = 1 - error (higher)", {}),
    "invasive-species-monitoring": ("auc", "higher_is_better", "exact", "ROC AUC", {}),
    "iwildcam-2019-fgvc6": ("f1_macro", "higher_is_better", "exact", "macro F1", {}),
    "iwildcam-2020-fgvc7": ("accuracy", "higher_is_better", "exact", "accuracy", {}),
    "jigsaw-toxic-comment-classification-challenge": ("mean_auc_multilabel", "higher_is_better", "exact", "column-wise ROC AUC", {}),
    "jigsaw-unintended-bias-in-toxicity-classification": ("auc", "higher_is_better", "proxy", "unintended-bias score (AUC-based proxy)", {}),
    "kuzushiji-recognition": ("f1_macro", "higher_is_better", "exact", "macro F1", {}),
    "leaf-classification": ("logloss", "lower_is_better", "exact", "multi-class log loss", {}),
    "learning-agency-lab-automated-essay-scoring-2": ("qwk", "higher_is_better", "exact", "quadratic weighted kappa", {}),
    "lmsys-chatbot-arena": ("logloss", "lower_is_better", "exact", "multi-class log loss", {}),
    "ml2021spring-hw2": ("accuracy", "higher_is_better", "exact", "accuracy", {}),
    "mlsp-2013-birds": ("mean_auc_multilabel", "higher_is_better", "exact", "mean AUC (multilabel)", {}),
    "movie-review-sentiment-analysis-kernels-only": ("accuracy", "higher_is_better", "exact", "accuracy", {}),
    "multi-modal-gesture-recognition": ("levenshtein", "lower_is_better", "exact", "Levenshtein distance", {}),
    "new-york-city-taxi-fare-prediction": ("rmse", "lower_is_better", "exact", "root mean squared error", {}),
    "nfl-player-contact-detection": ("mcc", "higher_is_better", "exact", "Matthews correlation coefficient", {}),
    "nomad2018-predict-transparent-conductors": ("rmsle", "lower_is_better", "proxy", "mean column-wise RMSLE", {}),
    "osic-pulmonary-fibrosis-progression": ("rmse", "lower_is_better", "proxy", "modified Laplace log likelihood (official higher; RMSE proxy)", {}),
    "paddy-disease-classification": ("accuracy", "higher_is_better", "exact", "accuracy", {}),
    "petfinder-pawpularity-score": ("rmse", "lower_is_better", "exact", "root mean squared error", {}),
    "plant-pathology-2020-fgvc7": ("mean_auc_multilabel", "higher_is_better", "exact", "mean column-wise ROC AUC", {}),
    "plant-pathology-2021-fgvc8": ("f1_micro", "higher_is_better", "exact", "micro F1", {}),
    "plant-seedlings-classification": ("f1_micro", "higher_is_better", "exact", "micro F1", {}),
    "playground-series-s3e18": ("mean_auc_multilabel", "higher_is_better", "exact", "multilabel AUROC", {}),
    "predict-volcanic-eruptions-ingv-oe": ("mae", "lower_is_better", "proxy", "mean absolute error (multi-output proxy)", {}),
    "random-acts-of-pizza": ("auc", "higher_is_better", "exact", "ROC AUC", {}),
    "ranzcr-clip-catheter-line-classification": ("mean_auc_multilabel", "higher_is_better", "exact", "ROC AUC (11 labels)", {}),
    "rsna-2022-cervical-spine-fracture-detection": ("weighted_logloss", "lower_is_better", "proxy", "weighted multi-label log loss", {}),
    "rsna-breast-cancer-detection": ("f1_binary", "higher_is_better", "proxy", "probabilistic F1 (0.5-threshold F1 proxy)", {}),
    "rsna-miccai-brain-tumor-radiogenomic-classification": ("auc", "higher_is_better", "exact", "ROC AUC", {}),
    "seti-breakthrough-listen": ("auc", "higher_is_better", "exact", "ROC AUC", {}),
    "siim-covid19-detection": ("map_at_k", "higher_is_better", "proxy", "mean average precision (detection; ranking proxy)", {"k": 5}),
    "siim-isic-melanoma-classification": ("auc", "higher_is_better", "exact", "ROC AUC", {}),
    "smartphone-decimeter-2022": ("haversine", "lower_is_better", "proxy", "average haversine distance", {}),
    "spaceship-titanic": ("accuracy", "higher_is_better", "exact", "accuracy", {}),
    "spooky-author-identification": ("logloss", "lower_is_better", "exact", "multi-class log loss", {}),
    "stanford-covid-vaccine": ("logloss", "lower_is_better", "exact", "multi-class log loss", {}),
    "statoil-iceberg-classifier-challenge": ("binary_logloss", "lower_is_better", "exact", "binary log loss", {}),
    "tabular-playground-series-dec-2021": ("accuracy", "higher_is_better", "exact", "accuracy", {}),
    "tabular-playground-series-may-2022": ("auc", "higher_is_better", "exact", "ROC AUC", {}),
    "tensorflow2-question-answering": ("f1_micro", "higher_is_better", "exact", "micro F1", {}),
    "tensorflow-speech-recognition-challenge": ("accuracy", "higher_is_better", "exact", "accuracy", {}),
    "text-normalization-challenge-english-language": ("accuracy", "higher_is_better", "proxy", "token-level accuracy proxy", {}),
    "text-normalization-challenge-russian-language": ("accuracy", "higher_is_better", "proxy", "token-level accuracy proxy", {}),
    "tgs-salt-identification-challenge": ("iou_mean", "higher_is_better", "proxy", "mean IoU @ 0.5..0.95 (flatten-pixel proxy of component-level official)", {}),
    "the-icml-2013-whale-challenge-right-whale-redux": ("auc", "higher_is_better", "exact", "ROC AUC", {}),
    "tweet-sentiment-extraction": ("jaccard", "higher_is_better", "exact", "token Jaccard", {}),
    "us-patent-phrase-to-phrase-matching": ("pearson", "higher_is_better", "exact", "Pearson correlation coefficient", {}),
    "uw-madison-gi-tract-image-segmentation": ("dice", "higher_is_better", "proxy", "dice+hausdorff combo (Dice proxy)", {}),
    "ventilator-pressure-prediction": ("mae", "lower_is_better", "exact", "mean absolute error (grade.py implementation)", {}),
    "vesuvius-challenge-ink-detection": ("f0_5", "higher_is_better", "exact", "F0.5 (pixel-level)", {}),
    "vinbigdata-chest-xray-abnormalities-detection": ("map_at_k", "higher_is_better", "proxy", "mAP@IoU>0.4 (detection; ranking proxy)", {"k": 5}),
    "whale-categorization-playground": ("map_at_k", "higher_is_better", "proxy", "MAP@5 (retrieval proxy)", {"k": 5}),
}


def get_metric_spec(competition: str) -> dict:
    """Official metric spec for a known MLE-Bench competition (or inferred)."""
    row = MLEBENCH_METRICS.get(competition or "")
    if row:
        metric, direction, alignment, label, params = row
        return {
            "metric_name": metric,
            "metric_direction": direction,
            "metric_alignment": alignment,
            "metric_label": label,
            "metric_params": dict(params or {}),
            "min_delta": metric_min_delta(metric),
        }
    return infer_metric_spec("")


def infer_metric_spec(task_type: str) -> dict:
    """Fallback for unknown competitions: task-type heuristic, auditable."""
    tt = (task_type or "").lower()
    if tt in ("regression", "timeseries"):
        return {
            "metric_name": "rmse", "metric_direction": "lower_is_better",
            "metric_alignment": "inferred", "metric_label": "root mean squared error (inferred)",
            "metric_params": {}, "min_delta": metric_min_delta("rmse"),
        }
    if tt == "segmentation":
        # v2.3.8: RLE-mask targets are never accuracy; dice is the generic
        # pixel-overlap family (iou_mean tasks keep their official mapping).
        return {
            "metric_name": "dice", "metric_direction": "higher_is_better",
            "metric_alignment": "inferred", "metric_label": "Dice coefficient (inferred)",
            "metric_params": {}, "min_delta": metric_min_delta("dice"),
        }
    if tt == "detection":
        # v2.3.8: box targets are ranked candidates; map@k is the generic
        # ranking family for unknown detection competitions.
        return {
            "metric_name": "map_at_k", "metric_direction": "higher_is_better",
            "metric_alignment": "inferred", "metric_label": "mean average precision @k (inferred)",
            "metric_params": {"k": 5}, "min_delta": metric_min_delta("map_at_k"),
        }
    return {
        "metric_name": "accuracy", "metric_direction": "higher_is_better",
        "metric_alignment": "inferred", "metric_label": "accuracy (inferred)",
        "metric_params": {}, "min_delta": metric_min_delta("accuracy"),
    }


def apply_metric_to_profile(profile) -> None:
    """Fill metric_* fields on an AnalysisProfile in place."""
    spec = get_metric_spec(getattr(profile, "competition", ""))
    if spec["metric_alignment"] == "inferred":
        # Unknown competition: re-infer with the measured task type so a
        # segmentation/detection profile never keeps the accuracy default
        # (v2.3.8 - masks/boxes are not class labels).
        spec = infer_metric_spec(getattr(profile, "task_type", ""))
    profile.metric_name = spec["metric_name"]
    profile.metric_direction = spec["metric_direction"]
    profile.metric_alignment = spec["metric_alignment"]
    profile.metric_label = spec["metric_label"]
    profile.metric_params = spec["metric_params"]
    profile.metric_min_delta = float(spec.get("min_delta", DEFAULT_MIN_DELTA))


SUPPORTED_METRICS = frozenset(METRIC_LABELS)
