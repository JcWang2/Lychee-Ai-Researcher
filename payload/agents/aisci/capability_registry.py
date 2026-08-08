# -*- coding: utf-8 -*-
"""capability_registry.py - v2.3 Capability Registry (declaration only).

Power boundary (frozen v2.3 contract):
  Analysis/HERA decide WHICH method and WHAT parameters;
  the registry only declares WHICH implementation capabilities exist,
  what inputs they accept, and under what constraints they can run.

The registry NEVER answers "which method should I choose?". It answers:
  - does this method exist? (method_id lookup)
  - is it usable for this dataset contract? (compatibility filter)
  - what parameters/preprocessing/validation does it accept? (schema)

Phase A+B built-ins (all sklearn/timm, no per-competition hardcoding):
  tabular.linear.logistic.v1     tabular.gbdt.histgb.v1
  tabular.neural.mlp.v1          image.embedding.timm.v1
  image.finetune.timm.v1         image.finetune.timm.v2
  image.finetune.ensemble.v1     ensemble.sklearn_soft_vote.v1
  text.embedding.tfidf.v1        text.neural.mlp.v1
  timeseries.lag_histgb.v1

v2.3.2: text modality has REAL capabilities (TF-IDF + linear/MLP heads);
timeseries is a first-class task type (lag-feature renderer) and the
tabular methods also accept it, so no dataset contract can ever land in
an empty capability space.

Phase C: run-local ephemeral capabilities (synthesized once by the LLM,
validated, then registered here and reused with parameter-only changes).
Ephemeral specs persist under <state_dir>/capabilities/ephemeral_specs.json
so a restarted daemon can replay the same invocation deterministically.
"""
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MethodSpec:
    """Declarative capability: what exists, its contract, NOT what to pick."""
    method_id: str
    family: str
    supported_modalities: List[str] = field(default_factory=list)
    supported_tasks: List[str] = field(default_factory=list)
    metric_outputs: Dict[str, str] = field(default_factory=dict)
    # key -> {"type": float|int|bool|str|list, "min", "max", "log",
    #         "choices", "max_len", "default"}
    parameter_schema: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    preprocessing_options: List[str] = field(default_factory=list)
    validation_schemes: List[str] = field(default_factory=list)
    renderer: str = ""
    resource_model: str = ""
    gpu: bool = False
    description: str = ""
    ephemeral: bool = False
    source_code: str = ""          # ephemeral: synthesized template body
    template_hash: str = ""        # ephemeral: sha256(source_code); built-in: filled by compiler
    broken: bool = False           # ephemeral capability failed at trial time
    # v2.5.2: declarative runnability defaults (replaces method-prefix
    # if/else in the compiler normalize()). Values are safe platform
    # defaults, never research choices; HERA may override preprocessing
    # and validation explicitly.
    default_preprocessing: List[str] = field(default_factory=list)
    default_validation: str = ""       # "" -> "stratified_kfold"
    validation_policy: str = "any"    # "any" | "fixed" (fixed forces
                                      # default_validation; used by
                                      # templates that implement exactly
                                      # one honest split)
    default_max_train_rows: int = 0    # 0 = no platform row cap; >0 caps
                                      # MAX_TRAIN_ROWS in compiled templates
                                      # (runnability default; HERA may still
                                      # request a smaller cap explicitly)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MethodSpec":
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})


BUILTIN_SPECS: List[MethodSpec] = [
    MethodSpec(
        method_id="tabular.linear.logistic.v1",
        family="linear",
        supported_modalities=["tabular"],
        supported_tasks=["classification", "regression", "timeseries"],
        metric_outputs={"logloss": "proba", "weighted_logloss": "proba",
                        "kl_div": "proba", "mean_auc_multilabel": "proba",
                        "auc": "proba", "binary_logloss": "proba",
                        "accuracy": "class", "f1_macro": "class",
                        "f1_micro": "class", "f1_binary": "class",
                        "f0_5": "class", "mcc": "class", "qwk": "class",
                        "rmse": "regression", "mae": "regression",
                        "r2": "regression", "spearman": "regression",
                        "pearson": "regression", "log_mae": "regression",
                        "rmsle": "regression", "kendall_tau": "regression",
                        "mean_angular_error": "regression"},
        parameter_schema={
            "C": {"type": "float", "min": 1e-4, "max": 1e2, "log": True,
                  "default": 1.0},
            "max_iter": {"type": "int", "min": 100, "max": 3000,
                         "default": 1000},
            "scaling": {"type": "str", "choices": ["standard", "none"],
                        "default": "standard"},
            "missing": {"type": "str", "choices": ["mean", "median"],
                        "default": "mean"},
            "val_seed": {"type": "int", "min": 0, "max": 99999,
                         "default": 42},
            "folds": {"type": "int", "min": 1, "max": 5, "default": 5},
        },
        preprocessing_options=["datetime_derive", "datetime_ordinal", "datetime_drop", "missing_value_impute", "frequency_encoding",
                               "standard_scaling"],
        validation_schemes=["stratified_kfold", "single_holdout"],
        default_max_train_rows=50000,
        renderer="tabular_linear",
        default_preprocessing=["missing_value_impute"],
        resource_model="sklearn_cost_v1",
        gpu=False,
        description="LogisticRegression / Ridge on tabular features with "
                    "frequency encoding, imputation and optional scaling."),
    MethodSpec(
        method_id="tabular.gbdt.histgb.v1",
        family="gradient_boosted_tree",
        supported_modalities=["tabular"],
        supported_tasks=["classification", "regression", "timeseries"],
        metric_outputs={"logloss": "proba", "weighted_logloss": "proba",
                        "kl_div": "proba", "mean_auc_multilabel": "proba",
                        "auc": "proba", "binary_logloss": "proba",
                        "accuracy": "class", "f1_macro": "class",
                        "f1_micro": "class", "f1_binary": "class",
                        "f0_5": "class", "mcc": "class", "qwk": "class",
                        "rmse": "regression", "mae": "regression",
                        "r2": "regression", "spearman": "regression",
                        "pearson": "regression", "log_mae": "regression",
                        "rmsle": "regression", "kendall_tau": "regression",
                        "mean_angular_error": "regression"},
        parameter_schema={
            "learning_rate": {"type": "float", "min": 0.005, "max": 0.3,
                              "log": True, "default": 0.05},
            "max_leaf_nodes": {"type": "int", "min": 8, "max": 256,
                               "default": 64},
            "max_iter": {"type": "int", "min": 50, "max": 1500,
                         "default": 300},
            "l2_regularization": {"type": "float", "min": 0.0, "max": 10.0,
                                  "default": 1.0},
            "early_stopping": {"type": "bool", "default": True},
            "scaling": {"type": "str", "choices": ["none", "standard"],
                        "default": "none"},
            "missing": {"type": "str", "choices": ["leave", "mean", "median"],
                        "default": "leave"},
            "val_seed": {"type": "int", "min": 0, "max": 99999,
                         "default": 42},
            "folds": {"type": "int", "min": 1, "max": 5, "default": 5},
        },
        preprocessing_options=["datetime_derive", "datetime_ordinal", "datetime_drop", "missing_value_native", "frequency_encoding"],
        validation_schemes=["stratified_kfold", "single_holdout"],
        default_max_train_rows=50000,
        renderer="tabular_histgb",
        default_preprocessing=[],
        resource_model="sklearn_cost_v1",
        gpu=False,
        description="HistGradientBoosting classifier/regressor (sklearn, "
                    "native NaN support, GPU-free, fast)."),
    MethodSpec(
        method_id="tabular.datetime_feature_histgb.v1",
        family="datetime_feature",
        supported_modalities=["tabular"],
        supported_tasks=["classification", "regression", "timeseries"],
        metric_outputs={"logloss": "proba", "weighted_logloss": "proba",
                        "kl_div": "proba", "mean_auc_multilabel": "proba",
                        "auc": "proba", "binary_logloss": "proba",
                        "accuracy": "class", "f1_macro": "class",
                        "f1_micro": "class", "f1_binary": "class",
                        "f0_5": "class", "mcc": "class", "qwk": "class",
                        "rmse": "regression", "mae": "regression",
                        "r2": "regression", "spearman": "regression",
                        "pearson": "regression", "log_mae": "regression",
                        "rmsle": "regression", "kendall_tau": "regression",
                        "mean_angular_error": "regression"},
        parameter_schema={
            "learning_rate": {"type": "float", "min": 0.005, "max": 0.3,
                              "log": True, "default": 0.05},
            "max_leaf_nodes": {"type": "int", "min": 8, "max": 256,
                               "default": 64},
            "max_iter": {"type": "int", "min": 50, "max": 1500,
                         "default": 300},
            "l2_regularization": {"type": "float", "min": 0.0, "max": 10.0,
                                  "default": 1.0},
            "early_stopping": {"type": "bool", "default": True},
            "scaling": {"type": "str", "choices": ["none", "standard"],
                        "default": "none"},
            "missing": {"type": "str", "choices": ["leave", "mean", "median"],
                        "default": "leave"},
            "val_seed": {"type": "int", "min": 0, "max": 99999,
                         "default": 42},
            "folds": {"type": "int", "min": 1, "max": 5, "default": 5},
        },
        preprocessing_options=["datetime_derive", "datetime_ordinal", "datetime_drop",
                               "missing_value_native", "frequency_encoding"],
        validation_schemes=["stratified_kfold", "single_holdout"],
        default_max_train_rows=50000,
        renderer="tabular_histgb",
        default_preprocessing=[],
        resource_model="sklearn_cost_v1",
        gpu=False,
        description="HistGradientBoosting with content-verified datetime "
                    "columns derived to calendar/elapsed features (year, "
                    "month, day, weekday, hour, seconds-since-median) so "
                    "trees never ordinal-encode timestamps."),
    MethodSpec(
        method_id="tabular.neural.mlp.v1",
        family="neural_net",
        supported_modalities=["tabular"],
        supported_tasks=["classification", "regression", "timeseries"],
        metric_outputs={"logloss": "proba", "weighted_logloss": "proba",
                        "kl_div": "proba", "mean_auc_multilabel": "proba",
                        "auc": "proba", "binary_logloss": "proba",
                        "accuracy": "class", "f1_macro": "class",
                        "f1_micro": "class", "f1_binary": "class",
                        "f0_5": "class", "mcc": "class", "qwk": "class",
                        "rmse": "regression", "mae": "regression",
                        "r2": "regression", "spearman": "regression",
                        "pearson": "regression", "log_mae": "regression",
                        "rmsle": "regression", "kendall_tau": "regression",
                        "mean_angular_error": "regression"},
        parameter_schema={
            "hidden_layers": {"type": "int", "min": 1, "max": 4,
                              "default": 2},
            "hidden_units": {"type": "int", "min": 16, "max": 512,
                             "default": 128},
            "alpha": {"type": "float", "min": 1e-5, "max": 1e-1, "log": True,
                      "default": 1e-3},
            "learning_rate_init": {"type": "float", "min": 1e-4, "max": 1e-2,
                                   "log": True, "default": 1e-3},
            "max_iter": {"type": "int", "min": 100, "max": 2000,
                         "default": 500},
            "early_stopping": {"type": "bool", "default": True},
            "val_seed": {"type": "int", "min": 0, "max": 99999,
                         "default": 42},
            "folds": {"type": "int", "min": 1, "max": 5, "default": 3},
        },
        preprocessing_options=["datetime_derive", "datetime_ordinal", "datetime_drop", "missing_value_impute", "frequency_encoding",
                               "standard_scaling"],
        validation_schemes=["stratified_kfold", "single_holdout"],
        default_max_train_rows=50000,
        renderer="tabular_mlp",
        default_preprocessing=["missing_value_impute", "standard_scaling"],
        resource_model="sklearn_cost_v1",
        gpu=False,
        description="MLPClassifier/MLPRegressor on scaled tabular features."),
    MethodSpec(
        method_id="image.embedding.timm.v1",
        family="image_embedding",
        supported_modalities=["image"],
        supported_tasks=["classification", "regression"],
        metric_outputs={"logloss": "proba", "weighted_logloss": "proba",
                        "kl_div": "proba", "mean_auc_multilabel": "proba",
                        "auc": "proba", "binary_logloss": "proba",
                        "accuracy": "class", "f1_macro": "class",
                        "f1_micro": "class", "f1_binary": "class",
                        "f0_5": "class", "mcc": "class", "qwk": "class",
                        "rmse": "regression", "mae": "regression",
                        "r2": "regression", "spearman": "regression",
                        "pearson": "regression", "log_mae": "regression",
                        "rmsle": "regression", "kendall_tau": "regression",
                        "mean_angular_error": "regression"},
        parameter_schema={
            "model_name": {"type": "str", "choices": [
                "efficientnet_b0", "efficientnet_b1", "resnet18", "resnet34",
                "resnet50", "mobilenetv3_large_100", "convnext_tiny",
                "vit_tiny_patch16_224", "swin_tiny_patch4_window7_224",
                "efficientnet_b0_rwightman"], "default": "efficientnet_b0"},
            "image_size": {"type": "int", "min": 64, "max": 256,
                           "default": 128},
            "C": {"type": "float", "min": 1e-4, "max": 1e2, "log": True,
                  "default": 1.0},
            "max_iter": {"type": "int", "min": 100, "max": 3000,
                         "default": 1000},
            "batch_size": {"type": "int", "min": 16, "max": 128,
                           "default": 64},
            "max_rows": {"type": "int", "min": 500, "max": 50000,
                         "default": 20000},
            "val_seed": {"type": "int", "min": 0, "max": 99999,
                         "default": 42},
            "folds": {"type": "int", "min": 1, "max": 5, "default": 3},
        },
        preprocessing_options=["cached_image_arrays", "imagenet_norm",
                               "pretrained_weight_cache"],
        validation_schemes=["stratified_kfold", "single_holdout"],
        default_max_train_rows=20000,
        renderer="image_embedding_timm",
        default_preprocessing=["cached_image_arrays", "imagenet_norm", "pretrained_weight_cache"],
        resource_model="timm_embed_cost_v1",
        gpu=True,
        description="Pretrained timm feature extractor over cached image "
                    "arrays + sklearn head (fast, low-GPU)."),
    MethodSpec(
        method_id="image.finetune.timm.v1",
        family="image_finetune",
        supported_modalities=["image"],
        supported_tasks=["classification", "regression"],
        metric_outputs={"logloss": "proba", "weighted_logloss": "proba",
                        "kl_div": "proba", "mean_auc_multilabel": "proba",
                        "auc": "proba", "binary_logloss": "proba",
                        "accuracy": "class", "f1_macro": "class",
                        "f1_micro": "class", "f1_binary": "class",
                        "f0_5": "class", "mcc": "class", "qwk": "class",
                        "rmse": "regression", "mae": "regression",
                        "r2": "regression", "spearman": "regression",
                        "pearson": "regression", "log_mae": "regression",
                        "rmsle": "regression", "kendall_tau": "regression",
                        "mean_angular_error": "regression"},
        parameter_schema={
            "model_name": {"type": "str", "choices": [
                "efficientnet_b0", "efficientnet_b1", "resnet18", "resnet34",
                "mobilenetv3_large_100", "convnext_tiny"],
                           "default": "efficientnet_b0"},
            "image_size": {"type": "int", "min": 64, "max": 256,
                           "default": 128},
            "epochs": {"type": "int", "min": 2, "max": 10, "default": 4},
            "lr": {"type": "float", "min": 1e-5, "max": 1e-2, "log": True,
                   "default": 3e-4},
            "batch_size": {"type": "int", "min": 16, "max": 64,
                           "default": 32},
            "weight_decay": {"type": "float", "min": 0.0, "max": 0.01,
                             "default": 1e-4},
            "early_stop_patience": {"type": "int", "min": 1, "max": 5,
                                    "default": 2},
            "max_rows": {"type": "int", "min": 500, "max": 50000,
                         "default": 15000},
            "val_seed": {"type": "int", "min": 0, "max": 99999,
                         "default": 42},
        },
        preprocessing_options=["cached_image_arrays", "imagenet_norm",
                               "pretrained_weight_cache", "flip_augment"],
        validation_schemes=["single_holdout"],
        default_max_train_rows=20000,
        renderer="image_finetune_timm",
        default_preprocessing=["cached_image_arrays", "imagenet_norm", "pretrained_weight_cache"],
        resource_model="timm_finetune_cost_v1",
        gpu=True,
        description="Short fine-tune of a pretrained timm model on cached "
                    "image arrays with early stopping (S3/S4 structural)."),
    MethodSpec(
        method_id="image.finetune.timm.v2",
        family="image_finetune",
        supported_modalities=["image"],
        supported_tasks=["classification", "regression"],
        metric_outputs={"logloss": "proba", "weighted_logloss": "proba",
                        "kl_div": "proba", "mean_auc_multilabel": "proba",
                        "auc": "proba", "binary_logloss": "proba",
                        "accuracy": "class", "f1_macro": "class",
                        "f1_micro": "class", "f1_binary": "class",
                        "f0_5": "class", "mcc": "class", "qwk": "class",
                        "rmse": "regression", "mae": "regression",
                        "r2": "regression", "spearman": "regression",
                        "pearson": "regression", "log_mae": "regression",
                        "rmsle": "regression", "kendall_tau": "regression",
                        "mean_angular_error": "regression"},
        parameter_schema={
            "model_name": {"type": "str", "choices": [
                "efficientnet_b0", "efficientnet_b1", "resnet18", "resnet34",
                "resnet50", "mobilenetv3_large_100", "convnext_tiny",
                "vit_tiny_patch16_224", "swin_tiny_patch4_window7_224",
                "efficientnet_b0_rwightman"],
                           "default": "efficientnet_b0"},
            "image_size": {"type": "int", "min": 64, "max": 384,
                           "default": 224},
            "epochs": {"type": "int", "min": 2, "max": 12, "default": 6},
            "lr": {"type": "float", "min": 1e-5, "max": 1e-2, "log": True,
                   "default": 3e-4},
            "batch_size": {"type": "int", "min": 16, "max": 64,
                           "default": 32},
            "weight_decay": {"type": "float", "min": 0.0, "max": 0.01,
                             "default": 1e-4},
            "early_stop_patience": {"type": "int", "min": 1, "max": 5,
                                    "default": 3},
            "max_rows": {"type": "int", "min": 500, "max": 50000,
                         "default": 15000},
            "val_seed": {"type": "int", "min": 0, "max": 99999,
                         "default": 42},
            "lr_schedule": {"type": "str", "choices": ["cosine", "step"],
                            "default": "cosine"},
            "augment": {"type": "str",
                        "choices": ["flip", "rcrop", "strong", "none"],
                        "default": "flip"},
            "amp": {"type": "bool", "default": True},
            "tta_flip": {"type": "bool", "default": True},
            "label_smoothing": {"type": "float", "min": 0.0, "max": 0.3,
                                "default": 0.0},
        },
        preprocessing_options=["cached_image_arrays", "imagenet_norm",
                               "pretrained_weight_cache", "flip_augment"],
        validation_schemes=["single_holdout"],
        default_max_train_rows=20000,
        renderer="image_finetune_timm_v2",
        default_preprocessing=["cached_image_arrays", "imagenet_norm", "pretrained_weight_cache"],
        resource_model="timm_finetune_cost_v1",
        gpu=True,
        description="Longer pretrained timm fine-tune (up to 12 epochs, "
                    "cosine/step LR, flip/random-crop/strong augmentation, "
                    "AMP, H-flip TTA, label smoothing) on cached image "
                    "arrays; writes probability-normalized %.9f submissions."),
    MethodSpec(
        method_id="image.finetune.ensemble.v1",
        family="image_finetune",
        supported_modalities=["image"],
        supported_tasks=["classification", "regression"],
        metric_outputs={"logloss": "proba", "weighted_logloss": "proba",
                        "kl_div": "proba", "mean_auc_multilabel": "proba",
                        "auc": "proba", "binary_logloss": "proba",
                        "accuracy": "class", "f1_macro": "class",
                        "f1_micro": "class", "f1_binary": "class",
                        "f0_5": "class", "mcc": "class", "qwk": "class",
                        "rmse": "regression", "mae": "regression",
                        "r2": "regression", "spearman": "regression",
                        "pearson": "regression", "log_mae": "regression",
                        "rmsle": "regression", "kendall_tau": "regression",
                        "mean_angular_error": "regression"},
        parameter_schema={
            "model_names": {"type": "list", "max_len": 3,
                            "default": ["efficientnet_b0"]},
            "seeds": {"type": "list", "max_len": 2, "default": [42]},
            "image_size": {"type": "int", "min": 64, "max": 384,
                           "default": 224},
            "epochs": {"type": "int", "min": 2, "max": 12, "default": 6},
            "lr": {"type": "float", "min": 1e-5, "max": 1e-2, "log": True,
                   "default": 3e-4},
            "batch_size": {"type": "int", "min": 16, "max": 64,
                           "default": 24},
            "weight_decay": {"type": "float", "min": 0.0, "max": 0.01,
                             "default": 1e-4},
            "early_stop_patience": {"type": "int", "min": 1, "max": 5,
                                    "default": 3},
            "max_rows": {"type": "int", "min": 500, "max": 50000,
                         "default": 15000},
            "val_seed": {"type": "int", "min": 0, "max": 99999,
                         "default": 42},
            "lr_schedule": {"type": "str", "choices": ["cosine", "step"],
                            "default": "cosine"},
            "augment": {"type": "str",
                        "choices": ["flip", "rcrop", "strong", "none"],
                        "default": "flip"},
            "amp": {"type": "bool", "default": True},
            "tta_flip": {"type": "bool", "default": True},
            "label_smoothing": {"type": "float", "min": 0.0, "max": 0.3,
                                "default": 0.0},
        },
        preprocessing_options=["cached_image_arrays", "imagenet_norm",
                               "pretrained_weight_cache", "flip_augment"],
        validation_schemes=["single_holdout"],
        default_max_train_rows=20000,
        renderer="image_finetune_ensemble",
        default_preprocessing=["cached_image_arrays", "imagenet_norm", "pretrained_weight_cache"],
        resource_model="timm_finetune_cost_v1",
        gpu=True,
        description="Ensemble fine-tune: up to 3 model architectures x 2 "
                    "seeds (max 4 members), logit-averaged predictions with "
                    "AMP/H-flip TTA and probability-normalized %.9f output."),
    MethodSpec(
        method_id="image.pixel.baseline.v1",
        family="image_pixel_baseline",
        supported_modalities=["image_pixel"],
        supported_tasks=["regression"],
        metric_outputs={"rmse": "regression", "mae": "regression",
                        "r2": "regression", "spearman": "regression",
                        "pearson": "regression", "log_mae": "regression",
                        "rmsle": "regression", "kendall_tau": "regression",
                        "mean_angular_error": "regression"},
        parameter_schema={
            "basis": {"type": "str", "choices": [
                "per_image", "spatial", "global"], "default": "per_image"},
            "val_seed": {"type": "int", "min": 0, "max": 99999,
                         "default": 42},
        },
        preprocessing_options=[],
        validation_schemes=["single_holdout"],
        renderer="image_pixel_regression",
        default_preprocessing=[],
        default_validation="single_holdout",
        validation_policy="fixed",
        resource_model="pixel_sklearn_cost_v1",
        gpu=False,
        description="Deterministic pixel-level baseline for image-to-image "
                    "regression (per-image / per-pixel-position / global "
                    "mean of training target intensities; image-level "
                    "holdout, no pixel leakage)."),
    MethodSpec(
        method_id="image.mask.rle.baseline.v1",
        family="image_mask_baseline",
        supported_modalities=["image_mask"],
        supported_tasks=["segmentation"],
        metric_outputs={"dice": "binary", "iou_mean": "binary",
                        "f0_5": "binary"},
        parameter_schema={
            "strategy": {"type": "str", "choices": ["empty", "dense"],
                         "default": "empty"},
            "val_seed": {"type": "int", "min": 0, "max": 99999,
                         "default": 42},
            "max_oof_pixels": {"type": "int", "min": 1000, "max": 2000000,
                               "default": 400000},
        },
        preprocessing_options=[],
        validation_schemes=["single_holdout"],
        renderer="image_mask_rle_baseline",
        default_preprocessing=[],
        default_validation="single_holdout",
        validation_policy="fixed",
        resource_model="pixel_sklearn_cost_v1",
        gpu=False,
        description="Deterministic RLE-mask baseline (empty or dense masks): "
                    "decodes train masks for an image-level holdout OOF and "
                    "writes a valid RLE submission for every sample id."),
    MethodSpec(
        method_id="image.detection.bbox.baseline.v1",
        family="image_detection_baseline",
        supported_modalities=["image_detection"],
        supported_tasks=["detection"],
        metric_outputs={"map_at_k": "rank", "f1_macro": "rank"},
        parameter_schema={
            "val_seed": {"type": "int", "min": 0, "max": 99999,
                         "default": 42},
            "max_candidates": {"type": "int", "min": 2, "max": 256,
                               "default": 32},
        },
        preprocessing_options=[],
        validation_schemes=["single_holdout"],
        renderer="image_detection_bbox_baseline",
        default_preprocessing=[],
        default_validation="single_holdout",
        validation_policy="fixed",
        resource_model="pixel_sklearn_cost_v1",
        gpu=False,
        description="Deterministic bbox baseline: no predicted boxes in the "
                    "submission (organizer placeholder / empty list) and a "
                    "class-frequency OOF ranking so the internal map@k is "
                    "defined and beatable."),
    MethodSpec(
        method_id="audio.tabular.baseline.v1",
        family="audio_baseline",
        supported_modalities=["audio"],
        supported_tasks=["classification", "regression"],
        metric_outputs={"accuracy": "class", "label_ranking_ap": "rank",
                        "mean_auc_multilabel": "proba",
                        "binary_logloss": "proba"},
        parameter_schema={
            "val_seed": {"type": "int", "min": 0, "max": 99999,
                         "default": 42},
        },
        preprocessing_options=[],
        validation_schemes=["single_holdout"],
        renderer="audio_tabular_baseline",
        default_preprocessing=[],
        default_validation="single_holdout",
        validation_policy="fixed",
        resource_model="sklearn_cost_v1",
        gpu=False,
        description="Deterministic audio baseline: majority class (or "
                    "per-class frequency) learned from the train table; "
                    "valid for single-label and multi-label audio tasks."),
    MethodSpec(
        method_id="ensemble.sklearn_soft_vote.v1",
        family="ensemble",
        supported_modalities=["tabular"],
        supported_tasks=["classification", "regression", "timeseries"],
        metric_outputs={"logloss": "proba", "weighted_logloss": "proba",
                        "kl_div": "proba", "mean_auc_multilabel": "proba",
                        "auc": "proba", "binary_logloss": "proba",
                        "accuracy": "class", "f1_macro": "class",
                        "f1_micro": "class", "f1_binary": "class",
                        "f0_5": "class", "mcc": "class", "qwk": "class",
                        "rmse": "regression", "mae": "regression",
                        "r2": "regression", "spearman": "regression",
                        "pearson": "regression", "log_mae": "regression",
                        "rmsle": "regression", "kendall_tau": "regression",
                        "mean_angular_error": "regression"},
        parameter_schema={
            "model_ids": {"type": "list", "max_len": 3, "default": [
                "tabular.linear.logistic.v1", "tabular.gbdt.histgb.v1"]},
            "weights": {"type": "list", "max_len": 3, "default": []},
            "val_seed": {"type": "int", "min": 0, "max": 99999,
                         "default": 42},
            "folds": {"type": "int", "min": 1, "max": 5, "default": 3},
        },
        preprocessing_options=["datetime_derive", "datetime_ordinal", "datetime_drop", "frequency_encoding", "missing_value_impute"],
        validation_schemes=["stratified_kfold", "single_holdout"],
        default_max_train_rows=50000,
        renderer="ensemble_sklearn_soft_vote",
        default_preprocessing=["missing_value_impute"],
        resource_model="sklearn_ensemble_cost_v1",
        gpu=False,
        description="Soft-vote ensemble of up to 3 tabular sklearn methods "
                    "with optional weights (S4 confirmation)."),
    MethodSpec(
        method_id="text.embedding.tfidf.v1",
        family="text_embedding",
        supported_modalities=["text"],
        supported_tasks=["classification", "regression"],
        metric_outputs={"logloss": "proba", "weighted_logloss": "proba",
                        "kl_div": "proba", "mean_auc_multilabel": "proba",
                        "auc": "proba", "binary_logloss": "proba",
                        "accuracy": "class", "f1_macro": "class",
                        "f1_micro": "class", "f1_binary": "class",
                        "f0_5": "class", "mcc": "class", "qwk": "class",
                        "rmse": "regression", "mae": "regression",
                        "r2": "regression", "spearman": "regression",
                        "pearson": "regression", "log_mae": "regression",
                        "rmsle": "regression", "kendall_tau": "regression",
                        "mean_angular_error": "regression"},
        parameter_schema={
            "max_features": {"type": "int", "min": 1000, "max": 50000,
                             "default": 20000},
            "ngram_max": {"type": "int", "min": 1, "max": 3, "default": 2},
            "min_df": {"type": "int", "min": 1, "max": 10, "default": 2},
            "C": {"type": "float", "min": 1e-4, "max": 1e2, "log": True,
                  "default": 1.0},
            "max_iter": {"type": "int", "min": 100, "max": 3000,
                         "default": 1000},
            "val_seed": {"type": "int", "min": 0, "max": 99999,
                         "default": 42},
            "folds": {"type": "int", "min": 1, "max": 5, "default": 3},
        },
        preprocessing_options=["datetime_derive", "datetime_ordinal", "datetime_drop", "tfidf_vectorization"],
        validation_schemes=["stratified_kfold", "single_holdout"],
        renderer="text_tfidf_linear",
        default_preprocessing=["tfidf_vectorization"],
        resource_model="sklearn_cost_v1",
        gpu=False,
        description="TF-IDF vectorization of content-verified text columns "
                    "plus a LogisticRegression / Ridge head (sparse, fast)."),
    MethodSpec(
        method_id="text.neural.mlp.v1",
        family="text_neural",
        supported_modalities=["text"],
        supported_tasks=["classification", "regression"],
        metric_outputs={"logloss": "proba", "weighted_logloss": "proba",
                        "kl_div": "proba", "mean_auc_multilabel": "proba",
                        "auc": "proba", "binary_logloss": "proba",
                        "accuracy": "class", "f1_macro": "class",
                        "f1_micro": "class", "f1_binary": "class",
                        "f0_5": "class", "mcc": "class", "qwk": "class",
                        "rmse": "regression", "mae": "regression",
                        "r2": "regression", "spearman": "regression",
                        "pearson": "regression", "log_mae": "regression",
                        "rmsle": "regression", "kendall_tau": "regression",
                        "mean_angular_error": "regression"},
        parameter_schema={
            "max_features": {"type": "int", "min": 500, "max": 20000,
                             "default": 4096},
            "ngram_max": {"type": "int", "min": 1, "max": 3, "default": 2},
            "min_df": {"type": "int", "min": 1, "max": 10, "default": 2},
            "hidden_units": {"type": "int", "min": 16, "max": 512,
                             "default": 128},
            "alpha": {"type": "float", "min": 1e-5, "max": 1e-1, "log": True,
                      "default": 1e-3},
            "max_iter": {"type": "int", "min": 100, "max": 2000,
                         "default": 500},
            "val_seed": {"type": "int", "min": 0, "max": 99999,
                         "default": 42},
            "folds": {"type": "int", "min": 1, "max": 5, "default": 3},
        },
        preprocessing_options=["datetime_derive", "datetime_ordinal", "datetime_drop", "tfidf_vectorization"],
        validation_schemes=["stratified_kfold", "single_holdout"],
        renderer="text_tfidf_mlp",
        default_preprocessing=["tfidf_vectorization"],
        resource_model="sklearn_cost_v1",
        gpu=False,
        description="TF-IDF vectorization plus MLPClassifier/MLPRegressor "
                    "head (dense adapter over a bounded feature space)."),
    MethodSpec(
        method_id="timeseries.lag_histgb.v1",
        family="timeseries_lag",
        supported_modalities=["tabular"],
        supported_tasks=["timeseries", "regression"],
        metric_outputs={"rmse": "regression", "mae": "regression",
                        "r2": "regression", "spearman": "regression",
                        "pearson": "regression", "log_mae": "regression",
                        "rmsle": "regression", "kendall_tau": "regression",
                        "mean_angular_error": "regression"},
        parameter_schema={
            "max_lag": {"type": "int", "min": 1, "max": 14, "default": 7},
            "rolling_window": {"type": "int", "min": 0, "max": 30,
                               "default": 7},
            "learning_rate": {"type": "float", "min": 1e-3, "max": 0.5,
                              "log": True, "default": 0.05},
            "max_leaf_nodes": {"type": "int", "min": 16, "max": 256,
                               "default": 64},
            "max_iter": {"type": "int", "min": 100, "max": 2000,
                         "default": 300},
            "val_seed": {"type": "int", "min": 0, "max": 99999,
                         "default": 42},
        },
        preprocessing_options=["lag_features"],
        validation_schemes=["time_holdout"],
        default_max_train_rows=50000,
        renderer="timeseries_lag",
        default_preprocessing=["lag_features"],
        default_validation="time_holdout",
        validation_policy="fixed",
        resource_model="sklearn_cost_v1",
        gpu=False,
        description="Time-ordered lag / rolling features over the detected "
                    "date column with a HistGradientBoosting head and a "
                    "strict time-holdout validation (no random-split "
                    "leakage)."),
]


class CapabilityRegistry:
    """Declarative capability registry with ephemeral persistence."""

    def __init__(self, specs: Optional[List[MethodSpec]] = None,
                 ephemeral_path: Optional[str] = None):
        self._specs: Dict[str, MethodSpec] = {}
        for s in (specs if specs is not None else BUILTIN_SPECS):
            # Own a per-instance copy: runtime flags (e.g. broken) must never
            # leak across registry instances sharing module-level built-ins.
            self._specs[s.method_id] = MethodSpec.from_dict(s.to_dict())
        self.ephemeral_path = ephemeral_path or ""
        if self.ephemeral_path:
            self._load_ephemeral()

    # ---- lookup / filtering ----
    def get(self, method_id: str) -> Optional[MethodSpec]:
        return self._specs.get(method_id)

    def all(self) -> List[MethodSpec]:
        return list(self._specs.values())

    def compatible(self, modality: str = "", task_type: str = "",
                   metric_name: str = "") -> List[MethodSpec]:
        """Compatibility filter: usable for THIS dataset contract.

        Filtering only - it never ranks or chooses.
        """
        out = []
        for s in self._specs.values():
            if s.broken:
                continue
            if modality and s.supported_modalities and \
                    modality not in s.supported_modalities:
                continue
            if task_type and s.supported_tasks and \
                    task_type not in s.supported_tasks:
                continue
            if metric_name and s.metric_outputs and \
                    metric_name not in s.metric_outputs:
                continue
            out.append(s)
        return out

    def prompt_summary(self, modality: str = "", task_type: str = "",
                       metric_name: str = "", max_chars: int = 2400) -> str:
        """Compact capability list for the HERA proposer prompt (no choice)."""
        lines = []
        for s in self.compatible(modality, task_type, metric_name):
            params = ", ".join(
                "%s:%s" % (k, _schema_hint(v)) for k, v in
                sorted(s.parameter_schema.items())[:8])
            lines.append("%s | %s | params(%s)" % (s.method_id, s.description,
                                                   params))
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        return text or "(no compatible capability declared)"

    def set_broken(self, method_id: str, reason: str = "") -> None:
        """Mark a capability broken after verified trial failure (PACT).

        Built-ins can be marked broken in-memory for the current process;
        ephemeral capabilities persist the flag so a restarted daemon skips
        them too.
        """
        spec = self._specs.get(method_id)
        if spec is None:
            return
        spec.broken = True
        if spec.ephemeral:
            self._save_ephemeral()

    # ---- ephemeral (Phase C) ----
    def register_ephemeral(self, spec: MethodSpec) -> None:
        spec.ephemeral = True
        self._specs[spec.method_id] = spec
        if self.ephemeral_path:
            self._save_ephemeral()

    def _load_ephemeral(self) -> None:
        try:
            with open(self.ephemeral_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for d in data:
                s = MethodSpec.from_dict(d)
                if s.method_id and s.source_code:
                    self._specs[s.method_id] = s
        except (OSError, ValueError):
            pass

    def _save_ephemeral(self) -> None:
        import os
        try:
            os.makedirs(os.path.dirname(self.ephemeral_path), exist_ok=True)
            data = [s.to_dict() for s in self._specs.values() if s.ephemeral]
            with open(self.ephemeral_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=1)
        except OSError:
            pass


def _schema_hint(v: dict) -> str:
    t = v.get("type", "")
    if t == "float":
        return "%g..%g" % (v.get("min", 0), v.get("max", 1))
    if t == "int":
        return "%d..%d" % (v.get("min", 0), v.get("max", 100))
    if t == "str" and v.get("choices"):
        return "/".join(str(c) for c in v["choices"][:4])
    if t == "bool":
        return "bool"
    if t == "list":
        return "list<=%d" % v.get("max_len", 3)
    return t


def load_ephemeral_path(state_dir: str) -> str:
    """Convention: <state_dir>/capabilities/ephemeral_specs.json"""
    import os
    return os.path.join(str(state_dir), "capabilities", "ephemeral_specs.json")

def synthesis_usage_path(state_dir: str) -> str:
    """Convention: <state_dir>/capabilities/synthesis_usage.json"""
    import os
    return os.path.join(str(state_dir), "capabilities", "synthesis_usage.json")


def load_synthesis_usage(state_dir: str) -> dict:
    """Phase C budget: {used: int, actions: [str]}."""
    try:
        with open(synthesis_usage_path(state_dir), "r",
                  encoding="utf-8") as fh:
            d = json.load(fh)
        if isinstance(d, dict):
            return {"used": int(d.get("used", 0) or 0),
                    "actions": list(d.get("actions", []) or [])}
    except (OSError, ValueError):
        pass
    return {"used": 0, "actions": []}


def save_synthesis_usage(state_dir: str, usage: dict) -> None:
    """Persist the Phase C synthesis budget (atomic tmp+replace)."""
    import os
    try:
        path = synthesis_usage_path(state_dir)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"used": int(usage.get("used", 0) or 0),
                       "actions": list(usage.get("actions", []) or [])},
                      fh, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except OSError:
        pass
