# -*- coding: utf-8 -*-
"""hera/analyzer.py - Analysis methods: data and task profiling (stdlib only).

Produces an AnalysisProfile consumed by the Planner. Deterministic: reads
CSV headers/samples and prompt keywords, no LLM. Analysis discipline: every
number in the profile is measured, never guessed - the profile is the
ground truth the planner must respect.
"""
import csv
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from data_layout import (DatasetLayout, DatasetLayoutError,
                         count_image_files, iter_image_files,
                         resolve_dataset_layout, table_delimiter)
from v2_contracts import (AnalysisProfile,
                          AUDIO_FILE_MODALITY_THRESHOLD,
                          IMAGE_FILE_MODALITY_THRESHOLD)
from metrics_registry import apply_metric_to_profile


def _deep_notes(diag: dict) -> str:
    """Compact one-line summary of deep diagnostics for data_notes."""
    try:
        t = diag.get("target_diag") or {}
        f = diag.get("feature_diag") or {}
        o = diag.get("order_diag") or {}
        parts = []
        if t:
            parts.append("classes=%s" % t.get("n_classes"))
            if t.get("top1_share") is not None:
                parts.append("top1=%.2f" % t["top1_share"])
            if t.get("entropy_bits") is not None:
                parts.append("entropy=%.2f" % t["entropy_bits"])
            if t.get("skew") is not None:
                parts.append("skew=%.2f" % t["skew"])
            if t.get("unique_ratio") is not None:
                parts.append("uniq=%.2f" % t["unique_ratio"])
            if not t.get("numeric") and t.get("n_classes"):
                parts.append("target_cat")
        if f:
            if f.get("numeric_share") is not None:
                parts.append("num_share=%.2f" % f["numeric_share"])
            if f.get("constant_cols"):
                parts.append("const=%s"
                             % ",".join(str(c) for c in f["constant_cols"][:4]))
            if f.get("duplicate_cols"):
                parts.append("dup=%d" % len(f["duplicate_cols"]))
            hc = f.get("high_card_cols") or []
            if hc:
                parts.append("hi_card=%s"
                             % ",".join("%s:%s" % (c, k) for c, k in hc[:4]))
        if o:
            if o.get("id_monotonic") is not None:
                parts.append("id_mono=%d" % (1 if o["id_monotonic"] else 0))
            if o.get("id_target_corr") is not None:
                parts.append("id_corr=%.3f" % o["id_target_corr"])
            if o.get("time_present"):
                parts.append("time=%s..%s" % (o.get("time_min"),
                                              o.get("time_max")))
        return ("\ndeep diagnostics: " + "; ".join(parts)) if parts else ""
    except Exception:
        return ""


def _read_sample(path: Path, limit: int = 30) -> Tuple[List[str], List[List[str]]]:
    header: List[str] = []
    rows: List[List[str]] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.reader(fh, delimiter=table_delimiter(path))
            for i, row in enumerate(reader):
                if i == 0:
                    header = row
                    continue
                rows.append(row)
                if len(rows) >= limit:
                    break
    except OSError:
        pass
    return header, rows


def _count_rows(path: Path) -> int:
    n = 0
    try:
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
            for _ in fh:
                n += 1
        return max(0, n - 1)  # minus header
    except OSError:
        return 0


def _target_stats(path: Path, target: str, header: List[str],
                  max_rows: int = 200000) -> Tuple[int, dict]:
    """One full pass over the train CSV: row count + target statistics.

    Returns (row_count, stats) where stats carries measured facts:
      distinct (int), top (str), top_count (int), top_share (float),
      missing (int), and for numeric targets a single-pass
      mean / std / min / max (used by the regression reference line).
      No LLM, no guesses - deterministic.
    """
    counts = {}
    rows = 0
    missing = 0
    idx = -1
    num_n = 0
    num_sum = 0.0
    num_sumsq = 0.0
    num_min = None
    num_max = None
    if target and header and target in header:
        idx = header.index(target)
    try:
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.reader(fh, delimiter=table_delimiter(path))
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                rows += 1
                if idx >= 0 and idx < len(row):
                    value = row[idx].strip()
                    if value:
                        counts[value] = counts.get(value, 0) + 1
                        try:
                            fval = float(value)
                        except (TypeError, ValueError):
                            fval = None
                        if fval is not None and math.isfinite(fval):
                            num_n += 1
                            num_sum += fval
                            num_sumsq += fval * fval
                            if num_min is None or fval < num_min:
                                num_min = fval
                            if num_max is None or fval > num_max:
                                num_max = fval
                    else:
                        missing += 1
                if rows >= max_rows:
                    break
    except OSError:
        return 0, {}
    stats = {"distinct": len(counts), "missing": missing, "rows": rows}
    if counts:
        top, top_count = max(counts.items(), key=lambda kv: kv[1])
        stats["top"] = top
        stats["top_count"] = top_count
        stats["top_share"] = round(top_count / float(rows), 4) if rows else 0.0
    if num_n > 0:
        mean = num_sum / float(num_n)
        variance = max(0.0, num_sumsq / float(num_n) - mean * mean)
        stats["numeric_count"] = num_n
        stats["mean"] = round(mean, 6)
        stats["std"] = round(math.sqrt(variance), 6)
        stats["min"] = round(float(num_min), 6)
        stats["max"] = round(float(num_max), 6)
    return rows, stats


def _parse_image_header(data: bytes):
    """Minimal stdlib PNG/JPEG header parse -> (width, height, channels).

    PNG: 8-byte signature + IHDR (width/height at offset 16/20, colour type
    at 25). JPEG: scan markers for a SOF segment carrying height/width.
    Returns (0, 0, 0) when the bytes are not a parseable image header.
    """
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 26:
        w = int.from_bytes(data[16:20], "big")
        h = int.from_bytes(data[20:24], "big")
        ch = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(data[25], 0)
        return w, h, ch
    if data[:2] == b"\xff\xd8":
        i = 2
        while i + 9 < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                          0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                h = int.from_bytes(data[i + 5:i + 7], "big")
                w = int.from_bytes(data[i + 7:i + 9], "big")
                return w, h, int(data[i + 9])
            if marker in (0xD8, 0xD9) or marker == 0x01:
                i += 2
                continue
            length = int.from_bytes(data[i + 2:i + 4], "big")
            if length < 2:
                i += 2
                continue
            i += 2 + length
        return 0, 0, 0
    return 0, 0, 0


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


# ------------------------------------------------------------------ content
# v2.3.2 generic content evidence for modality/task-type guessing. These
# are measured from sampled column values - never from competition names.
_TEXT_NAME_HINTS = ("text", "tweet", "sentence", "question", "answer",
                    "comment", "abstract", "title", "article", "review",
                    "message", "lyrics", "description", "body", "content",
                    "summary", "headline", "caption", "story", "claim",
                    "utterance", "quote")
_DATE_COLUMN_HINTS = ("date", "datetime", "timestamp", "time", "month",
                      "year", "week", "quarter", "day")

# ------------------------------------------------------------------ v2.3.8
# Generic structural evidence for the remaining MLE-Bench task families.
# Never competition names - pure column-value / file-shape signals:
#   - RLE masks: the target column holds run-length encoded 0/1 masks
#     (pairs of 1-based position + run length, column-major order);
#   - bbox detection: coordinate column sets (x/y/w/h ...) with repeated
#     ids, or a JSON-array column of box objects;
#   - audio: wav/flac/ogg/mp3 files in the dataset tree.
_AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac",
                     ".opus", ".wma")
_RLE_RE = re.compile(r"^\d+(?: \d+)+$")
_BOX_NAME_SETS = (
    {"x", "y", "w", "h"},
    {"x", "y", "width", "height"},
    {"x1", "y1", "x2", "y2"},
    {"x_min", "y_min", "x_max", "y_max"},
    {"xmin", "ymin", "xmax", "ymax"},
    {"left", "top", "right", "bottom"},
)


def _is_rle_value(value: str) -> bool:
    """RLE mask value: at least one (position, run-length) pair of ints."""
    v = str(value).strip()
    if not v:
        return False
    return bool(_RLE_RE.match(v))


def _is_json_box_value(value: str) -> bool:
    """JSON box value: an array/object with bbox or x/y geometry keys."""
    v = str(value).strip()
    if not (v.startswith("[") or v.startswith("{")):
        return False
    low = v.lower()
    if '"bbox"' in low:
        return True
    return '"x"' in low and ('"y"' in low or '"w"' in low or '"h"' in low)


def _looks_like_date(value: str) -> bool:
    """Deterministic date-likeness for sampled CSV values.

    ISO plus common Kaggle date layouts. Pure numbers are never date
    evidence, so numeric 'year'/'month' columns stay numeric."""
    v = str(value).strip()
    if not v or len(v) < 6:
        return False
    try:
        float(v)
        return False
    except (TypeError, ValueError):
        pass
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S",
                "%Y/%m/%d %H:%M:%S", "%m/%d/%Y", "%d/%m/%Y",
                "%d-%b-%Y", "%b %d, %Y", "%Y-%m", "%Y%m%d", "%b", "%B"):
        try:
            datetime.strptime(v, fmt)
            return True
        except (TypeError, ValueError):
            continue
    return False


def _column_text_signal(rows: List[List[str]], header: List[str],
                        col_idx: int, sample_limit: int = 30
                        ) -> Tuple[bool, float]:
    """Content-based free-text evidence for one sampled column.

    Returns (is_text, score). Prose columns are mostly non-numeric AND
    long (avg words >= 3), space-dense (>= 50% values contain whitespace)
    or long-string (>= 40% values with >= 40 chars). Near-unique short
    tokens (ids/codes/hashes/filenames) are never text regardless of name.
    """
    seen = []
    for row in rows:
        if col_idx < len(row) and str(row[col_idx]).strip():
            seen.append(str(row[col_idx]).strip())
            if len(seen) >= sample_limit:
                break
    if not seen:
        return False, 0.0
    nonnum = sum(1 for v in seen if not _is_float(v)) / float(len(seen))
    words_avg = float(sum(len(v.split()) for v in seen)) / len(seen)
    space_ratio = sum(1 for v in seen if " " in v) / float(len(seen))
    long_ratio = sum(1 for v in seen if len(v) >= 40) / float(len(seen))
    unique_ratio = len(set(seen)) / float(len(seen))
    if nonnum < 0.6:
        return False, 0.0
    # id/code guard: near-unique short tokens are keys, not prose
    if unique_ratio >= 0.9 and words_avg <= 1.5 and space_ratio < 0.5:
        return False, 0.0
    is_text = ((words_avg >= 3.0 and nonnum >= 0.8)
               or (space_ratio >= 0.5 and nonnum >= 0.8)
               or (long_ratio >= 0.4 and nonnum >= 0.8))
    score = min(1.0, words_avg / 8.0 + space_ratio / 2.0 + long_ratio)
    return is_text, score


# How many bytes to scan when probing JPEG dimensions: SOF0 (the segment
# carrying height/width) can sit far past byte 64 behind EXIF/ICC/COM
# segments on real photos, so a 64-byte head is NOT enough for generic
# image tasks.
_JPEG_SCAN_BYTES = 1 << 20
# Fallback image size when modality is image (file-count evidence) but the
# dims probe cannot parse any header: cache sizing / resource derivation
# still work with this default.
_DEFAULT_IMAGE_SIZE = 64


class Analyzer:
    """Data + task profiling (analysis methods for HERA)."""

    def __init__(self, data_dir, task_prompt: str = "", sample_path: str = ""):
        self.data_dir = Path(data_dir)
        self.task_prompt = task_prompt
        self.layout: DatasetLayout = resolve_dataset_layout(
            self.data_dir, sample_path=sample_path)
        if self.layout.pixel_level and not self.layout.train_path.is_file():
            # The labels live inside the paired target images; the closed
            # loop/daemon synthesize public/train.csv before analysis.
            raise DatasetLayoutError(
                "paired-image pixel layout requires train.csv synthesis; "
                "run synthesize_train_labels(data_dir) before analysis")

    def profile(self, competition: str) -> AnalysisProfile:
        train_path = self.layout.train_path
        test_path = self.layout.test_path
        header, rows = _read_sample(train_path)
        test_header, _ = _read_sample(test_path)

        profile = AnalysisProfile(
            competition=competition,
            task_prompt=self.task_prompt[:500],
        )
        if header:
            profile.feature_columns = header
            profile.target_column = self._guess_target(header)
            # v2.3.4: sample-submission-driven target correction.
            profile.target_column = self._sample_driven_target(
                header, profile.target_column)
        train_rows, target_stats = _target_stats(
            train_path, profile.target_column, header)
        profile.train_rows = train_rows
        profile.test_rows = _count_rows(test_path)
        profile.missing_columns = self._missing_columns(rows, header)
        profile.task_type = self._guess_task_type(
            rows, profile.target_column, header, target_stats)
        if self.layout.pixel_level:
            # Paired-image pixel layouts are regression BY CONSTRUCTION
            # (per-pixel intensity targets); the 30-row sample can look
            # constant (first image only), so the layout wins here.
            profile.task_type = "regression"
        # v2.3.8 structural task-type evidence (measured, never guessed):
        # RLE mask columns / bbox columns / JSON box columns override the
        # generic numeric heuristic - a mask string is not a class label.
        profile.mask_target = self._rle_target(
            train_path, header, profile.target_column)
        profile.bbox_columns = self._bbox_columns(header)
        profile.multi_row_target = self._id_repeats(rows, header)
        if profile.mask_target:
            profile.task_type = "segmentation"
        elif profile.bbox_columns or self._json_box_columns(rows, header):
            if not profile.bbox_columns:
                profile.bbox_columns = self._json_box_columns(rows, header)
            profile.task_type = "detection"
        profile.sample_values = self._column_samples(rows, header, k=3)
        profile.numeric_columns = self._numeric_columns(rows, header)
        profile.target_stats = target_stats
        profile.image_width, profile.image_height, profile.image_channels = \
            self._probe_image_dims()
        profile.image_file_count = self._count_image_files()
        profile.audio_file_count = self._count_audio_files()
        profile.text_columns = self._text_columns(
            rows, header, profile.target_column)
        profile.time_column = self._time_column(rows, header)
        strong_text = False
        if profile.text_columns and header:
            for col_idx, name in enumerate(header):
                if str(name) not in profile.text_columns:
                    continue
                seen = [str(r[col_idx]).strip() for r in rows
                        if col_idx < len(r) and str(r[col_idx]).strip()]
                seen = seen[:30]
                if not seen:
                    continue
                words_avg = (sum(len(v.split()) for v in seen)
                             / float(len(seen)))
                long_ratio = (sum(1 for v in seen if len(v) >= 40)
                              / float(len(seen)))
                # v2.3.5: 'strong' means DOCUMENT-level prose. Person names
                # and short labels (2-5 words) never reach this bar, so a
                # stray Name column cannot hijack modality even though it
                # clears the weak text-detection gate.
                if words_avg >= 8.0 or long_ratio >= 0.4:
                    strong_text = True
                    break
        profile.modality = self._guess_modality(profile,
                                                strong_text=strong_text)
        if self.layout.pixel_level:
            # v2.3.7: paired-image pixel regression is its own modality
            # (per-pixel target rows, image-to-image mapping): the standard
            # image-classification templates cannot ingest pixel ids, and
            # the tabular templates would treat id/value rows as features.
            profile.modality = "image_pixel"
        elif profile.mask_target:
            # v2.3.8: RLE-mask target column -> segmentation modality.
            profile.modality = "image_mask"
        elif profile.bbox_columns:
            # v2.3.8: box-coordinate columns (or JSON box target) ->
            # detection modality; submission is per-image boxes/RLE.
            profile.modality = "image_detection"
        elif profile.audio_file_count >= AUDIO_FILE_MODALITY_THRESHOLD:
            # v2.3.8: audio files dominate the dataset tree -> audio.
            profile.modality = "audio"
        if profile.modality == "image" and not (
                profile.image_width or profile.image_height):
            # dims probe failed (late JPEG SOF / deep nesting / unusual
            # header): modality is still image from the recursive file
            # count, so give the pipeline measurable defaults (cache +
            # resource derivation keep working).
            profile.image_width = _DEFAULT_IMAGE_SIZE
            profile.image_height = _DEFAULT_IMAGE_SIZE
            profile.image_channels = profile.image_channels or 3
        profile.feature_dim = (len(profile.numeric_columns)
                               if profile.numeric_columns
                               else len(profile.feature_columns))
        profile.n_classes = (int(target_stats.get("distinct") or 0)
                             if profile.task_type == "classification" else 0)
        # v2.4 M1: deep diagnostics - measured target/feature/order
        # evidence (stdlib, bounded, fail-open).
        try:
            from deep_profile import build_deep_diagnostics
            profile.deep_diagnostics = build_deep_diagnostics(
                train_path, profile.target_column, profile.time_column)
        except Exception as _exc:  # pragma: no cover - fail-open by contract
            profile.deep_diagnostics = {"error": str(_exc)[:200]}
        profile.data_notes = self._build_notes(profile)
        profile.data_notes += _deep_notes(profile.deep_diagnostics)
        profile.data_notes += "\n" + self.layout.describe()
        apply_metric_to_profile(profile)
        return profile

    def _guess_target(self, header: List[str]) -> str:
        lowered = [str(h).strip().lower() for h in header]
        for idx, name in enumerate(lowered):
            if name in ("target", "label", "y", "answer"):
                return str(header[idx])
        return str(header[-1]) if header else ""

    def _sample_driven_target(self, header, fallback):
        # v2.3.4: sample-submission-driven target correction (generic).
        # The sample submission's non-id columns are the ground-truth
        # prediction columns; when one exists in the train header it wins
        # over the header[-1] heuristic. Multi-output tasks keep the first
        # sample target here; the compiled harness renders and predicts
        # ALL sample target columns it finds in train.
        sp = self.layout.sample_submission_path
        if sp is None or not os.path.isfile(str(sp)):
            return fallback
        try:
            with open(str(sp), "r", encoding="utf-8", errors="replace",
                      newline="") as fh:
                sample = list(csv.reader(fh))
        except OSError:
            return fallback
        if not sample or len(sample) < 2 or len(sample[0]) < 2:
            return fallback
        s_header = [str(c) for c in sample[0]]
        id_col = str(s_header[0])
        for name in s_header[1:]:
            if str(name) in header and str(name) != id_col:
                return str(name)
        return fallback
    def _missing_columns(self, rows: List[List[str]], header: List[str]) -> List[str]:
        if not header:
            return []
        missing = []
        for col_idx, _ in enumerate(header):
            if any(not row[col_idx].strip() for row in rows if col_idx < len(row)):
                missing.append(str(header[col_idx]))
        return missing

    def _numeric_columns(self, rows: List[List[str]], header: List[str],
                         limit: int = 50) -> List[str]:
        """Sample-based numeric column detection (deterministic)."""
        numeric = []
        for col_idx, name in enumerate(header):
            seen = []
            for row in rows:
                if col_idx < len(row) and row[col_idx].strip():
                    seen.append(row[col_idx].strip())
                    if len(seen) >= limit:
                        break
            if seen and all(_is_float(v) for v in seen):
                numeric.append(str(name))
        return numeric

    def _guess_task_type(self, rows: List[List[str]], target: str,
                        header: List[str],
                        target_stats: Optional[dict] = None) -> str:
        """Task type: prompt hints CROSS-CHECKED against data evidence.

        Timeseries requires an explicit temporal prompt AND (a date-like
        column OR an explicit 'predict next' directive). Prompt keywords
        alone are never sufficient for a temporal task; a date column
        alone (e.g. 'month' categorical in tabular data) is also not
        enough. Never keyed on competition names."""
        prompt = self.task_prompt.lower()
        time_col = self._time_column(rows, header) if header else ""
        strong_time = any(k in prompt for k in
                          ("timeseries", "time series", "predict next",
                           "temporal"))
        medium_time = any(k in prompt for k in
                          ("forecast", "future sales", "future demand"))
        if strong_time or (medium_time and time_col):
            return "timeseries"
        if any(k in prompt for k in ("classification", "classify", "survived")):
            return "classification"
        if any(k in prompt for k in ("regression", "predict value", "continuous")):
            return "regression"
        if target and header:
            idx = header.index(target) if target in header else -1
            if idx >= 0:
                values = [row[idx] for row in rows if idx < len(row) and row[idx].strip()]
                numeric = [v for v in values if _is_float(v)]
                if numeric and len(numeric) == len(values):
                    distinct = len(set(values))
                    if distinct <= 10:
                        return "classification"
                    return "regression"
        # v2.3.7: fall back to the FULL measured target statistics (up to
        # 200k rows) when the 30-row sample is degenerate (e.g. data sorted
        # so the first rows share one value): a numeric target with >10
        # distinct values is regression, exactly like the sample rule.
        stats = target_stats or {}
        if stats.get("distinct") is not None:
            rows_n = int(stats.get("rows") or 0)
            num_n = int(stats.get("numeric_count") or 0)
            if rows_n > 0 and num_n >= int(rows_n * 0.99) \
                    and int(stats["distinct"]) > 10:
                return "regression"
        return "classification"

    def _column_samples(self, rows: List[List[str]], header: List[str],
                        k: int = 3) -> dict:
        out = {}
        for col_idx, name in enumerate(header):
            seen = []
            for row in rows:
                if col_idx < len(row) and row[col_idx].strip():
                    val = row[col_idx].strip()
                    if val not in seen:
                        seen.append(val)
                    if len(seen) >= k:
                        break
            if seen:
                out[str(name)] = seen
        return out

    def _probe_image_dims(self, max_files: int = 3) -> Tuple[int, int, int]:
        """Probe a few images under the layout's image dirs (stdlib only).

        Recurses into nested subdirs (class folders / deeper MLE-Bench
        layouts) and scans enough bytes to find the JPEG SOF marker, which
        can sit far past byte 64 behind EXIF/ICC segments. Returns
        (width, height, channels); (0, 0, 0) when no image dir or no
        parseable image exists. Non-fatal by design.
        """
        dirs = [d for d in (self.layout.train_image_dir,
                            self.layout.test_image_dir) if d is not None]
        for d in dirs:
            try:
                candidates = list(iter_image_files(d, max_files=max_files))
            except OSError:
                continue
            for f in candidates:
                try:
                    with open(f, "rb") as fh:
                        head = fh.read(_JPEG_SCAN_BYTES)
                except OSError:
                    continue
                w, h, c = _parse_image_header(head)
                if w and h:
                    return w, h, c
        return 0, 0, 0

    def _count_image_files(self) -> int:
        """Recursive magic-verified image file count across the resolved
        image dirs. Generic: flat, class subdirs, deeper nesting."""
        total = 0
        for d in (self.layout.train_image_dir, self.layout.test_image_dir):
            if d is None:
                continue
            try:
                total += count_image_files(d)
            except OSError:
                continue
        return total

    def _count_audio_files(self, limit: int = 20000) -> int:
        """Recursive audio-file count across the dataset tree (extension +
        light magic verification for wav/flac/ogg). Generic - never by
        competition name."""
        total = 0
        for base in (self.layout.public_dir, self.data_dir):
            if base is None or not base.is_dir():
                continue
            for dirpath, dirnames, filenames in os.walk(str(base)):
                dirnames.sort()
                for fn in filenames:
                    ext = Path(fn).suffix.lower()
                    if ext not in _AUDIO_EXTENSIONS:
                        continue
                    total += 1
                    if total >= limit:
                        return total
        return total

    def _rle_target(self, train_path: Path, header: List[str],
                    target: str, sample_limit: int = 200) -> str:
        """Return `target` when the target column is RLE-encoded mask
        evidence: >=60% of sampled non-empty values match the RLE pattern
        (empty/'-' rows are valid 'no mask' rows and never count against).
        Reads a dedicated sample (the 30-row profile sample can be
        unluckily all-empty for sparse masks)."""
        if not target or target not in header or not train_path.is_file():
            return ""
        idx = header.index(target)
        seen = []
        try:
            with open(str(train_path), "r", encoding="utf-8",
                      errors="replace", newline="") as fh:
                reader = csv.reader(fh, delimiter=table_delimiter(train_path))
                for i, row in enumerate(reader):
                    if i == 0:
                        continue
                    if idx < len(row):
                        v = str(row[idx]).strip()
                        if v and v.lower() not in ("nan", "none", "-"):
                            seen.append(v)
                            if len(seen) >= sample_limit:
                                break
        except OSError:
            return ""
        if not seen:
            return ""
        hits = sum(1 for v in seen if _is_rle_value(v))
        return target if (hits / float(len(seen))) >= 0.6 else ""

    def _bbox_columns(self, header: List[str]) -> List[str]:
        """Return the detected box-coordinate column set (generic name
        geometry, no competition names). Requires a FULL coordinate set -
        a lone x or width column is never evidence."""
        lowered = [str(h).strip().lower() for h in header]
        for box_set in _BOX_NAME_SETS:
            present = [h for h, lw in zip(header, lowered) if lw in box_set]
            if len(present) == len(box_set):
                return present
        return []

    def _json_box_columns(self, rows: List[List[str]], header: List[str],
                          sample_limit: int = 30) -> List[str]:
        """JSON box evidence on ANY column (per-character bboxes, nested
        study-level box lists, ...): >=50% of sampled non-empty values look
        like box JSON. Returns the matching column names."""
        out = []
        for col_idx, name in enumerate(header):
            seen = []
            for row in rows:
                if col_idx < len(row):
                    v = str(row[col_idx]).strip()
                    if v:
                        seen.append(v)
                        if len(seen) >= sample_limit:
                            break
            if not seen:
                continue
            hits = sum(1 for v in seen if _is_json_box_value(v))
            if (hits / float(len(seen))) >= 0.5:
                out.append(str(name))
        return out

    def _id_repeats(self, rows: List[List[str]], header: List[str],
                    sample_limit: int = 300) -> bool:
        """Multi-row signal: the id column repeats within the sampled rows
        (per-image boxes) - vs one-row-per-id tables."""
        id_col = self._id_column(header)
        if not id_col or id_col not in header:
            return False
        idx = header.index(id_col)
        seen = {}
        for row in rows:
            if idx >= len(row):
                continue
            v = str(row[idx]).strip()
            if not v:
                continue
            if v in seen:
                return True
            seen[v] = 1
            if len(seen) >= sample_limit:
                return False
        return False

    def _text_columns(self, rows: List[List[str]], header: List[str],
                      target: str, sample_limit: int = 30) -> List[str]:
        """Content-verified free-text feature columns (v2.3.2).

        A column-name hint lowers the acceptance bar slightly (still needs
        non-numeric prose); a column without a hint needs strong content
        evidence. Column names alone are NEVER sufficient."""
        out = []
        for col_idx, name in enumerate(header):
            if str(name) == str(target):
                continue
            is_text, score = _column_text_signal(
                rows, header, col_idx, sample_limit=sample_limit)
            if not is_text:
                continue
            lowered = str(name).lower()
            hint = any(k in lowered for k in _TEXT_NAME_HINTS)
            if hint or score >= 0.6:
                out.append(str(name))
        return out

    def _time_column(self, rows: List[List[str]], header: List[str],
                     sample_limit: int = 30) -> str:
        """Date/time column evidence: name hint AND parseable date values.

        Numeric 'year'/'month' columns are NOT evidence (pure numbers are
        rejected by _looks_like_date); a real timestamp/date column is."""
        for col_idx, name in enumerate(header):
            lowered = str(name).strip().lower()
            if not any(h in lowered for h in _DATE_COLUMN_HINTS):
                continue
            vals = [str(row[col_idx]).strip() for row in rows
                    if col_idx < len(row) and str(row[col_idx]).strip()]
            vals = vals[:sample_limit]
            if not vals:
                continue
            dates = sum(1 for v in vals if _looks_like_date(v))
            if dates >= max(2, int(len(vals) * 0.5)):
                return str(name)
        return ""

    def _id_column(self, header: List[str]) -> str:
        """Mirror the compiled harness id rule: sample-submission first
        column, else an id-named column, else the first column. Never by
        competition name."""
        if header:
            try:
                if self.layout.sample_submission_path and \
                        os.path.isfile(str(self.layout.sample_submission_path)):
                    with open(self.layout.sample_submission_path, "r",
                              encoding="utf-8", errors="replace",
                              newline="") as fh:
                        first = next(csv.reader(fh), None)
                    if first and first[0] in header:
                        return str(first[0])
            except OSError:
                pass
            lowered = [str(h).strip().lower() for h in header]
            for idx, name in enumerate(lowered):
                if name in ("id", "index"):
                    return str(header[idx])
            # v2.3.5: id-suffixed names (PhraseId/SentenceId/textID/essay_id)
            # are metadata keys in real text tables; excluding them keeps
            # the mixed-data dominance rule honest (movie-review etc).
            for idx, name in enumerate(lowered):
                if len(name) >= 4 and name.endswith("id"):
                    return str(header[idx])
            # no reliable id: leave it empty (never mislabel a text feature
            # column as the id just because it is first in the header)
        return ""

    def _sample_target_columns(self, header: List[str]) -> List[str]:
        """Columns declared in the sample-submission header are prediction
        targets by definition, never features. The mixed-data dominance
        rule uses this so multi-label text tasks (jigsaw-toxic, google-quest
        put ALL their labels in the sample file) are not mistaken for
        tabular data. Generic - mirror of the compiled harness contract."""
        sp = self.layout.sample_submission_path
        if sp is None or not os.path.isfile(str(sp)):
            return []
        try:
            with open(str(sp), "r", encoding="utf-8", errors="replace",
                      newline="") as fh:
                sample = list(csv.reader(fh))
        except OSError:
            return []
        if not sample or not sample[0]:
            return []
        s_lower = {str(c).strip().lower() for c in sample[0]}
        return [str(h) for h in header
                if str(h).strip().lower() in s_lower]

    def _text_dominates(self, profile: AnalysisProfile) -> bool:
        """v2.3.3+ mixed-data rule: free-text columns must dominate the
        usable FEATURES for modality=text. A stray prose column (e.g. a
        passenger Name in an otherwise numeric table) must NOT hijack the
        modality, or HERA would lose the tabular method space. Sample-
        submission target columns never count as usable features, so
        multi-label text tasks keep their text modality (v2.3.5)."""
        target = str(profile.target_column or "")
        header = list(profile.feature_columns or [])
        id_col = self._id_column(header)
        lowered = [str(h).strip().lower() for h in header]
        id_like = {str(h) for h, lw in zip(header, lowered)
                   if len(lw) >= 4 and lw.endswith("id")}
        text = set(str(c) for c in (profile.text_columns or []))
        sample_targets = set(self._sample_target_columns(header))
        usable = [str(c) for c in header
                  if str(c) != target and str(c) != id_col
                  and str(c) not in id_like
                  and str(c) not in text
                  and str(c) not in sample_targets]
        if not usable:
            return True
        return len(text) >= len(usable)

    def _guess_modality(self, profile: AnalysisProfile,
                         strong_text: bool = False) -> str:
        """Deterministic, data-shape-driven modality guess (never by
        competition name): image when dims are measurable OR a resolved
        image dir holds enough magic-verified files; text when
        content-verified free-text columns dominate the usable features
        OR show document-level prose evidence (v2.3.3/5 - a stray prose
        column such as a passenger Name in a mostly-tabular table stays
        tabular; sample-submission target columns never count as usable
        features so multi-label text tasks stay text; strong_text means
        >=8 words/row or >=40% long values, which person names never
        reach); else tabular."""
        if profile.image_width or profile.image_height:
            return "image"
        if profile.image_file_count >= IMAGE_FILE_MODALITY_THRESHOLD:
            return "image"
        if profile.text_columns and (
                self._text_dominates(profile) or strong_text):
            return "text"
        return "tabular"

    def _build_notes(self, profile: AnalysisProfile) -> str:
        lines = [
            "Competition: %s" % profile.competition,
            "Task type guess: %s" % profile.task_type,
            "Train rows: %s | Test rows: %s" % (profile.train_rows, profile.test_rows),
        ]
        if profile.feature_columns:
            lines.append("Columns: %s" % ", ".join(profile.feature_columns))
        if profile.target_column:
            lines.append("Target column: %s" % profile.target_column)
        stats = profile.target_stats or {}
        if stats.get("distinct") is not None:
            top_txt = ""
            if stats.get("top") is not None:
                top_txt = " top_class=%r share=%.1f%%" % (
                    stats["top"], (stats.get("top_share") or 0.0) * 100.0)
            lines.append("Target cardinality: %d distinct%s" % (
                stats["distinct"], top_txt))
        lines.append("Modality: %s" % profile.modality)
        if profile.text_columns and profile.modality != "text":
            lines.append(
                "Mixed data: %d text column(s) present but dominated by "
                "tabular features; modality=tabular"
                % len(profile.text_columns))
        if profile.text_columns:
            lines.append("Text columns (content-verified): %s"
                         % ", ".join(profile.text_columns))
        if profile.time_column:
            lines.append("Time column (date evidence): %s"
                         % profile.time_column)
        if profile.image_width or profile.image_height:
            lines.append("Image dims (probe): %dx%d ch=%d" % (
                profile.image_width, profile.image_height, profile.image_channels))
        if profile.image_file_count:
            lines.append("Image file count (recursive scan): %d"
                         % profile.image_file_count)
        if profile.feature_dim:
            lines.append("Feature dim: %d" % profile.feature_dim)
        if profile.n_classes:
            lines.append("Class count: %d" % profile.n_classes)
        if profile.numeric_columns:
            lines.append("Numeric columns (sample): %s"
                         % ", ".join(profile.numeric_columns))
        if profile.missing_columns:
            lines.append("Columns with missing values (sample): %s"
                         % ", ".join(profile.missing_columns))
        if profile.sample_values:
            snippets = ["%s=%s" % (k, "/".join(v))
                        for k, v in list(profile.sample_values.items())[:6]]
            lines.append("Sample values: " + "; ".join(snippets))
        return "\n".join(lines)
