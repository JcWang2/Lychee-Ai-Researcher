# -*- coding: utf-8 -*-
"""Deterministic data-layout resolution for flat and MLE-bench datasets."""
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple


class DatasetLayoutError(RuntimeError):
    """Raised when a supported train/test dataset layout cannot be found."""


@dataclass(frozen=True)
class DatasetLayout:
    root: Path
    train_path: Path
    test_path: Path
    sample_submission_path: Optional[Path]
    public_dir: Path
    private_dir: Path
    train_image_dir: Optional[Path]
    test_image_dir: Optional[Path]
    layout_name: str
    gold_test_csv: Optional[Path] = None
    test_has_labels: bool = False
    paired_target_dir: Optional[Path] = None
    pixel_level: bool = False

    def describe(self) -> str:
        lines = [
            "layout=%s" % self.layout_name,
            "dataset_root=%s" % self.root,
            "train_csv=%s" % self.train_path,
            "test_csv=%s" % self.test_path,
        ]
        if self.sample_submission_path:
            lines.append("sample_submission=%s" % self.sample_submission_path)
        if self.train_image_dir:
            lines.append("train_images=%s" % self.train_image_dir)
        if self.test_image_dir:
            lines.append("test_images=%s" % self.test_image_dir)
        if self.paired_target_dir:
            lines.append("paired_target_images=%s" % self.paired_target_dir)
        if self.pixel_level:
            lines.append("pixel_level=1 (image-to-image regression: one "
                         "row per pixel, target = intensity/255)")
        return "\n".join(lines)

    def prompt_paths(self) -> str:
        lines = [
            "Resolved dataset layout: %s" % self.layout_name,
            "Train CSV: %s" % self.train_path,
            "Test CSV: %s" % self.test_path,
        ]
        if self.sample_submission_path:
            lines.append("Sample submission: %s" % self.sample_submission_path)
        if self.train_image_dir:
            lines.append("Train image directory: %s" % self.train_image_dir)
        if self.test_image_dir:
            lines.append("Test image directory: %s" % self.test_image_dir)
        if self.gold_test_csv:
            lines.append("GOLD test CSV (labels, scoring only - NEVER read it): %s"
                         % self.gold_test_csv)
        if self.test_has_labels:
            lines.append("WARNING: resolved test CSV contains labels; use ONLY its "
                         "id column for submission ids, never the target column")
        if self.paired_target_dir:
            lines.append("Paired target image directory (training labels live "
                         "as pixels): %s" % self.paired_target_dir)
        if self.pixel_level:
            lines.append("Pixel-level regression layout: train.csv rows are "
                         "pixels '<stem>_<row>_<col>' with target "
                         "intensity/255; submission must follow the same "
                         "per-pixel id format")
        return "\n".join(lines)

    def manifest(self, train_rows: int = 0, test_rows: int = 0,
                 target_column: str = "", task_type: str = "",
                 sample_header: Optional[list] = None) -> dict:
        return {
            "layout": self.layout_name,
            "dataset_root": str(self.root),
            "public_dir": str(self.public_dir),
            "private_dir": str(self.private_dir),
            "train_csv": str(self.train_path),
            "test_csv": str(self.test_path),
            "sample_submission": (str(self.sample_submission_path)
                                  if self.sample_submission_path else ""),
            "train_images": (str(self.train_image_dir)
                             if self.train_image_dir else ""),
            "test_images": (str(self.test_image_dir)
                            if self.test_image_dir else ""),
            "gold_test_csv": (str(self.gold_test_csv)
                              if self.gold_test_csv else ""),
            "test_has_labels": bool(self.test_has_labels),
            "paired_target_dir": (str(self.paired_target_dir)
                                  if self.paired_target_dir else ""),
            "pixel_level": bool(self.pixel_level),
            "train_rows": int(train_rows or 0),
            "test_rows": int(test_rows or 0),
            "target_column": target_column or "",
            "task_type": task_type or "",
            "sample_submission_header": list(sample_header or []),
        }


def _first_file(directory: Path, names) -> Optional[Path]:
    for name in names:
        path = directory / name
        if path.is_file():
            return path
    return None


def _first_dir(directory: Path, names) -> Optional[Path]:
    for name in names:
        path = directory / name
        if path.is_dir():
            return path
    return None


# MLE-Bench / Kaggle image dir naming is NOT standardized: aerial uses
# train/, aptos uses train_images/, some tasks use images_train/ or nest
# one level deep (train_images/train_2019/). Generic aliases only - never
# competition names.
_TRAIN_IMAGE_DIR_NAMES = ("train", "train_images", "images_train",
                          "train_image", "train_imgs", "images")
_TEST_IMAGE_DIR_NAMES = ("test", "test_images", "images_test",
                         "test_image", "test_imgs")

# Paired image-to-image layouts (denoising/restoration-style): the train
# input dir has a sibling dir holding the TARGET images with matching
# stems (e.g. train_cleaned next to train). Generic aliases only - never
# competition names. Detection additionally requires a per-pixel sample
# submission, so a stray sibling dir can never hijack a normal layout.
_TRAIN_TARGET_IMAGE_DIR_NAMES = (
    "train_cleaned", "train_clean", "train_clean_images",
    "train_targets", "train_gt", "train_ground_truth",
    "train_hr", "train_highres", "train_original")

# Train tables MLE-Bench actually ships (csv + tsv variants). Generic, never
# competition-specific.
_TRAIN_FILE_NAMES = ("train.csv", "labels.csv", "train_labels.csv",
                     "training.csv", "train.tsv", "labels.tsv",
                     "train_labels.tsv", "training.tsv",
                     # v2.3.8: per-image / per-study train tables (siim-covid19
                     # style detection): image-level first so box columns win.
                     "train_image_level.csv", "train_study_level.csv")

# Sample-submission naming is NOT standardized across MLE-Bench: most use
# sample_submission.csv, movie-review uses sampleSubmission.csv, random-acts
# writes sampleSubmission.csv, text-normalization uses localized prefixes.
_SAMPLE_FILE_NAMES = ("sample_submission.csv", "sampleSubmission.csv",
                      "sample-submission.csv", "samplesubmission.csv",
                      "kaggle_sample_submission.csv",
                      "sample_submission_null.csv")

# Private (gold) test tables: test.csv is the majority; image competitions
# (aerial/histopathologic/plant-seedlings/dogs-vs-cats) ship answers.csv or
# gold_submission.csv instead.
_PRIVATE_TEST_FILE_NAMES = ("test.csv", "answers.csv",
                            "gold_submission.csv", "test.tsv", "answers.tsv")


def table_delimiter(path) -> str:
    """Delimiter for a table file: '\t' for .tsv, ',' for .csv, else a
    first-line sniff. Generic - analyzer and sanitizer share this rule."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".tsv":
        return "\t"
    if suffix == ".csv":
        return ","
    try:
        with open(p, "r", encoding="utf-8", errors="replace",
                  newline="") as fh:
            line = fh.readline()
    except OSError:
        return ","
    if "\t" in line and "," not in line:
        return "\t"
    return ","



def _first_image_dir(directory: Path, names) -> Optional[Path]:
    """First existing image dir, handling naming aliases. Returns the
    NAMED dir itself (train/, train_images/, ...) whenever it holds image
    files directly OR inside subdirs (class folders, one-level nesting).
    Downstream consumers recurse (count, dims probe, cache index), so a
    class-folder layout must resolve to the whole dir - never to a single
    class subdir. Empty dirs are skipped."""
    for name in names:
        path = directory / name
        if not path.is_dir():
            continue
        try:
            entries = list(path.iterdir())
        except OSError:
            return path  # unreadable: keep dir; caller degrades
        if not entries:
            continue
        if any(e.is_file() for e in entries):
            return path
        for e in entries:
            if not e.is_dir():
                continue
            try:
                if any(f.is_file() for f in e.iterdir()):
                    return path
            except OSError:
                return path
    return None


# Magic-number image detection (stdlib only, no extension assumptions).
# Covers PNG/JPEG/GIF/BMP/TIFF/WebP - the formats MLE-Bench image tasks
# actually use. Generic - never competition-specific.
_IMAGE_MAGIC_PREFIXES = (
    b"\x89PNG\r\n\x1a\n",   # PNG
    b"\xff\xd8",             # JPEG
    b"GIF87a",               # GIF
    b"GIF89a",               # GIF
    b"BM",                   # BMP
    b"II*\x00",              # TIFF (little-endian)
    b"MM\x00*",              # TIFF (big-endian)
)


def _is_image_file(path: Path) -> bool:
    """True when the file's leading bytes match a known image magic number."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(16)
    except OSError:
        return False
    if not head:
        return False
    if any(head.startswith(m) for m in _IMAGE_MAGIC_PREFIXES):
        return True
    # WebP: RIFF container with a WEBP chunk id at offset 8.
    return head[:4] == b"RIFF" and head[8:12] == b"WEBP"


def iter_image_files(directory, max_files: int = 100000, max_depth: int = 8,
                     per_dir_scan: int = 1024) -> Iterable[Path]:
    """Depth-first, bounded walk yielding magic-verified image file paths.

    Generic for every MLE-Bench layout: flat dirs (train/*.jpg), class
    subdirs (train/breed/*.jpg), or deeper nesting. Deterministic (sorted
    entries) and bounded (max_files total, max_depth levels, per_dir_scan
    entries per directory) so pathological trees cannot hang analysis.
    """
    root = Path(directory)
    if not root.is_dir():
        return
    count = 0

    def walk(d: Path, depth: int):
        nonlocal count
        if count >= max_files or depth > max_depth:
            return
        try:
            entries = sorted(d.iterdir(), key=lambda p: p.name)
        except OSError:
            return
        for entry in entries[:per_dir_scan]:
            if count >= max_files:
                return
            try:
                if entry.is_dir():
                    yield from walk(entry, depth + 1)
                elif entry.is_file() and _is_image_file(entry):
                    count += 1
                    yield entry
            except OSError:
                continue

    yield from walk(root, 0)


def count_image_files(directory, **kwargs) -> int:
    """Count magic-verified image files under directory (recursive, bounded)."""
    return sum(1 for _ in iter_image_files(directory, **kwargs))


def _png_dims(path: Path):
    """(width, height) from a PNG header (stdlib only) or None."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(33)
    except OSError:
        return None
    if len(head) < 33 or head[:8] != b"\x89PNG\r\n\x1a\n" \
            or head[12:16] != b"IHDR":
        return None
    w = int.from_bytes(head[16:20], "big")
    h = int.from_bytes(head[20:24], "big")
    if w <= 0 or h <= 0 or w * h > 100_000_000:
        return None
    return w, h


def _decode_png_gray(path: Path):
    """Decode a PNG to flat grayscale-luma samples (0..255) via stdlib.

    Supports color types 0 (gray), 2 (RGB), 4 (gray+alpha), 6 (RGBA) and
    bit depths 8/16; luma for color images uses the same ITU-R 601-2
    weights as PIL's Image.convert("L"). All PNG filter types (None/Sub/
    Up/Average/Paeth) are applied. Returns a list of floats (h*w) or None
    when the file is not a decodable PNG. Generic: pixel-level targets in
    MLE-Bench are PNG (lossless); non-PNG paired layouts are skipped with
    a clear report reason instead of a crash.
    """
    import zlib
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError:
        return None
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    pos = 8
    w = h = bit = ctype = None
    idat = b""
    while pos + 8 <= len(data):
        ln = int.from_bytes(data[pos:pos + 4], "big")
        typ = data[pos + 4:pos + 8]
        if ln < 0 or pos + 12 + ln > len(data):
            break
        chunk = data[pos + 8:pos + 8 + ln]
        if typ == b"IHDR" and len(chunk) >= 10:
            w = int.from_bytes(chunk[0:4], "big")
            h = int.from_bytes(chunk[4:8], "big")
            bit = chunk[8]
            ctype = chunk[9]
        elif typ == b"IDAT":
            idat += chunk
        elif typ == b"IEND":
            break
        pos += 12 + ln
    if not (w and h and bit and idat) or ctype not in (0, 2, 4, 6):
        return None
    if bit not in (8, 16):
        return None
    channels = {0: 1, 2: 3, 4: 2, 6: 4}[ctype]
    bpp = channels * (bit // 8)
    try:
        raw = zlib.decompress(idat)
    except zlib.error:
        return None
    stride = w * bpp
    if len(raw) < h * (stride + 1):
        return None
    out = []
    prev = [0] * stride
    pos = 0
    for _y in range(h):
        f = raw[pos]
        pos += 1
        line = list(raw[pos:pos + stride])
        pos += stride
        if len(line) < stride:
            return None
        if f == 1:
            for i in range(bpp, stride):
                line[i] = (line[i] + line[i - bpp]) & 255
        elif f == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 255
        elif f == 3:
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 255
        elif f == 4:
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                b = prev[i]
                c = prev[i - bpp] if i >= bpp else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pr) & 255
        elif f != 0:
            return None
        if bit == 8:
            if ctype == 0:
                out.extend(line)
            elif ctype == 2:
                for i in range(0, stride, 3):
                    out.append(0.299 * line[i] + 0.587 * line[i + 1]
                               + 0.114 * line[i + 2])
            elif ctype == 4:
                for i in range(0, stride, 2):
                    out.append(float(line[i]))
            else:  # 6
                for i in range(0, stride, 4):
                    out.append(0.299 * line[i] + 0.587 * line[i + 1]
                               + 0.114 * line[i + 2])
        else:  # 16-bit: keep the high byte (value / 255 semantics)
            if ctype == 0:
                for i in range(0, stride, 2):
                    out.append(float(line[i]))
            elif ctype == 2:
                for i in range(0, stride, 6):
                    out.append(0.299 * line[i] + 0.587 * line[i + 2]
                               + 0.114 * line[i + 4])
            elif ctype == 4:
                for i in range(0, stride, 4):
                    out.append(float(line[i]))
            else:  # 6
                for i in range(0, stride, 8):
                    out.append(0.299 * line[i] + 0.587 * line[i + 2]
                               + 0.114 * line[i + 4])
        prev = line
    if len(out) != w * h:
        return None
    return out


def _per_pixel_sample(sample_path):
    """Detect a per-pixel (image-to-image regression) sample submission.

    Generic structural evidence - never competition names: the sample
    table has exactly two columns; >=50% of the first 200 ids match the
    '<stem>_<row>_<col>' pixel pattern (1-based row/col) AND >=90% of the
    target values parse as finite floats. Returns (id_col, value_col) or
    None.
    """
    import csv as _csv
    import re as _re
    if sample_path is None or not Path(sample_path).is_file():
        return None
    try:
        with open(sample_path, "r", encoding="utf-8", errors="replace",
                  newline="") as fh:
            rows = list(_csv.reader(fh))[:201]
    except OSError:
        return None
    if len(rows) < 2 or len(rows[0]) != 2:
        return None
    id_col, value_col = str(rows[0][0]), str(rows[0][1])
    if not id_col.strip() or not value_col.strip():
        return None
    pat = _re.compile(r"^.+_\d+_\d+$")
    n_pixel = 0
    n_num = 0
    total = 0
    for row in rows[1:]:
        if len(row) < 2:
            continue
        total += 1
        if pat.match(str(row[0]).strip()):
            n_pixel += 1
        try:
            float(str(row[1]).strip())
            n_num += 1
        except (TypeError, ValueError):
            pass
    if total < 5 or n_pixel / float(total) < 0.5:
        return None
    if n_num / float(total) < 0.9:
        return None
    return id_col, value_col


def _paired_target_dir(public_dir: Path, train_image_dir: Optional[Path],
                       test_image_dir: Optional[Path]) -> Optional[Path]:
    """Find the sibling target-image dir of a paired image layout.

    Generic paired-input signature (input -> target mapping such as
    denoising/restoration): a direct child of public_dir that is NOT a
    standard train/test input alias, holds magic-verified images whose
    stems overlap the train input stems on >=10 files and >=50% of its
    own files. Deterministic (sorted); returns the best match or None.
    """
    if train_image_dir is None or not train_image_dir.is_dir():
        return None
    try:
        train_stems = {p.stem for p in iter_image_files(
            train_image_dir, max_files=10000)}
    except OSError:
        return None
    if len(train_stems) < 10:
        return None
    excluded = set(_TRAIN_IMAGE_DIR_NAMES) | set(_TEST_IMAGE_DIR_NAMES)
    if test_image_dir is not None:
        excluded.add(test_image_dir.name)
    excluded.add(train_image_dir.name)
    try:
        entries = sorted(public_dir.iterdir(), key=lambda p: p.name)
    except OSError:
        return None
    best = None
    best_score = 0.0
    for cand in entries:
        if not cand.is_dir() or cand.name in excluded:
            continue
        name_hint = cand.name in _TRAIN_TARGET_IMAGE_DIR_NAMES
        try:
            cand_stems = {p.stem for p in iter_image_files(
                cand, max_files=10000)}
        except OSError:
            continue
        if len(cand_stems) < 10:
            continue
        matched = len(train_stems & cand_stems)
        if matched < 10:
            continue
        score = matched / float(len(cand_stems))
        # A generic alias name (train_cleaned/train_targets/...) lowers the
        # overlap bar (the target dir may legitimately hold more files than
        # the train input); structural overlap still decides the winner.
        if score >= 0.5 or (name_hint and score >= 0.25):
            if score > best_score:
                best = cand
                best_score = score
    return best


def _synthesize_paired_rows(train_dir: Path, target_dir: Path, pixel,
                            cap: int):
    """Deterministic pixel rows for a paired-image layout.

    One row per sampled pixel: id '<stem>_<row>_<col>' (1-based, matching
    the official per-pixel sample format), value = grayscale intensity /
    255 (0..1). When the total pixel count exceeds `cap`, a deterministic
    stride keeps every image covered while bounding the CSV size. Target
    images must be decodable PNGs (lossless pixel targets); anything else
    degrades to a skipped report, never a crash. Returns
    (rows, images, stride, mode).
    """
    target_files = {}
    for p in iter_image_files(target_dir, max_files=10000):
        target_files.setdefault(p.stem, p)
    train_stems = set()
    for p in iter_image_files(train_dir, max_files=10000):
        train_stems.add(p.stem)
    dims = []
    total = 0
    for stem in sorted(train_stems & set(target_files)):
        d = _png_dims(target_files[stem])
        if d is None:
            continue
        dims.append((stem, target_files[stem], d[0], d[1]))
        total += d[0] * d[1]
    if not dims:
        return [], 0, 1, ""
    stride = 1
    if total > cap:
        stride = (total + cap - 1) // cap
    rows = []
    for stem, tpath, w, h in dims:
        gray = _decode_png_gray(tpath)
        if gray is None or len(gray) != w * h:
            continue
        for i in range(0, len(gray), stride):
            r, c = divmod(i, w)
            rows.append(["%s_%d_%d" % (stem, r + 1, c + 1),
                         "%.6f" % (float(gray[i]) / 255.0)])
    return rows, len(dims), stride, "paired-image"


def _candidate_pairs(root: Path):
    if root.name == "public" and root.is_dir():
        yield root, root.parent / "private", "mlebench_prepared"
    if root.name == "private" and root.is_dir():
        yield root.parent / "public", root, "mlebench_prepared"
    yield root, root, "flat"
    yield root / "prepared" / "public", root / "prepared" / "private", "mlebench_prepared"
    yield root / "public", root / "private", "mlebench_prepared"


def _resolve_sample(sample_path: Optional[str], public_dir: Path,
                    root: Path) -> Optional[Path]:
    if sample_path:
        explicit = Path(sample_path)
        if explicit.is_file():
            return explicit
    for base in (public_dir, root):
        found = _first_file(base, _SAMPLE_FILE_NAMES)
        if found is not None:
            return found
    # Generic fallback: any *sample*submission*.csv (localized prefixes,
    # future naming variants) - deterministic sorted pick.
    for base in (public_dir, root):
        try:
            hits = sorted(
                p for p in base.iterdir()
                if p.is_file() and p.suffix.lower() == ".csv"
                and "sample" in p.name.lower()
                and "submission" in p.name.lower())
        except OSError:
            continue
        if hits:
            return hits[0]
    return None


# v2.3.8: audio file extensions used by the label-free table synthesis
# (never competition-specific; content/extension driven).
_AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac",
                     ".opus", ".wma")


def _count_audio_files(directory: Path, limit: int = 20000) -> int:
    """Recursive audio-file count (extension based). Used to recognize
    audio layouts that ship no train table (labels in dir names or in the
    sample submission)."""
    total = 0
    if not directory.is_dir():
        return 0
    for dirpath, _dirnames, filenames in os.walk(str(directory)):
        for fn in filenames:
            if Path(fn).suffix.lower() in _AUDIO_EXTENSIONS:
                total += 1
                if total >= limit:
                    return total
    return total


def _synthesize_dir_label_table(public_dir: Path) -> Optional[Path]:
    """Class-subdirectory train table (labels in DIRECTORY NAMES): when the
    public tree holds `train/audio/<label>/*.wav` or `train/<label>/*.wav`,
    write public/train.csv as (fname, label) rows where fname is relative to
    public_dir. Generic; never competition-specific."""
    import csv as _csv
    train_dir = _first_dir(public_dir, ("train", "train_audio"))
    if train_dir is None:
        return None
    label_root = _first_dir(train_dir, ("audio",)) or train_dir
    try:
        entries = sorted(label_root.iterdir(), key=lambda p: p.name)
    except OSError:
        return None
    rows = []
    for e in entries:
        if not e.is_dir():
            continue
        label = e.name
        try:
            wavs = sorted(
                f for f in e.iterdir()
                if f.is_file() and f.suffix.lower() in _AUDIO_EXTENSIONS)
        except OSError:
            continue
        for w in wavs:
            rows.append((w.relative_to(public_dir).as_posix(), label))
    if not rows:
        return None
    out = public_dir / "train.csv"
    if not out.is_file():
        with open(out, "w", newline="", encoding="utf-8") as fh:
            _csv.writer(fh).writerows([("fname", "label")] + rows)
    return out


def _synthesize_sample_train(public_dir: Path, sample: Path) -> Optional[Path]:
    """Copy the sample submission as the train table when no train table
    exists (label-free image / detection / audio shapes). The placeholder target
    values let content sniffers (RLE / JSON box) classify the task; the
    deterministic baselines never train on them."""
    import csv as _csv
    try:
        with open(sample, "r", encoding="utf-8", errors="replace",
                  newline="") as fh:
            rows = list(_csv.reader(fh))
    except OSError:
        return None
    if len(rows) < 2 or len(rows[0]) < 1:
        return None
    out = public_dir / "train.csv"
    if not out.is_file():
        with open(out, "w", newline="", encoding="utf-8") as fh:
            _csv.writer(fh).writerows(rows)
    return out


def _synthesize_id_only_test(public_dir: Path, sample: Path) -> Optional[Path]:
    """Label-free id-only test table from the sample submission (used when
    the public tree ships no test table)."""
    import csv as _csv
    try:
        with open(sample, "r", encoding="utf-8", errors="replace",
                  newline="") as fh:
            rows = list(_csv.reader(fh))
    except OSError:
        return None
    if len(rows) < 2:
        return None
    out = public_dir / "test.csv"
    if not out.is_file():
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = _csv.writer(fh)
            w.writerow([rows[0][0]])
            for r in rows[1:]:
                if r:
                    w.writerow([r[0]])
    return out


def _synthesize_prefix_label_table(public_dir: Path,
                                   id_col: str = "file",
                                   label_col: str = "label") -> Optional[Path]:
    """Flat image dir whose FILE NAME prefixes carry the labels
    (cat.0.jpg / dog.1.jpg style): write public/train.csv as
    (id_col, label_col) rows where the id is the file name relative to the
    image dir. Guardrails: >= 50 flat image files, 2..64 distinct prefixes,
    and not every prefix is numeric (a plain numbered image dir must never
    trigger). Same evidence floor as the daemon-side synthesis. Generic;
    never competition-specific."""
    import csv as _csv
    train_dir = _first_image_dir(public_dir, _TRAIN_IMAGE_DIR_NAMES)
    if train_dir is None:
        return None
    try:
        flat = [f for f in sorted(train_dir.iterdir())
                if f.is_file() and _is_image_file(f)]
    except OSError:
        return None
    if len(flat) < 50:
        return None
    tokens = {}
    for f in flat:
        token = f.stem.split(".", 1)[0].strip() or f.stem
        tokens.setdefault(token, []).append(f)
    if not (2 <= len(tokens) <= 64):
        return None
    if all(tok.isdigit() for tok in tokens):
        return None
    rows = []
    for token in sorted(tokens):
        for f in tokens[token]:
            rows.append((f.name, token))
    if not rows:
        return None
    out = public_dir / "train.csv"
    if not out.is_file():
        with open(out, "w", newline="", encoding="utf-8") as fh:
            _csv.writer(fh).writerows([(id_col, label_col)] + rows)
    return out


def _synthesize_missing_tables(public_dir: Path, root: Path, sample: Optional[Path],
                               train_path: Optional[Path],
                               test_path: Optional[Path]):
    """v2.3.8/v2.5.5: label-free table synthesis for audio / mask /
    detection / prefix-labeled image layouts that ship no train table
    (and sometimes no test table). Order: dir labels -> filename-prefix
    labels -> placeholder sample copy (last resort). Returns
    (train_path, test_path) or None when impossible."""
    import csv as _csv
    if sample is None:
        return None
    # The synthesis belongs to the candidate layout that OWNS the sample:
    # an explicitly-passed sample under prepared/public must never trigger
    # table synthesis for the root "flat" candidate (that would hijack the
    # real mlebench_prepared layout with placeholder tables).
    try:
        sample_res = sample.resolve()
        pub_res = public_dir.resolve()
        if sample_res.parent != pub_res:
            return None
    except OSError:
        return None
    if train_path is None:
        tp = _synthesize_dir_label_table(public_dir)
        if tp is None:
            # v2.5.5: filename-prefix labels BEFORE
            # the placeholder sample copy, so image tasks train on real
            # labels instead of all-zero sample rows. Column names mirror
            # the sample submission (generic contract).
            id_col, label_col = "file", "label"
            try:
                with open(sample, "r", encoding="utf-8", errors="replace",
                          newline="") as fh:
                    s_head = next(_csv.reader(fh), [])
                if s_head:
                    id_col = str(s_head[0])
                    if len(s_head) > 1:
                        label_col = str(s_head[1])
            except OSError:
                pass
            tp = _synthesize_prefix_label_table(public_dir,
                                                id_col, label_col)
        if tp is None:
            tp = _synthesize_sample_train(public_dir, sample)
        if tp is None:
            return None
        train_path = tp
    if test_path is None:
        tp = _synthesize_id_only_test(public_dir, sample)
        if tp is None:
            return None
        test_path = tp
    return train_path, test_path


def _resolve_test_path(public_dir: Path, private_dir: Path,
                       root: Path) -> Tuple[Optional[Path], bool]:
    """Public (no labels) test first; private (labels) only as fallback."""
    for candidate in (public_dir / "test.csv",
                      public_dir / "test" / "test.csv",
                      public_dir / "test.tsv",
                      public_dir / "test" / "test.tsv"):
        if candidate.is_file():
            return candidate, False
    private_test = _first_file(private_dir, _PRIVATE_TEST_FILE_NAMES)
    if private_test is not None:
        return private_test, True
    if root != private_dir:
        root_test = root / "test.csv"
        if root_test.is_file():
            return root_test, False
    return None, False


def _localized_table_entries(entries, prefix, kind):
    hits = []
    for p in entries:
        if not p.is_file():
            continue
        name = p.name
        base = name[:-len(".zip")] if name.lower().endswith(".csv.zip") else name
        if not base.lower().endswith(".csv"):
            continue
        if base.lower().startswith(prefix.lower() + "_" + kind):
            hits.append(p)
    hits.sort(key=lambda p: p.name)
    return hits[0] if hits else None


def _materialize_localized_table(src, dst):
    import shutil
    try:
        if src.name.lower().endswith(".csv.zip"):
            import zipfile
            with zipfile.ZipFile(str(src)) as zf:
                members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
                if not members:
                    return False
                member = sorted(members, key=len)[-1]
                with zf.open(member) as fin, open(str(dst), "wb") as fout:
                    shutil.copyfileobj(fin, fout, 1 << 20)
        else:
            shutil.copy2(str(src), str(dst))
        return dst.is_file() and dst.stat().st_size > 0
    except (OSError, EOFError, KeyError, ValueError):
        return False


def _materialize_localized_tables(public_dir):
    # MLE-Bench prepare ships a few competitions with localized table
    # prefixes and zips (text-normalization en_/ru_ style). Prefix-agnostic
    # scan: any <prefix>_train.csv(.zip) with sibling <prefix>_test* and
    # <prefix>_sample_submission* tables is materialized to canonical
    # train.csv / test.csv / sample_submission.csv. Idempotent.
    if _first_file(public_dir, _TRAIN_FILE_NAMES) is not None:
        return False
    try:
        entries = list(public_dir.iterdir())
    except OSError:
        return False
    trains = {}
    for p in entries:
        if not p.is_file():
            continue
        name = p.name
        base = name[:-len(".zip")] if name.lower().endswith(".csv.zip") else name
        if base.lower().endswith("_train.csv"):
            prefix = base[:-len("_train.csv")]
            trains.setdefault(prefix, p)
    for prefix in sorted(trains):
        train_src = trains[prefix]
        test_src = _localized_table_entries(entries, prefix, "test")
        sample_src = _localized_table_entries(entries, prefix, "sample_submission")
        if test_src is None or sample_src is None:
            continue
        if (_materialize_localized_table(train_src, public_dir / "train.csv")
                and _materialize_localized_table(test_src, public_dir / "test.csv")
                and _materialize_localized_table(sample_src, public_dir / "sample_submission.csv")):
            return True
    return False

def resolve_dataset_layout(data_dir, sample_path: Optional[str] = None) -> DatasetLayout:
    root = Path(data_dir).expanduser()
    for public_dir, private_dir, layout_name in _candidate_pairs(root):
        if not public_dir.is_dir() or not private_dir.is_dir():
            continue
        train_path = _first_file(public_dir, _TRAIN_FILE_NAMES)
        if train_path is None and public_dir == root:
            train_path = _first_file(root, _TRAIN_FILE_NAMES)
        if train_path is None:
            # v2.5.5: localized-prefix tables (MLE-Bench prepare quirk).
            # Prefix-agnostic materialization; never competition-specific.
            if _materialize_localized_tables(public_dir):
                train_path = _first_file(public_dir, _TRAIN_FILE_NAMES)
        test_path, test_has_labels = _resolve_test_path(public_dir,
                                                        private_dir, root)
        train_image_dir = _first_image_dir(public_dir, _TRAIN_IMAGE_DIR_NAMES)
        test_image_dir = _first_image_dir(public_dir, _TEST_IMAGE_DIR_NAMES)
        if train_image_dir is None:
            train_image_dir = _first_image_dir(root, _TRAIN_IMAGE_DIR_NAMES)
        if test_image_dir is None:
            test_image_dir = _first_image_dir(root, _TEST_IMAGE_DIR_NAMES)
        # v2.3.7: paired-image pixel regression (generic). When NO train
        # table exists but the public dir holds a paired target-image dir
        # (matching stems) AND the sample submission is per-pixel
        # '<stem>_<row>_<col>', the labels live inside the target images:
        # the layout resolves to the synthesized train.csv (written by
        # synthesize_train_labels before analysis) + the label-free test
        # table. train.csv rows are pixels; the target is intensity/255.
        sample = _resolve_sample(sample_path, public_dir, root)
        pixel = _per_pixel_sample(sample)
        paired = _paired_target_dir(public_dir, train_image_dir,
                                    test_image_dir)
        if train_path is None and test_path is not None and \
                paired is not None and pixel is not None:
            return DatasetLayout(
                root=root,
                train_path=public_dir / "train.csv",
                test_path=test_path,
                sample_submission_path=sample,
                public_dir=public_dir,
                private_dir=private_dir,
                train_image_dir=train_image_dir,
                test_image_dir=test_image_dir,
                layout_name="paired_image_pixel_regression",
                gold_test_csv=(
                    _first_file(private_dir, _PRIVATE_TEST_FILE_NAMES)
                    if private_dir != public_dir else None),
                test_has_labels=test_has_labels,
                paired_target_dir=paired,
                pixel_level=True,
            )
        if train_path is None or test_path is None:
            # v2.3.8: no train/test table but a sample submission exists and
            # the tree holds image/audio content -> synthesize a train table
            # (dir labels first, then sample copy) and a label-free id-only
            # test table so mask/detection/audio layouts resolve.
            synthesized = _synthesize_missing_tables(
                public_dir, root, sample, train_path, test_path)
            if synthesized is not None:
                train_path, test_path = synthesized
        if train_path is None or test_path is None:
            continue
        return DatasetLayout(
            root=root,
            train_path=train_path,
            test_path=test_path,
            sample_submission_path=sample,
            public_dir=public_dir,
            private_dir=private_dir,
            train_image_dir=train_image_dir,
            test_image_dir=test_image_dir,
            layout_name=layout_name,
            gold_test_csv=(
                _first_file(private_dir, _PRIVATE_TEST_FILE_NAMES)
                if private_dir != public_dir else None),
            test_has_labels=test_has_labels,
            paired_target_dir=paired,
            pixel_level=bool(pixel and paired),
        )
    raise DatasetLayoutError(
        "unsupported dataset layout under %s; expected train.csv/test.csv "
        "or prepared/public + prepared/private" % root)


def materialize_dataset(data_dir) -> dict:
    """Extract train.zip/test.zip under prepared/public when missing.

    MLE-bench prepared dirs sometimes keep images zipped. This runs once on
    the host before analysis/preflight so candidate code always sees real
    image directories. Returns a report dict (extracted/skipped paths).
    """
    import os
    import zipfile
    root = Path(data_dir).expanduser()
    public = root / "prepared" / "public"
    report = {"extracted": [], "skipped": []}
    if not public.is_dir():
        return report
    # v2.3.8: image archives ship under train_images.zip / test_images.zip
    # (kuzushiji-recognition removes the raw dirs after archiving).
    for zip_name, top_name in (("train.zip", "train"), ("test.zip", "test"),
                               ("train_images.zip", "train_images"),
                               ("test_images.zip", "test_images")):
        zip_path = public / zip_name
        if not zip_path.is_file():
            continue
        target_dir = public / top_name
        if target_dir.is_dir() and any(target_dir.iterdir()):
            report["skipped"].append(str(zip_path))
            continue
        with zipfile.ZipFile(zip_path) as zf:
            names = [m.filename for m in zf.infolist() if not m.is_dir()]
            tops = {n.split("/", 1)[0] for n in names if "/" in n}
            flats = [n for n in names if "/" not in n]
            dest = public if (len(tops) == 1 and not flats) else target_dir
            dest.mkdir(parents=True, exist_ok=True)
            base = str(dest.resolve()) + os.sep
            for m in zf.infolist():
                if m.is_dir():
                    continue
                out = str((dest / m.filename).resolve())
                if not out.startswith(base):
                    raise DatasetLayoutError(
                        "unsafe zip member in %s: %s" % (zip_path.name, m.filename))
            zf.extractall(dest)
        report["extracted"].append(str(zip_path))
    return report


def _guess_target_column(public_dir: Path, train_names,
                          sample_path: Optional[Path] = None) -> str:
    """Generic target discovery shared by sanitize/analyzer paths: the
    sample submission's non-id column that exists in the train header wins
    (multi-output / mid-table targets like taxi fare_amount), else the last
    train column. Supports csv+tsv train tables."""
    import csv as _csv
    train = _first_file(public_dir, train_names)
    header = []
    if train is not None:
        try:
            with open(train, "r", encoding="utf-8", errors="replace",
                      newline="") as fh:
                header = next(_csv.reader(fh, delimiter=table_delimiter(train)),
                              [])
        except OSError:
            header = []
    if not header:
        return ""
    if sample_path is not None and Path(sample_path).is_file():
        try:
            with open(sample_path, "r", encoding="utf-8", errors="replace",
                      newline="") as fh:
                sample = list(_csv.reader(fh))
        except OSError:
            sample = []
        if len(sample) >= 2 and len(sample[0]) >= 2:
            s_lower = [str(c).strip().lower() for c in sample[0]]
            for idx, name in enumerate(s_lower):
                if idx == 0:
                    continue  # id column
                for h in header:
                    if str(h).strip().lower() == name:
                        return str(h)
    return header[-1] if header else ""


def sanitize_test_csv(data_dir, target_column: str = "") -> dict:
    """Write a label-free public/test.csv when only the private test exists.

    The private (gold) test CSV contains the target column; candidate code
    must never see it. When public/test.csv (or test/test.csv / *.tsv) is
    missing, copy the private test rows with the target column dropped into
    prepared/public/test.csv so the resolver picks a label-free source and
    the container never needs to mount private/. Handles csv+tsv private
    tables under every MLE-Bench private name (test.csv / answers.csv /
    gold_submission.csv) and discovers the target via the train header OR
    the sample submission (generic, not competition-specific). Idempotent.
    """
    import csv as _csv
    root = Path(data_dir).expanduser()
    public = root / "prepared" / "public"
    private = root / "prepared" / "private"
    report = {"written": "", "skipped": ""}
    if not public.is_dir() or not private.is_dir():
        return report
    for name in ("test.csv", "test.tsv"):
        existing = public / name
        if existing.is_file():
            report["skipped"] = str(existing)
            return report
    nested = public / "test" / "test.csv"
    if nested.is_file():
        report["skipped"] = str(nested)
        return report
    private_test = _first_file(private, _PRIVATE_TEST_FILE_NAMES)
    if private_test is None:
        return report
    existing = public / "test.csv"  # always write the canonical csv name
    # Guess the target column from the sample submission first, then the
    # train header (same generic rule as the analyzer).
    target = target_column or ""
    if not target:
        target = _guess_target_column(
            public, _TRAIN_FILE_NAMES,
            sample_path=_resolve_sample(None, public, root))
    delim = table_delimiter(private_test)
    target_lower = target.strip().lower() if target else ""
    keep_idx = None
    rows_written = 0
    with open(existing, "w", encoding="utf-8", newline="") as fh:
        writer = _csv.writer(fh)
        with open(private_test, "r", encoding="utf-8", errors="replace",
                  newline="") as src_fh:
            reader = _csv.reader(src_fh, delimiter=delim)
            for i, row in enumerate(reader):
                if i == 0:
                    header = row
                    if target_lower:
                        keep_idx = [k for k, col in enumerate(header)
                                    if col.strip().lower() != target_lower]
                        if len(keep_idx) == len(header):
                            # Target name not present in the private table
                            # under ANY case: the table carries no gold
                            # column under that name, so keep every column.
                            # (The old last-column fallback over-dropped
                            # real features such as passenger_count in
                            # taxi-shaped tables; every MLE-Bench private
                            # answers table shares the train/sample column
                            # names, so the gold drop above is sufficient.)
                            keep_idx = list(range(len(header)))
                    else:
                        keep_idx = list(range(len(header)))
                    writer.writerow([row[k] for k in keep_idx
                                     if k < len(row)])
                    continue
                if keep_idx is not None:
                    writer.writerow([row[k] for k in keep_idx
                                     if k < len(row)])
                    rows_written += 1
    if rows_written == 0:
        # Empty private table: leave nothing behind (invalid dataset).
        try:
            existing.unlink()
        except OSError:
            pass
        return report
    report["written"] = str(existing)
    report["rows"] = rows_written
    return report


def synthesize_train_labels(data_dir) -> dict:
    """Generic fallback: build public/train.csv when a competition ships NO
    train labels table (labels live in the file structure).

    Two deterministic modes, both never touching a dataset that already has
    a train table:
      * class-dir mode: train image dir has one level of class subdirs
        (plant-seedlings train/<species>/*.png, iwildcam-style folders) ->
        (file, label) rows from the subdir names;
      * flat-prefix mode: flat train image dir where basenames carry a
        short repeated label token before the first '.' (cat.0.jpg /
        dog.1.jpg) with 2..64 distinct tokens and >= 50 image
        files (image-evidence floor) -> (file, token) rows.
    Column names mirror the sample submission (id column first, first
    non-id column as label) so the analyzer's sample-driven target and the
    compiled harness agree. Idempotent.
    """
    import csv as _csv
    root = Path(data_dir).expanduser()
    public = root / "prepared" / "public"
    report = {"written": "", "skipped": ""}
    if not public.is_dir():
        return report
    if _first_file(public, _TRAIN_FILE_NAMES) is not None:
        report["skipped"] = "has-train-table"
        return report
    train_dir = _first_image_dir(public, _TRAIN_IMAGE_DIR_NAMES)
    if train_dir is None:
        report["skipped"] = "no-image-dir"
        return report
    # Column names from the sample submission (generic contract).
    sample_path = _resolve_sample(None, public, root)
    # Mode 0: paired-image pixel regression (v2.3.7, generic). The train
    # input dir has a sibling TARGET dir with matching stems and the sample
    # submission is per-pixel ('<stem>_<row>_<col>', value 0..1): labels
    # live inside the target images. One CSV row per sampled pixel.
    pixel = _per_pixel_sample(sample_path)
    if pixel is not None:
        test_dir = _first_image_dir(public, _TEST_IMAGE_DIR_NAMES)
        target_dir = _paired_target_dir(public, train_dir, test_dir)
        if target_dir is not None:
            import os as _os
            cap = int(_os.environ.get("V2_PIXEL_ROW_CAP", "400000") or 400000)
            cap = max(1, int(cap))
            rows, images, stride, mode = _synthesize_paired_rows(
                train_dir, target_dir, pixel, cap)
            if rows:
                id_col, value_col = pixel
                out = public / "train.csv"
                with open(out, "w", encoding="utf-8", newline="") as fh:
                    writer = _csv.writer(fh)
                    writer.writerow([id_col, value_col])
                    writer.writerows(rows)
                report["written"] = str(out)
                report["mode"] = "paired-image"
                report["rows"] = len(rows)
                report["images"] = images
                report["stride"] = stride
                report["target_dir"] = str(target_dir)
                return report
            report["skipped"] = "paired-images-not-decodable"
    id_col, label_col = "file", "label"
    if sample_path is not None:
        try:
            with open(sample_path, "r", encoding="utf-8", errors="replace",
                      newline="") as fh:
                sample = list(_csv.reader(fh))
        except OSError:
            sample = []
        if sample and sample[0]:
            id_col = str(sample[0][0])
            if len(sample[0]) > 1:
                label_col = str(sample[0][1])
    rows = []
    mode = ""
    # Mode 1: class subdirs.
    subdirs = sorted(p for p in train_dir.iterdir() if p.is_dir())
    class_rows = []
    for sub in subdirs:
        try:
            imgs = [f for f in sorted(sub.iterdir())
                    if f.is_file() and _is_image_file(f)]
        except OSError:
            imgs = []
        for f in imgs:
            class_rows.append((f.name, sub.name))
    if len(class_rows) >= 2 and len(subdirs) >= 2:
        rows = class_rows
        mode = "class-dirs"
    else:
        # Mode 2: flat dir with label token before first '.'.
        try:
            flat = [f for f in sorted(train_dir.iterdir())
                    if f.is_file() and _is_image_file(f)]
        except OSError:
            flat = []
        if len(flat) >= 50:
            tokens = {}
            for f in flat:
                token = f.stem.split(".", 1)[0].strip() or f.stem
                tokens.setdefault(token, []).append(f)
            if 2 <= len(tokens) <= 64:
                for token in sorted(tokens):
                    for f in tokens[token]:
                        rows.append((f.name, token))
                mode = "flat-prefix"
    if not rows:
        report["skipped"] = "no-synthesizable-labels"
        return report
    out = public / "train.csv"
    with open(out, "w", encoding="utf-8", newline="") as fh:
        writer = _csv.writer(fh)
        writer.writerow([id_col, label_col])
        writer.writerows(rows)
    report["written"] = str(out)
    report["mode"] = mode
    report["rows"] = len(rows)
    return report
