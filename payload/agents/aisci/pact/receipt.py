# -*- coding: utf-8 -*-
"""pact/receipt.py - TrialReceipt JSON serialization helpers."""
import json
from pathlib import Path
from typing import Union

from v2_contracts import TrialReceipt


def dump_receipt(receipt: TrialReceipt, path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(receipt.to_dict(), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    return path


def load_receipt(path: Union[str, Path]) -> TrialReceipt:
    return TrialReceipt.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
