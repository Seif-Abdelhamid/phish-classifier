"""Download, verify and clean the public phishing-email dataset.

Source: zefang-liu/phishing-email-dataset on Hugging Face (LGPL-3.0), a mirror of a Kaggle
collection of 18,650 emails labeled "Phishing Email" or "Safe Email". The dataset is not
redistributed here: ``download`` fetches it and refuses a file whose SHA-256 does not match.

The raw file has many exact and near-exact duplicates. If the same email lands in both the
training and the test split, the test score measures memory, not detection. ``load`` therefore
drops empty rows and removes duplicates after normalizing whitespace and case, before any split.
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
import sys
import urllib.request
from dataclasses import dataclass

DATA_URL = "https://huggingface.co/datasets/zefang-liu/phishing-email-dataset/resolve/main/Phishing_Email.csv"
DATA_SHA256 = "18ef4fff1acb8986f0ab01e83ede6025420c829656af5a377560fce17e153b97"
MAX_BYTES = 100 * 1024 * 1024
LABELS = {"Phishing Email": 1, "Safe Email": 0}


class DataError(ValueError):
    """The dataset is missing, altered or not in the expected shape."""


@dataclass
class Dataset:
    texts: list[str]
    labels: list[int]
    raw_rows: int
    dropped_empty: int
    dropped_duplicates: int


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(dest: str, url: str = DATA_URL, expected_sha256: str = DATA_SHA256) -> str:
    """Fetch the CSV to ``dest`` (once) and verify its hash. Returns the path."""
    if not os.path.exists(dest):
        if not url.startswith("https://"):
            raise DataError("refusing a non-HTTPS dataset URL")
        os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
        tmp = dest + ".part"
        with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as out:  # noqa: S310  # nosec B310
            total = 0
            while chunk := resp.read(1 << 20):
                total += len(chunk)
                if total > MAX_BYTES:
                    raise DataError("download larger than expected; refusing")
                out.write(chunk)
        os.replace(tmp, dest)
    actual = sha256_of(dest)
    if actual != expected_sha256:
        raise DataError(f"{dest}: SHA-256 {actual} does not match the pinned {expected_sha256}")
    return dest


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def load(path: str) -> Dataset:
    """Read the CSV, map labels, drop empty rows and duplicates (keeping the first copy)."""
    csv.field_size_limit(min(sys.maxsize, MAX_BYTES))
    texts: list[str] = []
    labels: list[int] = []
    seen: dict[str, int] = {}
    raw = empty = dupes = 0
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or not {"Email Text", "Email Type"} <= set(reader.fieldnames):
            raise DataError("expected columns 'Email Text' and 'Email Type'")
        for row in reader:
            raw += 1
            text = row.get("Email Text") or ""
            label_name = (row.get("Email Type") or "").strip()
            if label_name not in LABELS:
                raise DataError(f"row {raw}: unknown label {label_name!r}")
            norm = normalize(text)
            if not norm or norm == "empty":
                empty += 1
                continue
            key = hashlib.sha256(norm.encode()).hexdigest()
            if key in seen:
                dupes += 1
                continue
            seen[key] = LABELS[label_name]
            texts.append(text)
            labels.append(LABELS[label_name])
    if not texts:
        raise DataError("no usable rows")
    return Dataset(texts, labels, raw, empty, dupes)
