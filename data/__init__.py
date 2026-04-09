"""Helpers để load dataset UIT-ViQuAD2.0 đã xử lý."""

import json
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent / "processed"


def load_corpus() -> list[dict]:
    """Load danh sách đoạn văn (passage)."""
    with open(PROCESSED_DIR / "corpus.json", encoding="utf-8") as f:
        return json.load(f)


def load_split(split: str) -> list[dict]:
    """Load QA examples của một split (train / validation / test)."""
    assert split in ("train", "validation", "test"), f"Split không hợp lệ: {split}"
    with open(PROCESSED_DIR / f"{split}.json", encoding="utf-8") as f:
        return json.load(f)
