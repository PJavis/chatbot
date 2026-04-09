"""
Download và làm sạch dataset UIT-ViQuAD2.0.

Usage:
    python -m data.download
"""

import json
import re
import unicodedata
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

RAW_DIR = Path(__file__).parent / "raw"
PROCESSED_DIR = Path(__file__).parent / "processed"


# ---------------------------------------------------------------------------
# Làm sạch văn bản
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Chuẩn hoá text: NFC unicode + thu gọn khoảng trắng."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_answer(text: str) -> str:
    """Xoá khoảng trắng thừa đầu/cuối trong đáp án."""
    return normalize_text(text)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_raw() -> object:
    """Tải UIT-ViQuAD2.0 từ HuggingFace, lưu JSON thô theo từng split."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("Đang tải UIT-ViQuAD2.0 từ HuggingFace...")
    dataset = load_dataset("taidng/UIT-ViQuAD2.0")

    for split in ("train", "validation", "test"):
        out_path = RAW_DIR / f"{split}.json"
        records = [dict(item) for item in tqdm(dataset[split], desc=f"  {split}")]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"  {split}: {len(records):,} mẫu → {out_path}")

    return dataset


# ---------------------------------------------------------------------------
# Xây dựng corpus đoạn văn
# ---------------------------------------------------------------------------

def build_corpus(dataset) -> list[dict]:
    """
    Tổng hợp tất cả đoạn văn (passage) duy nhất từ ba split.

    Mỗi passage: {passage_id, title, context}
    """
    seen: set[tuple] = set()
    corpus: list[dict] = []

    for split in ("train", "validation", "test"):
        for item in dataset[split]:
            key = (item["title"], item["context"])
            if key not in seen:
                seen.add(key)
                corpus.append(
                    {
                        "passage_id": f"p{len(corpus):05d}",
                        "title": normalize_text(item["title"]),
                        "context": normalize_text(item["context"]),
                    }
                )
    return corpus


# ---------------------------------------------------------------------------
# Xử lý từng split thành QA examples
# ---------------------------------------------------------------------------

def process_split(
    dataset,
    split: str,
    corpus_index: dict[tuple, str],
) -> list[dict]:
    """
    Chuyển đổi split gốc thành danh sách QA examples sạch.

    Mỗi example: {id, passage_id, title, question, is_impossible, answers}
    """
    examples = []
    for item in tqdm(dataset[split], desc=f"  Xử lý {split}"):
        key = (normalize_text(item["title"]), normalize_text(item["context"]))
        passage_id = corpus_index.get(key, "")

        raw_answers = item.get("answers") or {}
        answer_texts = raw_answers.get("text", [])
        answer_starts = raw_answers.get("answer_start", [])

        examples.append(
            {
                "id": item["id"],
                "passage_id": passage_id,
                "title": normalize_text(item["title"]),
                "question": normalize_text(item["question"]),
                "is_impossible": bool(item.get("is_impossible", False)),
                "answers": [
                    {"text": clean_answer(t), "answer_start": s}
                    for t, s in zip(answer_texts, answer_starts)
                ],
            }
        )
    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    dataset = download_raw()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("\nXây dựng corpus đoạn văn...")
    corpus = build_corpus(dataset)
    corpus_path = PROCESSED_DIR / "corpus.json"
    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    print(f"  Corpus: {len(corpus):,} đoạn văn → {corpus_path}")

    corpus_index: dict[tuple, str] = {
        (p["title"], p["context"]): p["passage_id"] for p in corpus
    }

    print("\nXử lý các splits...")
    total_answerable = 0
    total_unanswerable = 0
    for split in ("train", "validation", "test"):
        examples = process_split(dataset, split, corpus_index)
        out_path = PROCESSED_DIR / f"{split}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)
        answerable = sum(1 for e in examples if not e["is_impossible"])
        unanswerable = len(examples) - answerable
        total_answerable += answerable
        total_unanswerable += unanswerable
        print(
            f"  {split:12s}: {len(examples):6,} mẫu "
            f"({answerable:,} có đáp án / {unanswerable:,} không có đáp án)"
        )

    print("\n========== Thống kê tổng ==========")
    print(f"  Corpus      : {len(corpus):,} đoạn văn")
    print(f"  Có đáp án   : {total_answerable:,}")
    print(f"  Không đáp án: {total_unanswerable:,}")
    total = total_answerable + total_unanswerable
    print(f"  Tổng cộng   : {total:,}")
    print("====================================")


if __name__ == "__main__":
    main()
