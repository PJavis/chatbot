"""Build chunked knowledge docs for Rasa Enterprise Search."""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _load_corpus(source: Path) -> list[dict]:
    corpus_path = source / "corpus.json" if source.is_dir() else source
    if not corpus_path.exists():
        raise FileNotFoundError(f"Không tìm thấy corpus.json tại: {corpus_path}")

    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)

    if not isinstance(corpus, list):
        raise ValueError("corpus.json phải chứa một list các passage")

    return corpus


def _slugify(value: str) -> str:
    value = value.lower().strip()
    chars = []
    for ch in value:
        if ch.isalnum():
            chars.append(ch)
        else:
            chars.append("_")
    slug = "".join(chars)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "chunk"


def _chunk_words(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size phải lớn hơn 0")
    if overlap >= chunk_size:
        raise ValueError("overlap phải nhỏ hơn chunk_size")

    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(words):
        chunk = words[start : start + chunk_size]
        if not chunk:
            break
        chunks.append(" ".join(chunk).strip())
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks


def build_docs(
    corpus: list[dict],
    target_dir: Path,
    chunk_size: int,
    overlap: int,
    workers: int,
) -> dict[str, int]:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    def _build_passage_docs(item: tuple[int, dict]) -> tuple[int, int]:
        index, passage = item
        passage_id = passage.get("passage_id", f"passage_{index:05d}")
        title = passage.get("title", "").strip()
        context = passage.get("context", "").strip()
        if not context:
            return 0, 0

        chunks = _chunk_words(context, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            return 0, 0

        for idx, chunk in enumerate(chunks, start=1):
            doc_name = f"{_slugify(passage_id)}_{idx:03d}.txt"
            doc_path = target_dir / doc_name
            content = (
                f"Title: {title}\n"
                f"Passage ID: {passage_id}\n"
                f"Chunk: {idx}/{len(chunks)}\n\n"
                f"{chunk}\n"
            )
            with open(doc_path, "w", encoding="utf-8") as f:
                f.write(content)

        return 1, len(chunks)

    doc_count = 0
    chunk_count = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for passages, chunks in executor.map(_build_passage_docs, enumerate(corpus)):
            doc_count += passages
            chunk_count += chunks

    return {"passages": doc_count, "chunks": chunk_count}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build chunked docs for Rasa Enterprise Search")
    parser.add_argument(
        "--source",
        default=str(Path(__file__).resolve().parents[2] / "data" / "processed"),
        help="Directory containing corpus.json",
    )
    parser.add_argument(
        "--target",
        default=str(Path(__file__).resolve().parents[1] / "docs"),
        help="Output directory for chunked .txt docs",
    )
    parser.add_argument("--chunk-size", type=int, default=140, help="Words per chunk")
    parser.add_argument("--overlap", type=int, default=40, help="Word overlap between chunks")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent workers for chunking/writing")
    args = parser.parse_args()

    source = Path(args.source)
    target = Path(args.target)

    corpus = _load_corpus(source)
    stats = build_docs(corpus, target, chunk_size=args.chunk_size, overlap=args.overlap, workers=args.workers)

    print(f"Đã tạo knowledge docs tại {target}")
    print(f"  Passages: {stats['passages']:,}")
    print(f"  Chunks   : {stats['chunks']:,}")


if __name__ == "__main__":
    main()
