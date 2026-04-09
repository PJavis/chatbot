"""
Script đánh giá RAG trên tập validation/test của UIT-ViQuAD2.0.

Usage:
    python -m evaluation.evaluate --predictions path/to/predictions.json --split validation

Định dạng file predictions.json:
    {"example_id": "predicted_answer_text", ...}

Nếu không có predictions, script sẽ dùng Rasa knowledge base theo mặc định,
hoặc baseline dummy khi chọn `--source oracle`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from data import load_split
from evaluation.metrics import evaluate_predictions
from rasa_inference import generate_rasa_predictions


# ---------------------------------------------------------------------------
# Baseline giả lập để demo (khi chưa có hệ thống RAG thật)
# ---------------------------------------------------------------------------

def make_dummy_predictions(examples: list[dict], strategy: str = "first_answer") -> dict[str, str]:
    """
    Tạo predictions giả để minh hoạ evaluation pipeline.

    Strategies:
      - "first_answer" : trả về đáp án đầu tiên (oracle, upper bound)
      - "random_answer": lấy đáp án ngẫu nhiên từ một example khác
      - "empty"        : luôn trả về chuỗi rỗng (lower bound)
    """
    preds: dict[str, str] = {}
    all_answers = [
        a["text"]
        for ex in examples
        for a in ex.get("answers", [])
        if a["text"]
    ]

    for ex in examples:
        if ex["is_impossible"]:
            preds[ex["id"]] = ""  # Đúng với unanswerable
            continue

        if strategy == "first_answer":
            answers = ex.get("answers", [])
            preds[ex["id"]] = answers[0]["text"] if answers else ""
        elif strategy == "random_answer":
            preds[ex["id"]] = random.choice(all_answers) if all_answers else ""
        else:  # empty
            preds[ex["id"]] = ""

    return preds


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(
    examples: list[dict],
    predictions: dict[str, str],
    retrieved_ids_map: dict[str, list[str]] | None = None,
    output_path: Path | None = None,
) -> dict:
    """Chạy evaluation và (tuỳ chọn) lưu kết quả."""
    metrics = evaluate_predictions(examples, predictions, retrieved_ids_map)

    print("\n========== KẾT QUẢ ĐÁNH GIÁ ==========")
    print(f"  Số mẫu         : {metrics['num_examples']:,}")
    print(f"  Exact Match    : {metrics['exact_match']:.2f}%")
    print(f"  Token F1       : {metrics['token_f1']:.2f}%")
    print(f"  BLEU-1         : {metrics['bleu1']:.2f}%")
    print(f"  BLEU-4         : {metrics['bleu4']:.2f}%")
    if "unanswerable_accuracy" in metrics:
        print(f"  Unanswerable Acc: {metrics['unanswerable_accuracy']:.2f}%")
    for key, val in metrics.items():
        if key.startswith("recall@"):
            print(f"  {key:15s}: {val:.2f}%")
    print("========================================\n")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Đã lưu kết quả → {output_path}")

    return metrics


def evaluate_with_rasa(
    examples: list[dict],
    predictions: dict[str, str] | None = None,
    retrieved_ids_map: dict[str, list[str]] | None = None,
    output_path: Path | None = None,
    rasa_url: str | None = None,
    sender: str | None = None,
    timeout: float | None = None,
) -> dict:
    """Run evaluation using Rasa responses if predictions are not provided."""
    if predictions is None:
        predictions = generate_rasa_predictions(
            examples,
            rasa_url=rasa_url or os.getenv("RASA_REST_URL", "http://localhost:5005/webhooks/rest/webhook"),
            sender=sender or os.getenv("RASA_CHAT_SENDER", "evaluation"),
            timeout=timeout if timeout is not None else float(os.getenv("RASA_API_TIMEOUT", "30")),
            concurrency=4,
        )
    return evaluate(examples, predictions, retrieved_ids_map, output_path)


def main():
    parser = argparse.ArgumentParser(description="Đánh giá RAG trên UIT-ViQuAD2.0")
    parser.add_argument("--split", default="validation", choices=("train", "validation", "test"))
    parser.add_argument("--predictions", default=None, help="Path đến file predictions.json")
    parser.add_argument("--retrieved_ids", default=None, help="Path đến file retrieved_ids.json")
    parser.add_argument("--source", default="rasa", choices=("rasa", "oracle"),
                        help="Nguồn prediction khi không có file predictions")
    parser.add_argument("--strategy", default="first_answer",
                        choices=("first_answer", "random_answer", "empty"),
                        help="Baseline strategy khi không có predictions thật")
    parser.add_argument("--rasa_url", default=os.getenv("RASA_REST_URL", "http://localhost:5005/webhooks/rest/webhook"))
    parser.add_argument("--sender", default=os.getenv("RASA_CHAT_SENDER", "evaluation"))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("RASA_API_TIMEOUT", "30")))
    parser.add_argument("--concurrency", type=int, default=4, help="Số request Rasa chạy đồng thời")
    parser.add_argument("--output", default=None, help="Lưu kết quả JSON vào file này")
    args = parser.parse_args()

    print(f"Đang load split '{args.split}'...")
    examples = load_split(args.split)
    print(f"  {len(examples):,} examples")

    # Load predictions
    if args.predictions:
        with open(args.predictions, encoding="utf-8") as f:
            predictions = json.load(f)
        print(f"Loaded {len(predictions):,} predictions từ {args.predictions}")
    else:
        if args.source == "rasa":
            print(f"Không có predictions thật → gọi Rasa tại {args.rasa_url}")
            predictions = generate_rasa_predictions(
                examples,
                rasa_url=args.rasa_url,
                sender=args.sender,
                timeout=args.timeout,
                concurrency=args.concurrency,
            )
        else:
            print(f"Không có predictions thật → dùng baseline strategy='{args.strategy}'")
            predictions = make_dummy_predictions(examples, args.strategy)

    # Load retrieved ids (tuỳ chọn)
    retrieved_ids_map = None
    if args.retrieved_ids:
        with open(args.retrieved_ids, encoding="utf-8") as f:
            retrieved_ids_map = json.load(f)

    output_path = Path(args.output) if args.output else None
    evaluate(examples, predictions, retrieved_ids_map, output_path)


if __name__ == "__main__":
    main()
