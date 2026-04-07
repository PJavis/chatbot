"""
Gradio demo – UIT-ViQuAD2.0 Dataset Explorer & RAG Evaluator.

Tabs:
  1. Khám phá Dataset  – xem examples, tìm kiếm theo từ khoá
  2. Đánh giá Metrics  – nhập prediction/ground-truth, xem EM/F1/BLEU ngay lập tức
  3. Chạy Evaluation   – upload file predictions.json, chạy đánh giá trên validation/test

Usage:
    python demo/app.py
    # hoặc
    python -m demo.app
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import gradio as gr
import pandas as pd

# Thêm root vào sys.path để import data/evaluation
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.metrics import bleu_score, exact_match, token_f1

# ---------------------------------------------------------------------------
# Load dữ liệu (lazy – chỉ load khi app chạy)
# ---------------------------------------------------------------------------

_corpus: list[dict] | None = None
_splits: dict[str, list[dict]] = {}


def _load_data():
    global _corpus, _splits
    try:
        from data import load_corpus, load_split
        _corpus = load_corpus()
        for split in ("train", "validation", "test"):
            _splits[split] = load_split(split)
    except FileNotFoundError:
        _corpus = []
        _splits = {"train": [], "validation": [], "test": []}


# ---------------------------------------------------------------------------
# Tab 1: Khám phá Dataset
# ---------------------------------------------------------------------------

def search_examples(keyword: str, split: str, show_impossible: str, max_results: int):
    """Tìm kiếm examples theo từ khoá trong câu hỏi hoặc context."""
    examples = _splits.get(split, [])
    if not examples:
        return pd.DataFrame(), "Dataset chưa được tải. Hãy chạy `python -m data.download` trước."

    keyword = keyword.strip().lower()
    show_imp = show_impossible == "Tất cả"

    results = []
    for ex in examples:
        if not show_imp and ex.get("is_impossible"):
            continue
        q = ex["question"].lower()
        c_title = ex.get("title", "").lower()
        if keyword and keyword not in q and keyword not in c_title:
            continue

        answers = [a["text"] for a in ex.get("answers", [])]
        results.append(
            {
                "ID": ex["id"],
                "Tiêu đề": ex.get("title", ""),
                "Câu hỏi": ex["question"],
                "Đáp án": " | ".join(answers) if answers else "(không có đáp án)",
                "Không trả lời được": "Có" if ex.get("is_impossible") else "Không",
            }
        )
        if len(results) >= max_results:
            break

    if not results:
        return pd.DataFrame(), f"Không tìm thấy kết quả cho '{keyword}'."

    df = pd.DataFrame(results)
    info = f"Hiển thị {len(results)} / {len(examples)} examples trong split '{split}'"
    return df, info


def show_dataset_stats():
    """Hiển thị thống kê tổng quan dataset."""
    if not _splits:
        return "Dataset chưa được tải."

    lines = ["**Thống kê UIT-ViQuAD2.0**\n"]
    total = 0
    for split, examples in _splits.items():
        if not examples:
            continue
        n = len(examples)
        answerable = sum(1 for e in examples if not e.get("is_impossible"))
        unanswerable = n - answerable
        total += n
        lines.append(
            f"- **{split}**: {n:,} mẫu "
            f"({answerable:,} có đáp án / {unanswerable:,} không có đáp án)"
        )

    corpus_size = len(_corpus) if _corpus else 0
    lines.append(f"\n**Corpus**: {corpus_size:,} đoạn văn duy nhất")
    lines.append(f"**Tổng cộng**: {total:,} mẫu")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tab 2: Đánh giá tương tác (Metrics Calculator)
# ---------------------------------------------------------------------------

def calculate_metrics(prediction: str, ground_truths_raw: str) -> str:
    """Tính EM, F1, BLEU giữa một prediction và ground truths."""
    if not prediction.strip():
        return "Vui lòng nhập câu trả lời dự đoán."
    if not ground_truths_raw.strip():
        return "Vui lòng nhập ít nhất một đáp án đúng."

    ground_truths = [line.strip() for line in ground_truths_raw.strip().splitlines() if line.strip()]

    em = exact_match(prediction, ground_truths)
    f1 = token_f1(prediction, ground_truths)
    bleu = bleu_score(prediction, ground_truths)

    lines = [
        "### Kết quả",
        f"| Metric  | Giá trị  |",
        f"|---------|----------|",
        f"| Exact Match | {'✅ 1.0' if em else '❌ 0.0'} |",
        f"| Token F1    | {f1:.4f} ({f1 * 100:.1f}%) |",
        f"| BLEU-1  | {bleu['bleu1']:.4f} |",
        f"| BLEU-2  | {bleu['bleu2']:.4f} |",
        f"| BLEU-3  | {bleu['bleu3']:.4f} |",
        f"| BLEU-4  | {bleu['bleu4']:.4f} |",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tab 3: Chạy Evaluation từ file predictions
# ---------------------------------------------------------------------------

def run_evaluation(predictions_file, split: str) -> str:
    """Upload file predictions.json → chạy evaluation → hiển thị kết quả."""
    from evaluation.evaluate import evaluate
    from evaluation.metrics import evaluate_predictions

    examples = _splits.get(split, [])
    if not examples:
        return f"Split '{split}' chưa được tải. Hãy chạy `python -m data.download` trước."

    if predictions_file is None:
        # Chạy baseline oracle để demo
        from evaluation.evaluate import make_dummy_predictions
        predictions = make_dummy_predictions(examples, "first_answer")
        note = "> **Lưu ý:** Không có file predictions → đang chạy **Oracle baseline** (lấy đáp án đầu tiên).\n\n"
    else:
        try:
            with open(predictions_file.name, encoding="utf-8") as f:
                predictions = json.load(f)
            note = f"> Đã load **{len(predictions):,}** predictions từ file.\n\n"
        except Exception as e:
            return f"Lỗi khi đọc file predictions: {e}"

    metrics = evaluate_predictions(examples, predictions)

    lines = [note, f"### Kết quả trên split `{split}`\n"]
    lines.append("| Metric | Giá trị |")
    lines.append("|--------|---------|")
    lines.append(f"| Số mẫu | {metrics['num_examples']:,} |")
    lines.append(f"| **Exact Match** | **{metrics['exact_match']:.2f}%** |")
    lines.append(f"| **Token F1** | **{metrics['token_f1']:.2f}%** |")
    lines.append(f"| BLEU-1 | {metrics['bleu1']:.2f}% |")
    lines.append(f"| BLEU-4 | {metrics['bleu4']:.2f}% |")
    if "unanswerable_accuracy" in metrics:
        lines.append(f"| Unanswerable Acc. | {metrics['unanswerable_accuracy']:.2f}% |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    _load_data()

    with gr.Blocks(title="UIT-ViQuAD2.0 Explorer & Evaluator", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# UIT-ViQuAD2.0 – Dataset Explorer & RAG Evaluator\n"
            "Dataset tiếng Việt cho bài toán **Reading Comprehension** (SQuAD 2.0 format).\n"
            "> Chạy `python -m data.download` để tải dataset trước khi sử dụng."
        )

        with gr.Tabs():

            # ----------------------------------------------------------------
            # Tab 1: Khám phá Dataset
            # ----------------------------------------------------------------
            with gr.Tab("Khám phá Dataset"):
                gr.Markdown("### Tìm kiếm & xem examples")
                with gr.Row():
                    keyword_input = gr.Textbox(label="Từ khoá (câu hỏi hoặc tiêu đề)", placeholder="Ví dụ: Hà Nội")
                    split_dropdown = gr.Dropdown(
                        choices=["train", "validation", "test"],
                        value="validation",
                        label="Split",
                    )
                    impossible_dropdown = gr.Dropdown(
                        choices=["Tất cả", "Chỉ có đáp án"],
                        value="Tất cả",
                        label="Loại câu hỏi",
                    )
                    max_results_slider = gr.Slider(10, 200, value=50, step=10, label="Số kết quả tối đa")

                search_btn = gr.Button("Tìm kiếm", variant="primary")
                info_text = gr.Markdown()
                results_table = gr.Dataframe(wrap=True)

                stats_btn = gr.Button("Xem thống kê dataset")
                stats_output = gr.Markdown()

                search_btn.click(
                    search_examples,
                    inputs=[keyword_input, split_dropdown, impossible_dropdown, max_results_slider],
                    outputs=[results_table, info_text],
                )
                stats_btn.click(show_dataset_stats, outputs=stats_output)

            # ----------------------------------------------------------------
            # Tab 2: Metrics Calculator
            # ----------------------------------------------------------------
            with gr.Tab("Tính Metrics (EM / F1 / BLEU)"):
                gr.Markdown(
                    "### Nhập prediction và ground truths để tính ngay EM, F1, BLEU\n"
                    "Mỗi ground truth một dòng."
                )
                with gr.Row():
                    pred_input = gr.Textbox(label="Câu trả lời dự đoán (prediction)", lines=3)
                    gt_input = gr.Textbox(
                        label="Đáp án đúng (ground truths, mỗi dòng một đáp án)",
                        lines=3,
                        placeholder="năm 1831\n1831",
                    )
                calc_btn = gr.Button("Tính metrics", variant="primary")
                metrics_output = gr.Markdown()
                calc_btn.click(calculate_metrics, inputs=[pred_input, gt_input], outputs=metrics_output)

                gr.Examples(
                    examples=[
                        ["1831", "năm 1831\n1831\nnăm Minh Mạng 1831"],
                        ["Hà Nội", "Thành phố Hà Nội\nHà Nội\nHN"],
                        ["câu trả lời sai hoàn toàn", "năm 1831"],
                    ],
                    inputs=[pred_input, gt_input],
                )

            # ----------------------------------------------------------------
            # Tab 3: Evaluation từ file
            # ----------------------------------------------------------------
            with gr.Tab("Chạy Evaluation"):
                gr.Markdown(
                    "### Upload file predictions.json và chạy evaluation trên validation/test\n"
                    "**Định dạng file**: `{\"example_id\": \"predicted_answer\", ...}`\n\n"
                    "Nếu không upload file → chạy **Oracle baseline** để xem kết quả mẫu."
                )
                with gr.Row():
                    pred_file = gr.File(label="File predictions.json (tuỳ chọn)", file_types=[".json"])
                    eval_split = gr.Dropdown(
                        choices=["validation", "test"],
                        value="validation",
                        label="Split để đánh giá",
                    )
                eval_btn = gr.Button("Chạy Evaluation", variant="primary")
                eval_output = gr.Markdown()
                eval_btn.click(run_evaluation, inputs=[pred_file, eval_split], outputs=eval_output)

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)
