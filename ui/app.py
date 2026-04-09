"""Integrated Gradio UI for dataset exploration, evaluation, and Rasa chat."""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import unicodedata
from pathlib import Path

import gradio as gr
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import load_corpus, load_split
from evaluation.metrics import bleu_score, evaluate_predictions, exact_match, token_f1
from rasa_inference import call_rasa, format_response, generate_rasa_predictions

RASA_REST_URL = os.getenv("RASA_REST_URL", "http://localhost:5005/webhooks/rest/webhook")
RASA_SENDER = os.getenv("RASA_CHAT_SENDER", "gradio_user")
RASA_TIMEOUT = float(os.getenv("RASA_API_TIMEOUT", "30"))
APP_CSS = """
.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto;
}
"""

_corpus: list[dict] | None = None
_splits: dict[str, list[dict]] = {}
_corpus_by_id: dict[str, dict] = {}
LOGGER = logging.getLogger("ui.chat")


def _configure_logging() -> None:
    if LOGGER.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _load_data() -> None:
    _configure_logging()
    global _corpus, _splits, _corpus_by_id
    try:
        _corpus = load_corpus()
        _corpus_by_id = {item["passage_id"]: item for item in _corpus}
        for split in ("train", "validation", "test"):
            _splits[split] = load_split(split)
    except FileNotFoundError:
        _corpus = []
        _splits = {"train": [], "validation": [], "test": []}
        _corpus_by_id = {}


def _fold_text(text: str) -> str:
    """Lowercase text and remove accents for tolerant search matching."""
    normalized = unicodedata.normalize("NFD", text.casefold())
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def _snippet(text: str, keyword: str, radius: int = 140) -> str:
    """Return a compact excerpt around the first keyword match."""
    if not text:
        return ""

    folded_text = _fold_text(text)
    folded_keyword = _fold_text(keyword)
    pos = folded_text.find(folded_keyword)
    if pos < 0:
        excerpt = text[: radius * 2].strip()
        return excerpt if len(text) <= radius * 2 else f"{excerpt}..."

    start = max(0, pos - radius)
    end = min(len(text), pos + len(keyword) + radius)
    excerpt = text[start:end].strip()
    if start > 0:
        excerpt = f"...{excerpt}"
    if end < len(text):
        excerpt = f"{excerpt}..."
    return excerpt


def search_examples(keyword: str, split: str, show_impossible: str, max_results: int):
    """Search examples across the selected split or across all splits."""
    if split == "all":
        examples = [
            {**example, "_split": split_name}
            for split_name, split_examples in _splits.items()
            for example in split_examples
        ]
    else:
        examples = [{**example, "_split": split} for example in _splits.get(split, [])]

    if not examples:
        return pd.DataFrame(), "Dataset chưa được tải. Hãy chạy `python -m data.download` trước."

    keyword = keyword.strip()
    show_imp = show_impossible == "Tất cả"
    folded_keyword = _fold_text(keyword)

    results = []
    for ex in examples:
        if not show_imp and ex.get("is_impossible"):
            continue

        title = ex.get("title", "")
        question = ex.get("question", "")
        answers = [a["text"] for a in ex.get("answers", [])]
        answers_text = " | ".join(answers)
        context = _corpus_by_id.get(ex.get("passage_id", ""), {}).get("context", "")

        searchable_fields = [title, question, answers_text, context]
        if folded_keyword and not any(folded_keyword in _fold_text(field) for field in searchable_fields):
            continue

        results.append(
            {
                "ID": ex["id"],
                "Split": ex.get("_split", split),
                "Tiêu đề": title,
                "Câu hỏi": question,
                "Đáp án": " | ".join(answers) if answers else "(không có đáp án)",
                "Ngữ cảnh": _snippet(context, keyword) if keyword else _snippet(context, ""),
                "Không trả lời được": "Có" if ex.get("is_impossible") else "Không",
            }
        )
        if len(results) >= max_results:
            break

    if not results:
        return pd.DataFrame(), (
            f"Không tìm thấy kết quả cho '{keyword}' trong tiêu đề, câu hỏi, đáp án hoặc ngữ cảnh."
        )

    df = pd.DataFrame(results)
    info = (
        f"Hiển thị {len(results)} / {len(examples)} examples trong split '{split}' "
        "theo tiêu đề, câu hỏi, đáp án hoặc ngữ cảnh."
    )
    return df, info


def show_dataset_stats():
    """Display summary statistics for the processed dataset."""
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


def calculate_metrics(prediction: str, ground_truths_raw: str) -> str:
    """Calculate EM, F1, and BLEU for one prediction and one or more references."""
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
        "| Metric  | Giá trị  |",
        "|---------|----------|",
        f"| Exact Match | {'✅ 1.0' if em else '❌ 0.0'} |",
        f"| Token F1    | {f1:.4f} ({f1 * 100:.1f}%) |",
        f"| BLEU-1  | {bleu['bleu1']:.4f} |",
        f"| BLEU-2  | {bleu['bleu2']:.4f} |",
        f"| BLEU-3  | {bleu['bleu3']:.4f} |",
        f"| BLEU-4  | {bleu['bleu4']:.4f} |",
    ]
    return "\n".join(lines)


def run_evaluation(predictions_file, split: str) -> str:
    """Load predictions.json and score them against the selected split."""
    examples = _splits.get(split, [])
    if not examples:
        return f"Split '{split}' chưa được tải. Hãy chạy `python -m data.download` trước."

    if predictions_file is None:
        predictions = generate_rasa_predictions(
            examples,
            rasa_url=RASA_REST_URL,
            sender=RASA_SENDER,
            timeout=RASA_TIMEOUT,
            concurrency=4,
        )
        note = "> **Lưu ý:** Không có file predictions → đang chạy **Rasa knowledge evaluation** từ vector database.\n\n"
    else:
        try:
            predictions_path = getattr(predictions_file, "name", predictions_file)
            with open(predictions_path, encoding="utf-8") as f:
                predictions = json.load(f)
            note = f"> Đã load **{len(predictions):,}** predictions từ file.\n\n"
        except Exception as exc:
            return f"Lỗi khi đọc file predictions: {exc}"

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
    for key, val in metrics.items():
        if key.startswith("recall@"):
            lines.append(f"| {key} | {val:.2f}% |")

    return "\n".join(lines)


def generate_predictions_file(examples_file, concurrency: int) -> tuple[str | None, str]:
    """Generate a predictions.json file from an uploaded processed split JSON."""
    if examples_file is None:
        return None, "Vui lòng upload một file JSON có format như `data/processed/test.json`."

    try:
        examples_path = Path(getattr(examples_file, "name", examples_file))
        with open(examples_path, encoding="utf-8") as f:
            examples = json.load(f)
    except Exception as exc:
        return None, f"Lỗi khi đọc file examples: {exc}"

    if not isinstance(examples, list):
        return None, "File JSON không hợp lệ: nội dung phải là một list các examples."

    required_keys = {"id", "question", "answers", "is_impossible"}
    invalid_index = next(
        (
            idx for idx, item in enumerate(examples)
            if not isinstance(item, dict) or not required_keys.issubset(item.keys())
        ),
        None,
    )
    if invalid_index is not None:
        return (
            None,
            "File JSON không đúng format processed split. "
            f"Example tại vị trí {invalid_index} thiếu các key bắt buộc: {sorted(required_keys)}.",
        )

    predictions = generate_rasa_predictions(
        examples,
        rasa_url=RASA_REST_URL,
        sender=RASA_SENDER,
        timeout=RASA_TIMEOUT,
        concurrency=int(concurrency),
    )

    temp_dir = Path(tempfile.mkdtemp(prefix="rasa-preds-"))
    output_path = temp_dir / "predictions.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    status = (
        f"Đã tạo `predictions.json` cho {len(predictions):,} examples. "
        "Bạn có thể tải file này xuống và upload sang tab evaluation."
    )
    return str(output_path), status


def _normalize_history(history: list[object] | None) -> list[dict[str, str]]:
    """Convert any prior chatbot state into the messages format this Gradio build expects."""
    normalized: list[dict[str, str]] = []
    for item in history or []:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and isinstance(content, str):
                normalized.append({"role": role, "content": content})
            continue

        if isinstance(item, (list, tuple)) and len(item) == 2:
            user_text, assistant_text = item
            if isinstance(user_text, str) and user_text:
                normalized.append({"role": "user", "content": user_text})
            if isinstance(assistant_text, str) and assistant_text:
                normalized.append({"role": "assistant", "content": assistant_text})

    return normalized


def stream_response(message: str, history: list[object] | None):
    """Stream assistant text back to Gradio while preserving chat history."""
    history = _normalize_history(history)
    if not message or not message.strip():
        yield history, ""
        return

    LOGGER.info("Sending message to Rasa: %s", message)
    LOGGER.info("Current chatbot history before request: %s", history)

    try:
        rasa_payload = call_rasa(message, rasa_url=RASA_REST_URL, sender=RASA_SENDER, timeout=RASA_TIMEOUT)
    except requests.RequestException as exc:
        LOGGER.exception("Rasa request failed for message: %s", message)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"Failed to reach Rasa: {exc}"})
        yield history[:], ""
        return

    LOGGER.info(
        "Received raw response from Rasa: %s",
        json.dumps(rasa_payload, ensure_ascii=False),
    )
    assistant_text = format_response(rasa_payload)
    LOGGER.info("Formatted assistant response: %s", assistant_text)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})
    streamed = ""

    for word in assistant_text.split():
        streamed = f"{streamed} {word}".strip()
        history[-1]["content"] = streamed
        yield history[:], ""

    if streamed != assistant_text:
        history[-1]["content"] = assistant_text
        yield history[:], ""


def new_conversation():
    return [], ""


def build_app() -> gr.Blocks:
    _load_data()

    with gr.Blocks(title="UIT-ViQuAD2.0 Explorer & Rasa UI") as app:
        gr.Markdown(
            "# UIT-ViQuAD2.0 Explorer & Rasa UI\n"
            "Dataset explorer, evaluation tools, and a live chat tab wired to the Rasa REST API.\n"
            "> Chạy `python -m data.download` để tải dataset trước khi sử dụng."
        )

        with gr.Tabs():
            with gr.Tab("Khám phá Dataset"):
                gr.Markdown("### Tìm kiếm & xem examples")
                with gr.Row():
                    keyword_input = gr.Textbox(
                        label="Từ khoá (câu hỏi hoặc tiêu đề)",
                        placeholder="Ví dụ: Hà Nội",
                    )
                    split_dropdown = gr.Dropdown(
                        choices=["all", "train", "validation", "test"],
                        value="all",
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

            with gr.Tab("Tạo Predictions"):
                gr.Markdown(
                    "### Upload processed split JSON và sinh file `predictions.json` từ Rasa\n"
                    "File đầu vào phải có format như `data/processed/test.json`. "
                    "Kết quả đầu ra là file JSON có thể upload trực tiếp ở tab evaluation."
                )
                with gr.Row():
                    examples_file = gr.File(label="File examples JSON", file_types=[".json"])
                    concurrency_slider = gr.Slider(
                        1,
                        16,
                        value=4,
                        step=1,
                        label="Số request Rasa chạy đồng thời",
                    )
                generate_btn = gr.Button("Tạo predictions.json", variant="primary")
                predictions_file_output = gr.File(label="predictions.json")
                predictions_status = gr.Markdown()
                generate_btn.click(
                    generate_predictions_file,
                    inputs=[examples_file, concurrency_slider],
                    outputs=[predictions_file_output, predictions_status],
                )

            with gr.Tab("Chat với Rasa"):
                gr.Markdown(
                    "### Gửi tin nhắn đến Rasa REST channel\n"
                    "Tab này forward trực tiếp message đến `RASA_REST_URL` và stream câu trả lời theo từng từ."
                )
                chatbot = gr.Chatbot(
                    elem_id="rasa-chatbot",
                    height=520,
                )
                with gr.Row():
                    message_input = gr.Textbox(
                        show_label=False,
                        placeholder="Ask the bot...",
                        lines=1,
                    )
                    send_btn = gr.Button("Send")
                    reset_btn = gr.Button("New conversation")

                message_input.submit(stream_response, [message_input, chatbot], [chatbot, message_input])
                send_btn.click(stream_response, [message_input, chatbot], [chatbot, message_input])
                reset_btn.click(new_conversation, outputs=[chatbot, message_input])

    return app


if __name__ == "__main__":
    build_app().launch(
        share=False,
        server_name="0.0.0.0",
        server_port=int(os.getenv("UI_PORT", "7860")),
        theme=gr.themes.Soft(),
        css=APP_CSS,
    )
