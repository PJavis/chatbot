"""
Các metrics đánh giá RAG trên UIT-ViQuAD2.0.

Metrics được implement theo chuẩn SQuAD 2.0 (đã điều chỉnh cho tiếng Việt):
  - Exact Match (EM)
  - Token-level F1
  - BLEU-1..4 (via sacrebleu)
  - Retrieval Recall@k  (khi có ground-truth passage_id)
"""

from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter

import sacrebleu


# ---------------------------------------------------------------------------
# Text normalisation (dùng chung)
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """
    Chuẩn hoá để so sánh:
    - NFC unicode
    - Chữ thường
    - Bỏ dấu câu
    - Thu gọn khoảng trắng
    """
    text = unicodedata.normalize("NFC", text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> list[str]:
    return _normalize(text).split()


# ---------------------------------------------------------------------------
# Exact Match
# ---------------------------------------------------------------------------

def exact_match(prediction: str, ground_truths: list[str]) -> int:
    """
    EM = 1 nếu prediction (sau chuẩn hoá) khớp với BẤT KỲ đáp án nào.

    Returns:
        1 hoặc 0
    """
    pred_norm = _normalize(prediction)
    return int(any(pred_norm == _normalize(gt) for gt in ground_truths))


# ---------------------------------------------------------------------------
# Token-level F1
# ---------------------------------------------------------------------------

def token_f1(prediction: str, ground_truths: list[str]) -> float:
    """
    F1 token-level: lấy F1 cao nhất so với tất cả ground truths.

    Xử lý đặc biệt: khi prediction hoặc ground truth là chuỗi rỗng.
    """
    pred_tokens = _tokenize(prediction)

    best_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = _tokenize(gt)

        # Cả hai đều rỗng → hoàn hảo
        if not pred_tokens and not gt_tokens:
            return 1.0
        # Một trong hai rỗng → 0
        if not pred_tokens or not gt_tokens:
            continue

        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_common = sum(common.values())
        if num_common == 0:
            continue

        precision = num_common / len(pred_tokens)
        recall = num_common / len(gt_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)

    return best_f1


# ---------------------------------------------------------------------------
# BLEU
# ---------------------------------------------------------------------------

def bleu_score(prediction: str, ground_truths: list[str]) -> dict[str, float]:
    """
    BLEU-1..4 dùng sacrebleu (tokenised bằng space, phù hợp tiếng Việt).

    Returns:
        {"bleu1": ..., "bleu2": ..., "bleu3": ..., "bleu4": ...}
    """
    # sacrebleu nhận list[hypothesis] và list[list[reference]]
    hypothesis = [prediction]
    references = [[gt for gt in ground_truths]]

    result = {}
    for n in range(1, 5):
        bleu = sacrebleu.corpus_bleu(
            hypothesis,
            references,
            max_ngram_order=n,
            tokenize="char",   # char-level tokenise tốt hơn cho tiếng Việt
        )
        result[f"bleu{n}"] = round(bleu.score, 4)
    return result


# ---------------------------------------------------------------------------
# Retrieval Recall@k
# ---------------------------------------------------------------------------

def retrieval_recall_at_k(
    retrieved_ids: list[str],
    gold_passage_id: str,
    k: int,
) -> int:
    """
    Recall@k = 1 nếu gold_passage_id nằm trong top-k kết quả retrieve.

    retrieved_ids: danh sách passage_id đã xếp hạng (index 0 = cao nhất)
    k: ngưỡng
    """
    return int(gold_passage_id in retrieved_ids[:k])


# ---------------------------------------------------------------------------
# Aggregate evaluation
# ---------------------------------------------------------------------------

def evaluate_predictions(
    examples: list[dict],
    predictions: dict[str, str],
    retrieved_ids_map: dict[str, list[str]] | None = None,
    top_k_values: tuple[int, ...] = (1, 3, 5),
) -> dict:
    """
    Đánh giá một bộ predictions trên tập examples.

    Args:
        examples: list QA examples (mỗi item có 'id', 'answers', 'is_impossible', 'passage_id')
        predictions: dict {example_id → predicted_answer_string}
        retrieved_ids_map: dict {example_id → [ranked passage_ids]} (tuỳ chọn)
        top_k_values: các giá trị k để tính Recall@k

    Returns:
        dict với các metrics tổng hợp (macro-average)
    """
    em_scores, f1_scores = [], []
    bleu1_scores, bleu4_scores = [], []
    recall_at_k: dict[int, list[int]] = {k: [] for k in top_k_values}

    n_impossible_correct = 0
    n_impossible = 0

    for ex in examples:
        ex_id = ex["id"]
        pred = predictions.get(ex_id, "")

        gold_texts = [a["text"] for a in ex.get("answers", [])]
        is_impossible = ex.get("is_impossible", False)

        if is_impossible:
            n_impossible += 1
            # Đúng khi prediction là chuỗi rỗng hoặc "không có thông tin"
            if not pred.strip() or _normalize(pred) in (
                "", "không có thông tin", "không biết", "n/a"
            ):
                n_impossible_correct += 1
            em_scores.append(0)
            f1_scores.append(0.0)
            bleu1_scores.append(0.0)
            bleu4_scores.append(0.0)
        else:
            if not gold_texts:
                continue
            em_scores.append(exact_match(pred, gold_texts))
            f1_scores.append(token_f1(pred, gold_texts))
            b = bleu_score(pred, gold_texts)
            bleu1_scores.append(b["bleu1"])
            bleu4_scores.append(b["bleu4"])

        # Retrieval
        if retrieved_ids_map and ex_id in retrieved_ids_map:
            gold_pid = ex.get("passage_id", "")
            r_ids = retrieved_ids_map[ex_id]
            for k in top_k_values:
                recall_at_k[k].append(retrieval_recall_at_k(r_ids, gold_pid, k))

    def avg(lst):
        return round(sum(lst) / len(lst) * 100, 2) if lst else 0.0

    result = {
        "num_examples": len(em_scores),
        "exact_match": avg(em_scores),
        "token_f1": avg(f1_scores),
        "bleu1": avg(bleu1_scores),
        "bleu4": avg(bleu4_scores),
    }
    if n_impossible > 0:
        result["unanswerable_accuracy"] = round(
            n_impossible_correct / n_impossible * 100, 2
        )
    for k in top_k_values:
        if recall_at_k[k]:
            result[f"recall@{k}"] = avg(recall_at_k[k])

    return result
