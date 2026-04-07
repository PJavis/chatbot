"""
Test cases thủ công để kiểm tra hệ thống RAG.

Mỗi test case gồm:
  - id          : định danh
  - question    : câu hỏi
  - context     : đoạn văn nguồn (có thể để trống → test open-domain)
  - answer      : đáp án kỳ vọng (list, để chứa nhiều dạng đúng)
  - category    : loại câu hỏi
  - is_impossible: True nếu không thể trả lời từ context

Dùng run_test_cases() để kiểm tra output của bất kỳ hàm RAG nào.
"""

from __future__ import annotations

from evaluation.metrics import exact_match, token_f1

# ---------------------------------------------------------------------------
# Định nghĩa test cases
# ---------------------------------------------------------------------------

TEST_CASES: list[dict] = [
    # --- Câu hỏi thực thể / sự kiện ---
    {
        "id": "tc_01",
        "category": "factoid",
        "question": "Hà Nội được thành lập năm nào?",
        "context": (
            "Hà Nội là thủ đô và thành phố trực thuộc trung ương lớn nhất Việt Nam. "
            "Tên gọi Hà Nội được đặt vào năm 1831 dưới thời vua Minh Mạng."
        ),
        "answer": ["1831", "năm 1831"],
        "is_impossible": False,
    },
    {
        "id": "tc_02",
        "category": "factoid",
        "question": "Sông Hồng bắt nguồn từ đâu?",
        "context": (
            "Sông Hồng, còn gọi là Hồng Hà, bắt nguồn từ tỉnh Vân Nam, Trung Quốc, "
            "chảy qua Lào Cai rồi đổ ra biển tại cửa Ba Lạt."
        ),
        "answer": ["tỉnh Vân Nam, Trung Quốc", "Vân Nam", "Trung Quốc"],
        "is_impossible": False,
    },
    {
        "id": "tc_03",
        "category": "factoid",
        "question": "Diện tích của Việt Nam là bao nhiêu?",
        "context": (
            "Việt Nam có tổng diện tích khoảng 331.212 km², "
            "xếp thứ 65 trên thế giới về diện tích."
        ),
        "answer": ["331.212 km²", "331.212", "khoảng 331.212 km²"],
        "is_impossible": False,
    },

    # --- Câu hỏi so sánh / tổng hợp ---
    {
        "id": "tc_04",
        "category": "comparison",
        "question": "Thành phố nào đông dân hơn: Hà Nội hay Hồ Chí Minh?",
        "context": (
            "Thành phố Hồ Chí Minh là thành phố đông dân nhất Việt Nam với khoảng "
            "9 triệu người, trong khi Hà Nội có khoảng 8 triệu người."
        ),
        "answer": ["Hồ Chí Minh", "Thành phố Hồ Chí Minh", "TP. Hồ Chí Minh"],
        "is_impossible": False,
    },

    # --- Câu hỏi không có đáp án trong context (is_impossible) ---
    {
        "id": "tc_05",
        "category": "unanswerable",
        "question": "GDP của Việt Nam năm 2050 là bao nhiêu?",
        "context": (
            "Việt Nam là quốc gia đang phát triển ở Đông Nam Á. "
            "Năm 2023, GDP Việt Nam đạt khoảng 430 tỷ USD."
        ),
        "answer": [],
        "is_impossible": True,
    },
    {
        "id": "tc_06",
        "category": "unanswerable",
        "question": "Ai là người phát minh ra điện thoại?",
        "context": (
            "Hà Nội có hệ thống viễn thông hiện đại với nhiều nhà cung cấp dịch vụ "
            "di động như Viettel, Mobifone và Vinaphone."
        ),
        "answer": [],
        "is_impossible": True,
    },

    # --- Câu hỏi định nghĩa ---
    {
        "id": "tc_07",
        "category": "definition",
        "question": "RAG là gì?",
        "context": (
            "Retrieval-Augmented Generation (RAG) là kỹ thuật kết hợp tìm kiếm thông tin "
            "từ cơ sở dữ liệu với mô hình ngôn ngữ lớn để tạo ra câu trả lời chính xác hơn."
        ),
        "answer": [
            "kỹ thuật kết hợp tìm kiếm thông tin từ cơ sở dữ liệu với mô hình ngôn ngữ lớn",
            "Retrieval-Augmented Generation",
        ],
        "is_impossible": False,
    },

    # --- Câu hỏi về số liệu ---
    {
        "id": "tc_08",
        "category": "numerical",
        "question": "Chiều dài của sông Mekong là bao nhiêu km?",
        "context": (
            "Sông Mekong là một trong những con sông dài nhất châu Á với chiều dài "
            "khoảng 4.880 km, chảy qua 6 quốc gia."
        ),
        "answer": ["4.880 km", "4.880", "khoảng 4.880 km"],
        "is_impossible": False,
    },

    # --- Câu hỏi nhân vật lịch sử ---
    {
        "id": "tc_09",
        "category": "historical",
        "question": "Hồ Chí Minh sinh năm nào?",
        "context": (
            "Chủ tịch Hồ Chí Minh, tên khai sinh Nguyễn Sinh Cung, sinh ngày 19 tháng 5 "
            "năm 1890 tại làng Kim Liên, huyện Nam Đàn, tỉnh Nghệ An."
        ),
        "answer": ["1890", "năm 1890", "19 tháng 5 năm 1890"],
        "is_impossible": False,
    },

    # --- Câu hỏi đa nghĩa / mơ hồ (kiểm tra độ robust) ---
    {
        "id": "tc_10",
        "category": "ambiguous",
        "question": "Ai là tác giả?",
        "context": (
            "Truyện Kiều là tác phẩm nổi tiếng nhất của Nguyễn Du, "
            "được sáng tác vào đầu thế kỷ XIX."
        ),
        "answer": ["Nguyễn Du"],
        "is_impossible": False,
    },
]


# ---------------------------------------------------------------------------
# Chạy test cases
# ---------------------------------------------------------------------------

def run_test_cases(
    rag_fn,
    test_cases: list[dict] | None = None,
    verbose: bool = True,
) -> dict:
    """
    Chạy danh sách test cases qua hàm RAG và báo cáo kết quả.

    Args:
        rag_fn: callable(question: str, context: str | None) → str
                Hàm RAG cần đánh giá. Nếu context=None → open-domain.
        test_cases: danh sách test cases (mặc định dùng TEST_CASES)
        verbose: in chi tiết từng case

    Returns:
        dict với {pass_rate, em, f1, results}
    """
    if test_cases is None:
        test_cases = TEST_CASES

    results = []
    for tc in test_cases:
        prediction = rag_fn(tc["question"], tc.get("context"))

        em = exact_match(prediction, tc["answer"]) if tc["answer"] else (
            1 if not prediction.strip() else 0
        )
        f1 = token_f1(prediction, tc["answer"]) if tc["answer"] else (
            1.0 if not prediction.strip() else 0.0
        )

        result = {
            "id": tc["id"],
            "category": tc["category"],
            "question": tc["question"],
            "expected": tc["answer"],
            "predicted": prediction,
            "is_impossible": tc["is_impossible"],
            "em": em,
            "f1": round(f1, 4),
            "pass": bool(em or f1 >= 0.5),
        }
        results.append(result)

        if verbose:
            status = "PASS" if result["pass"] else "FAIL"
            print(f"[{status}] {tc['id']} ({tc['category']})")
            print(f"  Q: {tc['question']}")
            print(f"  Expected : {tc['answer']}")
            print(f"  Predicted: {prediction}")
            print(f"  EM={em}  F1={f1:.3f}")
            print()

    pass_count = sum(1 for r in results if r["pass"])
    summary = {
        "total": len(results),
        "passed": pass_count,
        "pass_rate": round(pass_count / len(results) * 100, 1) if results else 0,
        "avg_em": round(sum(r["em"] for r in results) / len(results) * 100, 2),
        "avg_f1": round(sum(r["f1"] for r in results) / len(results) * 100, 2),
        "by_category": _breakdown_by_category(results),
        "results": results,
    }

    if verbose:
        print("=" * 50)
        print(f"TỔNG KẾT: {pass_count}/{len(results)} passed ({summary['pass_rate']}%)")
        print(f"  EM trung bình : {summary['avg_em']}%")
        print(f"  F1 trung bình : {summary['avg_f1']}%")
        print("  Theo danh mục:")
        for cat, stats in summary["by_category"].items():
            print(f"    {cat:15s}: {stats['passed']}/{stats['total']} passed")
        print("=" * 50)

    return summary


def _breakdown_by_category(results: list[dict]) -> dict:
    cats: dict[str, dict] = {}
    for r in results:
        c = r["category"]
        if c not in cats:
            cats[c] = {"total": 0, "passed": 0}
        cats[c]["total"] += 1
        if r["pass"]:
            cats[c]["passed"] += 1
    return cats
