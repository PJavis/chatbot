# Vietnamese Chatbot – RAG trên UIT-ViQuAD2.0

Dự án xây dựng hệ thống **RAG (Retrieval-Augmented Generation)** cho tiếng Việt, lấy dataset **UIT-ViQuAD2.0** làm bộ dữ liệu huấn luyện và đánh giá.

Phase hiện tại tập trung vào 3 phần:
- **Thu thập & làm sạch dataset** – tải UIT-ViQuAD2.0 từ HuggingFace, chuẩn hoá văn bản
- **Đánh giá RAG** – tính các metrics chuẩn (EM, F1, BLEU) và viết test cases thủ công
- **Demo** – Gradio app để khám phá dataset và chạy evaluation tương tác

---

## Yêu cầu

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (trình quản lý package)

---

## Cài đặt

```bash
# Clone repo
git clone <repo-url>
cd chatbot

# Cài dependencies
uv sync
```

---

## Cấu trúc thư mục

```
chatbot/
├── data/
│   ├── __init__.py          # Hàm tiện ích: load_corpus(), load_split()
│   ├── download.py          # Script tải & làm sạch dataset
│   ├── raw/                 # JSON gốc từ HuggingFace (sau khi download)
│   └── processed/           # Dữ liệu đã xử lý (corpus + splits)
├── evaluation/
│   ├── metrics.py           # Các metrics: EM, F1, BLEU, Recall@k
│   ├── test_cases.py        # 10 test cases thủ công + hàm run_test_cases()
│   └── evaluate.py          # CLI chạy evaluation trên file predictions
├── demo/
│   └── app.py               # Gradio demo (3 tab)
└── main.py                  # Entry point tổng hợp
```

---

## Sử dụng

### Bước 1 – Tải & làm sạch dataset

```bash
python main.py download
# hoặc
python -m data.download
```

Script sẽ:
1. Tải UIT-ViQuAD2.0 từ HuggingFace (`taidng/UIT-ViQuAD2.0`)
2. Lưu JSON thô vào `data/raw/`
3. Chuẩn hoá văn bản (NFC unicode, thu gọn khoảng trắng)
4. Tổng hợp corpus 138 đoạn văn duy nhất
5. Lưu dữ liệu đã xử lý vào `data/processed/`

**Kết quả mong đợi:**

```
train      : 28,500 mẫu (19,875 có đáp án / 8,625 không có đáp án)
validation :  3,810 mẫu
test       :  7,290 mẫu
Corpus     :    138 đoạn văn
```

---

### Bước 2 – Chạy Gradio Demo

```bash
python main.py demo
# hoặc
python demo/app.py
```

Mở trình duyệt tại `http://localhost:7860`. Demo gồm 3 tab:

| Tab | Chức năng |
|-----|-----------|
| **Khám phá Dataset** | Tìm kiếm examples theo từ khoá, lọc theo split, xem thống kê |
| **Tính Metrics** | Nhập prediction + ground truths, xem ngay EM / F1 / BLEU |
| **Chạy Evaluation** | Upload file `predictions.json`, đánh giá trên validation hoặc test |

---

### Bước 3 – Đánh giá hệ thống RAG

#### Chuẩn bị file predictions

Hệ thống RAG của bạn cần tạo ra một file JSON theo định dạng:

```json
{
  "56be85543aeaaa14008c9063": "câu trả lời 1",
  "56be85543aeaaa14008c9064": "câu trả lời 2"
}
```

Trong đó key là `id` của mỗi example trong dataset.

#### Chạy evaluation qua CLI

```bash
# Đánh giá trên validation (oracle baseline, để xem kết quả mẫu)
python main.py evaluate

# Đánh giá predictions thật của hệ thống RAG
python -m evaluation.evaluate \
    --predictions path/to/predictions.json \
    --split validation

# Đánh giá trên test set
python -m evaluation.evaluate \
    --predictions path/to/predictions.json \
    --split test \
    --output results/test_metrics.json
```

#### Đánh giá có kèm Retrieval metrics

Nếu hệ thống RAG trả về danh sách passage đã retrieve, tạo thêm file `retrieved_ids.json`:

```json
{
  "example_id": ["p00012", "p00034", "p00007"]
}
```

Rồi chạy:

```bash
python -m evaluation.evaluate \
    --predictions predictions.json \
    --retrieved_ids retrieved_ids.json \
    --split validation
```

Kết quả sẽ bao gồm thêm **Recall@1**, **Recall@3**, **Recall@5**.

---

### Bước 4 – Chạy test cases thủ công

```bash
python main.py test
```

10 test cases thủ công trong `evaluation/test_cases.py` bao gồm các dạng câu hỏi:

| Danh mục | Ví dụ |
|----------|-------|
| `factoid` | "Sông Hồng bắt nguồn từ đâu?" |
| `numerical` | "Chiều dài sông Mekong là bao nhiêu km?" |
| `historical` | "Hồ Chí Minh sinh năm nào?" |
| `comparison` | "Thành phố nào đông dân hơn: Hà Nội hay Hồ Chí Minh?" |
| `definition` | "RAG là gì?" |
| `unanswerable` | Câu hỏi không có đáp án trong context |
| `ambiguous` | Câu hỏi mơ hồ để test độ robust |

Để tích hợp hệ thống RAG vào test cases, truyền vào hàm `run_test_cases()` bất kỳ callable nào nhận `(question, context)` và trả về `str`:

```python
from evaluation.test_cases import run_test_cases

def my_rag(question: str, context: str | None) -> str:
    # ... gọi hệ thống RAG của bạn
    return answer

results = run_test_cases(my_rag, verbose=True)
print(f"Pass rate: {results['pass_rate']}%")
```

---

## Chuẩn hoá văn bản (`_normalize`)

Trước khi so sánh bất kỳ cặp (prediction, ground truth) nào, cả hai đều được đưa qua pipeline chuẩn hoá sau:

```
Input:  "  Năm  1831,  "
   │
   ├─ 1. NFC unicode      →  "  Năm  1831,  "   (gộp ký tự tổ hợp, ví dụ a + ̀ → à)
   ├─ 2. Lowercase        →  "  năm  1831,  "
   ├─ 3. Bỏ dấu câu       →  "  năm  1831   "   (xoá , . ! ? : ; " ' ...)
   └─ 4. Thu gọn spaces   →  "năm 1831"

Output: "năm 1831"
```

**Tại sao cần bước này?**

Không có chuẩn hoá, các cặp sau đây sẽ bị đánh giá **sai** dù đúng về nghĩa:

| Prediction | Ground truth | Không chuẩn hoá | Sau chuẩn hoá |
|------------|-------------|-----------------|---------------|
| `"năm 1831"` | `"Năm 1831"` | ❌ khác nhau | ✅ giống nhau |
| `"1831,"` | `"1831"` | ❌ khác nhau | ✅ giống nhau |
| `"năm  1831"` | `"năm 1831"` | ❌ khác nhau | ✅ giống nhau |
| `"Hà\u0300 Nội"` | `"Hà Nội"` | ❌ khác nhau (unicode) | ✅ giống nhau |

> **Lưu ý:** Bước này **không** bỏ dấu thanh tiếng Việt (à, á, ả...). Chỉ bỏ dấu câu Latin như `,`, `.`, `"`. Tiếng Việt phân biệt nghĩa bằng dấu thanh nên không được bỏ.

---

## Logic đánh giá từng metric

### 1. Exact Match (EM)

**Ý nghĩa:** Prediction có khớp hoàn toàn với đáp án không?

**Cách tính:**
```
EM = 1  nếu  normalize(prediction) == normalize(bất kỳ ground truth nào)
EM = 0  nếu  không khớp với ground truth nào
```

**Ví dụ:**
```
Ground truths: ["năm 1831", "1831", "Minh Mạng 1831"]

prediction = "Năm 1831,"   → normalize → "năm 1831"  → khớp gt[0] → EM = 1
prediction = "1831"        → normalize → "1831"       → khớp gt[1] → EM = 1
prediction = "khoảng 1831" → normalize → "khoảng 1831" → không khớp → EM = 0
```

**Điểm yếu:** Rất nghiêm khắc. Trả lời đúng nhưng thêm 1 từ → EM = 0. Vì vậy cần dùng kèm F1.

---

### 2. Token F1

**Ý nghĩa:** Bao nhiêu % từ của prediction trùng với ground truth?

**Cách tính:** Tính trên tập token (từ), lấy F1 cao nhất so với tất cả ground truths.

```
Token overlap = tập token chung giữa prediction và ground truth (đếm theo tần suất)

Precision = |overlap| / |prediction tokens|   (prediction có bao nhiêu từ đúng)
Recall    = |overlap| / |ground truth tokens| (ground truth được bao phủ bao nhiêu)
F1        = 2 × Precision × Recall / (Precision + Recall)
```

**Ví dụ chi tiết:**
```
Ground truth : "miền Bắc Việt Nam"   → tokens: [miền, bắc, việt, nam]
Prediction   : "ở miền Bắc"          → tokens: [ở, miền, bắc]

Overlap  = {miền, bắc}  → |overlap| = 2
Precision = 2 / 3 = 0.667
Recall    = 2 / 4 = 0.500
F1        = 2 × 0.667 × 0.500 / (0.667 + 0.500) = 0.571
```

**Trường hợp đặc biệt:**
```
prediction = ""  và  ground truth = ""  → F1 = 1.0  (cùng rỗng = đúng)
prediction = ""  và  ground truth ≠ ""  → F1 = 0.0
prediction ≠ ""  và  ground truth = ""  → F1 = 0.0
```

---

### 3. BLEU (1..4)

**Ý nghĩa:** Prediction có chứa cụm từ (n-gram) của ground truth không?

**Khác F1 ở chỗ:** BLEU xét n-gram liên tiếp (thứ tự từ quan trọng), F1 chỉ xét tập từ (thứ tự không quan trọng).

```
BLEU-1: đếm unigram (từ đơn)       trùng nhau
BLEU-2: đếm bigram  (cặp 2 từ)     trùng nhau
BLEU-3: đếm trigram (cụm 3 từ)     trùng nhau
BLEU-4: đếm 4-gram  (cụm 4 từ)     trùng nhau
```

**Ví dụ:**
```
Ground truth: "bắt nguồn từ tỉnh Vân Nam"
Prediction  : "bắt nguồn từ Trung Quốc"

Unigram trùng : bắt, nguồn, từ        → BLEU-1 cao
Bigram trùng  : "bắt nguồn", "nguồn từ" → BLEU-2 vừa
Trigram trùng : "bắt nguồn từ"          → BLEU-3 thấp
4-gram trùng  : (không có)              → BLEU-4 = 0
```

**Tokenise cho tiếng Việt:** Dùng `sacrebleu` với `tokenize="char"` (ký tự), vì tiếng Việt không có khoảng trắng rõ ràng giữa âm tiết.

---

### 4. Unanswerable Accuracy

**Ý nghĩa:** Với các câu hỏi `is_impossible=True` (không có đáp án trong context), hệ thống có nhận ra được không?

**Cách tính:**
```
Đúng khi prediction là:  ""  hoặc  "không có thông tin"  hoặc  "không biết"  hoặc  "n/a"
Sai  khi prediction là bất kỳ câu trả lời nào khác
```

> Câu hỏi `is_impossible` **không** tính vào EM/F1/BLEU – chúng được tính riêng qua metric này.

---

### 5. Retrieval Recall@k

**Ý nghĩa:** Trong top-k đoạn văn được retrieve, có đoạn nào chứa đáp án không?

```
Recall@k = 1  nếu  gold_passage_id ∈ top-k retrieved passage IDs
Recall@k = 0  nếu  không có
```

**Ví dụ:**
```
gold_passage_id = "p00012"
retrieved_ids   = ["p00034", "p00012", "p00007", "p00019", "p00055"]

Recall@1 = 0  (p00012 không ở vị trí 1)
Recall@2 = 1  (p00012 ở vị trí 2, nằm trong top-2)
Recall@5 = 1  (p00012 nằm trong top-5)
```

**Tại sao quan trọng?** Nếu Recall@5 thấp → hệ thống retrieve kém → dù reader tốt cũng không có context đúng để trả lời.

---

### Luồng đánh giá tổng thể (`evaluate_predictions`)

```
Với mỗi example trong dataset:
   │
   ├── is_impossible = True?
   │      ├── Có → tính Unanswerable Accuracy riêng
   │      │        ghi EM=0, F1=0, BLEU=0 (không tính vào trung bình QA)
   │      └── Không → tính EM, F1, BLEU-1, BLEU-4
   │
   └── Có retrieved_ids? → tính Recall@1, @3, @5
   
Kết quả cuối = macro-average trên toàn bộ examples (×100 để ra %)
```

---

## Metrics được hỗ trợ

| Metric | Phạm vi | Ý nghĩa ngắn gọn |
|--------|---------|------------------|
| **Exact Match (EM)** | 0–100% | Tỷ lệ prediction khớp hoàn toàn với ground truth |
| **Token F1** | 0–100% | Tỷ lệ từ trùng nhau giữa prediction và ground truth |
| **BLEU-1** | 0–100% | Tỷ lệ unigram (từ đơn) trùng nhau |
| **BLEU-4** | 0–100% | Tỷ lệ 4-gram trùng nhau (yêu cầu câu trả lời sát nghĩa) |
| **Unanswerable Acc.** | 0–100% | Tỷ lệ nhận biết đúng câu hỏi không có đáp án |
| **Retrieval Recall@k** | 0–100% | Tỷ lệ lấy đúng đoạn văn nguồn trong top-k |

---

## Dataset – UIT-ViQuAD2.0

- **Nguồn:** [taidng/UIT-ViQuAD2.0](https://huggingface.co/datasets/taidng/UIT-ViQuAD2.0) trên HuggingFace
- **Định dạng:** SQuAD 2.0 (extractive QA, tiếng Việt)
- **Ngôn ngữ:** Tiếng Việt
- **Nguồn văn bản:** 138 bài Wikipedia tiếng Việt
- **Đặc điểm:** ~30% câu hỏi không có đáp án (`is_impossible=True`)

Cấu trúc mỗi example:

```python
{
    "id": "56be85543aeaaa14008c9063",
    "passage_id": "p00012",          # ID đoạn văn trong corpus
    "title": "Hà Nội",               # Tiêu đề bài Wikipedia
    "question": "Hà Nội nằm ở đâu?",
    "is_impossible": False,
    "answers": [
        {"text": "miền Bắc Việt Nam", "answer_start": 42}
    ]
}
```

---

## Rasa Chatbot

Thư mục `rasa/` chứa một chatbot hỗ trợ nội bộ bằng **tiếng Việt**, dùng Rasa làm dialog engine. Bot được thiết kế để trả lời câu hỏi về chính sách HR và tài liệu kỹ thuật thông qua RAG API.

### Kiến trúc

```
Giao diện (Gradio / REST)
         ↓
Rasa Dialog Engine     (port mặc định)
         ↓
Action Server          (port 5055)
         ↓
RAG API                (port 8000)
         ↓
Knowledge Base         (fake: 12 tài liệu / real: thay bằng FAISS + LLM)
```

### Cấu trúc thư mục

```
rasa/
├── config.yml          # NLU pipeline + policies
├── domain.yml          # Intents, entities, slots, responses, actions
├── endpoints.yml       # Action server, tracker store, event broker
├── credentials.yml     # Kênh kết nối (REST, Socket.IO, ...)
├── fake_rag_api.py     # FastAPI mock RAG (để test không cần LLM thật)
├── data/
│   ├── nlu.yml         # Training data (~96 examples, tiếng Việt)
│   ├── stories.yml     # Conversation stories (7 flows)
│   └── rules.yml       # Hard rules (7 rules)
└── actions/
    └── actions.py      # Custom actions
```

### NLU Pipeline (`config.yml`)

- **Tokenizer:** WhitespaceTokenizer
- **Features:** CountVectors (word + char n-gram 1–4) + BERT multilingual (`bert-base-multilingual-cased`)
- **Classifier:** DIETClassifier (100 epochs) + FallbackClassifier (ngưỡng 70%)
- **Policies:** TEDPolicy, RulePolicy, MemoizationPolicy, UnexpecTEDIntentPolicy

### Intents & Entities (`domain.yml`)

| Intent | Mô tả |
|---|---|
| `greet` / `goodbye` | Chào hỏi / tạm biệt |
| `ask_information` | Câu hỏi HR / chính sách |
| `ask_technical` | Câu hỏi kỹ thuật |
| `ask_faq` | Câu hỏi FAQ |
| `affirm` / `deny` | Xác nhận / phủ nhận |
| `out_of_scope` | Ngoài phạm vi |
| `ask_human_handoff` | Yêu cầu gặp nhân viên hỗ trợ |

**Entities:** `topic`, `product`, `error_code`

**Slots:** `user_query`, `rag_answer`, `conversation_context` (lưu 3 lượt gần nhất)

### Custom Actions (`actions/actions.py`)

| Action | Mô tả |
|---|---|
| `ActionRagSearch` | Enrich query bằng entity → gọi RAG API → trả kết quả + nguồn tài liệu |
| `ActionSetContext` | Reset slot `conversation_context` khi đổi chủ đề |
| `ActionHumanHandoff` | Gửi webhook đến team support kèm lịch sử hội thoại |

Cấu hình qua environment variables:

```bash
RAG_API_URL=http://localhost:8000   # mặc định
RAG_TIMEOUT=15                      # giây
RAG_TOP_K=3
SUPPORT_WEBHOOK_URL=http://localhost:9000
```

### Luồng xử lý chính

```
User gửi câu hỏi
       ↓
DIETClassifier phân loại intent + trích entity
       ↓
RulePolicy khớp rule:  ask_* → utter_searching → ActionRagSearch
       ↓
ActionRagSearch: enrich query + gọi RAG API
       ↓
Trả về câu trả lời + nguồn tài liệu (📎 Nguồn: ...)
```

### Fake RAG API (`fake_rag_api.py`)

FastAPI trên port 8000, dùng để test mà không cần LLM/FAISS thật. Có 12 tài liệu về: nghỉ phép, WFH, VPN, mật khẩu, lương thưởng, onboarding, GitLab CI/CD, bảo mật, v.v.

```bash
# Chạy fake RAG API
uv run python rasa/fake_rag_api.py

# Chạy action server
uv run rasa run actions --actions actions --port 5055

# Chạy Rasa
uv run rasa run --enable-api
```

---

## Kiến trúc tổng thể (roadmap)

```
Câu hỏi người dùng
        │
        ▼
  ┌─────────────┐
  │   RASA NLU  │  ← Intent / Entity recognition
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐     ┌────────────┐
  │  Retriever  │────▶│  Corpus    │  ← UIT-ViQuAD2.0 passages
  └──────┬──────┘     └────────────┘
         │ top-k passages
         ▼
  ┌─────────────┐
  │    Reader   │  ← LLM / Extractive QA model
  └──────┬──────┘
         │
         ▼
     Câu trả lời
```

Phase hiện tại (phần được implement trong repo này) bao gồm **Data** và **Evaluation**. Các phần **Retriever** và **Reader** sẽ được tích hợp ở phase tiếp theo.
