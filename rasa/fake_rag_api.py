"""
Fake RAG API - dùng để test RASA mà không cần setup LLM/FAISS thật
Chạy: python3 fake_rag_api.py
URL:  http://localhost:8000
"""

import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI(title="Fake RAG API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Fake knowledge base ──────────────────────────────────────────
FAKE_KB = [
    (["nghỉ phép", "xin nghỉ", "ngày phép"],
     "Nhân viên được 12 ngày nghỉ phép có lương/năm. Đăng ký trên HRM trước 3 ngày làm việc và cần quản lý phê duyệt.",
     "FAQ - Chính sách nhân sự"),

    (["remote", "từ xa", "wfh", "làm nhà"],
     "Nhân viên được làm việc từ xa tối đa 2 ngày/tuần sau thử việc. Thông báo quản lý trước 1 ngày.",
     "FAQ - Chính sách WFH"),

    (["mật khẩu", "password", "reset", "quên"],
     "Reset mật khẩu tại intranet.company.vn/reset-password. Nếu không nhận được email sau 5 phút, liên hệ #it-support trên Slack.",
     "Tài liệu kỹ thuật - IT Support"),

    (["vpn", "mạng", "kết nối nội bộ"],
     "Tải Cisco AnyConnect tại intranet.company.vn/vpn. Server: vpn.company.vn. Dùng email + mật khẩu Windows.",
     "Tài liệu kỹ thuật - VPN"),

    (["lỗi", "error", "500", "timeout", "database"],
     "Kiểm tra VPN đã bật chưa. Restart service: sudo systemctl restart app-service. Vẫn lỗi → báo #db-support.",
     "Tài liệu kỹ thuật - Troubleshooting"),

    (["giờ làm việc", "mấy giờ", "thứ mấy", "làm việc"],
     "Làm việc Thứ 2 - Thứ 6, 8h00 - 17h30. Nghỉ trưa 12h - 13h30.",
     "FAQ - Nội quy"),

    (["phòng họp", "đặt phòng", "meeting"],
     "Đặt phòng họp qua Google Calendar → Rooms. Đặt trước 30 phút. Phòng >10 người đặt trước 1 ngày.",
     "FAQ - Tiện ích"),

    (["lương", "thưởng", "phúc lợi", "bảo hiểm"],
     "Bảo hiểm sức khỏe cao cấp, hỗ trợ ăn trưa 50k/ngày, hỗ trợ đào tạo 5tr/năm, team building hàng quý.",
     "FAQ - Phúc lợi"),

    (["onboarding", "nhân viên mới", "bắt đầu"],
     "Ngày 1: nhận máy + tài khoản. Tuần 1: gặp team + buddy. Tháng 1: hoàn thành LMS. Tháng 2: check-in HR.",
     "FAQ - Onboarding"),

    (["gitlab", "cicd", "ci/cd", "deploy", "pipeline"],
     "Pipeline cấu hình trong .gitlab-ci.yml. Stage: lint→test→build→deploy. Staging tự động, production cần Tech Lead duyệt.",
     "Tài liệu kỹ thuật - DevOps"),

    (["wifi", "mạng wifi", "internet"],
     "Wifi nhân viên: CorpNet-Staff (mật khẩu trong email onboarding). Wifi khách: CorpNet-Guest.",
     "FAQ - Tiện ích"),

    (["bảo mật", "security", "password policy"],
     "Mật khẩu ≥12 ký tự, gồm chữ hoa + số + ký tự đặc biệt. Đổi mỗi 90 ngày. Không cài phần mềm chưa được duyệt.",
     "Chính sách - Bảo mật"),
]

# ── Request/Response schema ──────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    context: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: list
    confidence: float

# ── Search logic ─────────────────────────────────────────────────
def fake_search(query: str):
    q = query.lower()
    for keywords, answer, source in FAKE_KB:
        if any(kw in q for kw in keywords):
            return answer, source, 0.85
    return (
        "Xin lỗi, tôi không tìm thấy thông tin liên quan. Vui lòng liên hệ HR hoặc IT Support.",
        "Hệ thống",
        0.1,
    )

# ── Endpoints ────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "Fake RAG API đang chạy"}

@app.get("/health")
def health():
    return {"status": "healthy", "mode": "fake", "docs_count": len(FAKE_KB)}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    answer, source, confidence = fake_search(req.query)
    print(f"[QUERY] '{req.query}' → conf={confidence} | {source}")
    return QueryResponse(answer=answer, sources=[source], confidence=confidence)

@app.post("/add_document")
def add_document(doc: dict):
    # Fake: không làm gì, chỉ báo OK
    return {"status": "ok", "message": "Fake API - document không được lưu thật"}

if __name__ == "__main__":
    print("=" * 50)
    print("  Fake RAG API")
    print("  URL   : http://localhost:8000")
    print("  Docs  : http://localhost:8000/docs")
    print("  Health: http://localhost:8000/health")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
