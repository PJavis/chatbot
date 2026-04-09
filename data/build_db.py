"""
Script đọc corpus.json và embed vào ChromaDB sử dụng mô hình mã nguồn mở.
Usage:
    uv run python data/build_db.py
"""

import json
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

PROCESSED_DIR = Path(__file__).parent / "processed"
CHROMA_PATH = Path(__file__).parent / "chroma_db"

def build_vector_db():
    corpus_file = PROCESSED_DIR / "corpus.json"
    
    if not corpus_file.exists():
        print(f"Lỗi: Không tìm thấy {corpus_file}. Hãy chạy data/download.py trước!")
        return

    print("1. Đang đọc dữ liệu từ corpus.json...")
    with open(corpus_file, "r", encoding="utf-8") as f:
        corpus_data = json.load(f)
    
    print(f"   -> Đã load {len(corpus_data):,} đoạn văn.")

    print("2. Chuyển đổi sang format Langchain...")
    documents = []
    for item in corpus_data:
        doc = Document(
            page_content=item["context"],
            metadata={
                "passage_id": item["passage_id"],
                "title": item["title"]
            }
        )
        documents.append(doc)

    print("3. Đang gọi API để Embedding và lưu vào ChromaDB...")
    print("   (Quá trình này sử dụng HuggingFace chạy nội bộ cục bộ, lần đầu sẽ tải model mất một chút thời gian...)")
    
    # Khởi tạo model biến chữ thành vector của HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert")
    
    # Tạo và lưu DB cục bộ
    db = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings, 
        persist_directory=str(CHROMA_PATH)
    )
    
    print(f"====================================")
    print(f"HOÀN TẤT! ChromaDB đã lưu tại: {CHROMA_PATH}")
    print(f"====================================")

if __name__ == "__main__":
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    build_vector_db()