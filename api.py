from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from rag_pipeline import answer_question

# Khởi tạo app FastAPI
app = FastAPI(
    title="RAG Chatbot API",
    description="API tìm kiếm tài liệu và trả lời câu hỏi bằng AI cho RASA"
)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

@app.post("/api/ask", response_model=QueryResponse)
async def ask_rag_system(request: QueryRequest):
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Câu hỏi không được để trống")
            
        result = answer_question(request.question)
        return QueryResponse(answer=result)
        
    except Exception as e:
        print(f"Lỗi hệ thống RAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "RAG API đang hoạt động!"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
