"""
Script thực hiện luồng RAG: Retrieval-Augmented Generation
Nhận câu hỏi -> Tìm context trong ChromaDB -> Hỏi LLM -> Trả về kết quả.
"""

from pathlib import Path
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load API Key
load_dotenv()

CHROMA_PATH = Path(__file__).parent / "data" / "chroma_db"

def get_rag_chain():
    """Khởi tạo và trả về Langchain RAG chain."""
    
    # 1. QUAN TRỌNG: Phải dùng đúng model đã dùng để tạo DB
    embeddings = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert")
    db = Chroma(persist_directory=str(CHROMA_PATH), embedding_function=embeddings)
    
    # Tạo retriever (Lấy top 3 đoạn văn bản liên quan nhất)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # 2. Khởi tạo LLM 
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)


    # 3. Tạo Prompt Template dặn dò AI
    template = """Bạn là một trợ lý ảo thông minh. Hãy sử dụng các tài liệu ngữ cảnh (context) dưới đây để trả lời câu hỏi của người dùng.
    Nếu bạn không biết câu trả lời từ ngữ cảnh được cung cấp, hãy nói là "Tôi không có thông tin về vấn đề này", đừng cố bịa ra câu trả lời.
    Hãy trả lời ngắn gọn, súc tích và chính xác.

    Ngữ cảnh (Context):
    {context}

    Câu hỏi: {question}

    Trả lời:"""
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        # Nối nội dung các document tìm được thành một đoạn text dài
        return "\n\n".join(doc.page_content for doc in docs)

    # 4. Xây dựng Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def answer_question(question: str) -> str:
    """Hàm giao tiếp chính để gọi RAG."""
    chain = get_rag_chain()
    print(f"\n[?] Câu hỏi: {question}")
    print("... Đang tìm kiếm và suy nghĩ ...")
    response = chain.invoke(question)
    return response

if __name__ == "__main__":
    test_question = "California kề cận với đâu?"
    answer = answer_question(test_question)
    print(f"\n[=>] Trả lời: {answer}")
