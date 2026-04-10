"""
Entry point cho chatbot project.
"""

import sys

def cmd_download():
    from data.download import main
    main()

def cmd_build_db():
    # Gọi hàm build_vector_db từ file build_db.py bạn vừa tạo
    from data.build_db import build_vector_db
    build_vector_db()

def cmd_api():
    # Khởi động server FastAPI
    import uvicorn
    print("🚀 Đang khởi động RAG API Server...")
    # Lưu ý: "api:app" nghĩa là tìm biến 'app' trong file 'api.py'
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

def cmd_evaluate():
    from evaluation.evaluate import main
    sys.argv = ["evaluate", "--split", "validation", "--strategy", "first_answer"]
    main()

def cmd_demo():
    from demo.app import build_app
    app = build_app()
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)

def cmd_test():
    from evaluation.test_cases import run_test_cases
    print("Chạy manual test cases với dummy RAG (trả về chuỗi rỗng)...\n")
    run_test_cases(lambda *_: "", verbose=True)

COMMANDS = {
    "download": cmd_download,
    "build_db": cmd_build_db,
    "api": cmd_api,
    "evaluate": cmd_evaluate,
    "demo": cmd_demo,
    "test": cmd_test,
}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd not in COMMANDS:
        print(f"Lệnh không hợp lệ: '{cmd}'")
        print(f"Các lệnh hợp lệ: {', '.join(COMMANDS)}")
        sys.exit(1)
        
    # Thực thi lệnh tương ứng
    COMMANDS[cmd]()
