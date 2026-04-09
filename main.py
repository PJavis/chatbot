"""
Entry point cho chatbot project.

Commands:
    python main.py download   – Tải & xử lý UIT-ViQuAD2.0
    python main.py evaluate   – Chạy evaluation qua Rasa knowledge base
    python main.py demo       – Khởi động Gradio UI tích hợp
    python main.py rasa       – Khởi động backend Rasa
    python main.py test       – Chạy manual test cases
"""

import os
import sys
from pathlib import Path


def cmd_download():
    from data.download import main
    main()


def cmd_evaluate():
    from evaluation.evaluate import main
    sys.argv = ["evaluate", "--split", "validation", "--source", "rasa"]
    main()


def cmd_demo():
    from ui.app import build_app
    app = build_app()
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)


def cmd_rasa():
    script = Path(__file__).parent / "rasa_app" / "start-rasa.sh"
    os.execvp("bash", ["bash", str(script)])


def cmd_test():
    from evaluation.test_cases import run_test_cases

    print("Chạy manual test cases với dummy RAG (trả về chuỗi rỗng)...\n")
    run_test_cases(lambda *_: "", verbose=True)


COMMANDS = {
    "download": cmd_download,
    "evaluate": cmd_evaluate,
    "demo": cmd_demo,
    "rasa": cmd_rasa,
    "test": cmd_test,
}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd not in COMMANDS:
        print(f"Lệnh không hợp lệ: '{cmd}'")
        print(f"Các lệnh hợp lệ: {', '.join(COMMANDS)}")
        sys.exit(1)
    COMMANDS[cmd]()
