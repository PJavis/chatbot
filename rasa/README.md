# Rasa Project Setup Guide

Dự án chatbot sử dụng Rasa 3.6.21 và Python 3.10.

## 🛠 1. Cài đặt môi trường
```bash
# Cài đặt Python 3.10
pyenv install 3.10.13

# Thiết lập venv cho dự án
cd rasa_project
pyenv local 3.10.13
python -m venv venv
source venv/bin/activate

# Cài đặt thư viện
pip install rasa==3.6.21 rasa-sdk==3.6.2 requests
```

## 🚀 2. Chạy dự án (Mở 4 Terminal)

| Terminal | Lệnh thực thi | Ghi chú |
| :--- | :--- | :--- |
| **T1** | `python3 fake_rag_api.py` | Chạy API giả lập (Python 3.12) |
| **T2** | `rasa train` | Huấn luyện mô hình (venv 3.10) |
| **T3** | `rasa run actions` | Chạy Action Server (venv 3.10) |
| **T4** | `rasa run --enable-api --cors "*"` | Chạy Rasa Server (venv 3.10) |

> **Lưu ý:** Nhớ chạy `source venv/bin/activate` trước khi thực hiện các lệnh ở T2, T3, T4.
