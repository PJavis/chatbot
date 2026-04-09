FROM python:3.11-slim

ENV LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 5005

COPY start-rasa.sh /app/start-rasa.sh
RUN chmod +x /app/start-rasa.sh

ENTRYPOINT ["/app/start-rasa.sh"]
