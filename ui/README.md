# Gradio Rasa UI

This folder contains a minimal Gradio-based streaming chat interface that forwards messages to the Rasa REST `webhook/rest/webhook` endpoint and renders assistant replies in real time.

## Setup

1. Install the dependencies for the UI.

```bash
cd ui
pip install -r requirements.txt
```

2. Make sure your Rasa backend is running with the REST channel enabled (`rasa run --enable-api`).

3. Configure the UI environment variables (defaults are shown):

```bash
export RASA_REST_URL=http://localhost:5005/webhooks/rest/webhook
export RASA_CHAT_SENDER=gradio_user
export UI_PORT=7860
```

4. Start the UI and open the provided URL in a browser.

```bash
python app.py
```

Every time you send a message from the Gradio interface it will post to the configured Rasa endpoint, stream the assistant text word by word, and append the final reply to the chat history.
