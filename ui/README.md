# Integrated Gradio UI

This app combines:

- dataset exploration for processed UIT-ViQuAD2.0 examples,
- interactive EM / F1 / BLEU scoring,
- file-based prediction generation from an uploaded processed split JSON through the live Rasa backend,
- file-based evaluation for `predictions.json`, or live evaluation through the Rasa knowledge base when no file is uploaded,
- a chat tab that forwards messages to the Rasa REST API.

## Setup

1. Install the UI dependencies.

```bash
cd ui
pip install -r requirements.txt
```

2. Make sure the processed dataset exists:

```bash
python -m data.download
```

3. Start the Rasa backend with the REST channel enabled.

4. Configure the UI environment variables if needed:

```bash
export RASA_REST_URL=http://localhost:5005/webhooks/rest/webhook
export RASA_CHAT_SENDER=gradio_user
export RASA_API_TIMEOUT=30
export UI_PORT=7860
```

5. Launch the app from the repo root:

```bash
python ui/app.py
```

The chat tab posts each message to the configured Rasa endpoint and streams assistant text back into the conversation.

## Docker

The `ui` service in `docker-compose.yml` mounts `./data` into `/app/data` so the dataset tabs can read the processed splits inside the container. Make sure `python -m data.download` has been run on the host before starting the UI container. The evaluation tab will call Rasa when no predictions file is uploaded, so the Rasa container must be running too.

The `Tạo Predictions` tab accepts a file shaped like `data/processed/test.json`, sends each example question to Rasa, and returns a downloadable `predictions.json` file that can be uploaded directly in the evaluation tab.
