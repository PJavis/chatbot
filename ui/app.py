"""Gradio web interface that streams responses from the Rasa REST chat API."""
import os
from typing import List, Sequence

import gradio as gr
import requests

RASA_REST_URL = os.getenv("RASA_REST_URL", "http://localhost:5005/webhooks/rest/webhook")
RASA_SENDER = os.getenv("RASA_CHAT_SENDER", "gradio_user")
RASA_TIMEOUT = float(os.getenv("RASA_API_TIMEOUT", "30"))


def call_rasa(message: str) -> List[dict]:
    payload = {"sender": RASA_SENDER, "message": message}
    response = requests.post(RASA_REST_URL, json=payload, timeout=RASA_TIMEOUT)
    response.raise_for_status()
    return response.json()


def format_response(response: Sequence[dict]) -> str:
    texts = [segment.get("text", "") for segment in response if segment.get("text")]
    return " ".join(texts).strip() or "(no response)"


def stream_response(message: str, history: List[dict]):
    """Yield intermediate chatter results back to Gradio for streaming."""
    if not message or not message.strip():
        yield history, message
        return

    history = history or []
    history.append({"role": "user", "content": message})
    yield history, ""

    try:
        rasa_payload = call_rasa(message)
    except requests.RequestException as exc:
        error_text = f"Failed to reach Rasa: {exc}"
        history.append({"role": "assistant", "content": error_text})
        yield history, ""
        return

    assistant_text = format_response(rasa_payload)
    history.append({"role": "assistant", "content": ""})
    streamed = ""

    for word in assistant_text.split():
        streamed = f"{streamed} {word}".strip()
        history[-1]["content"] = streamed
        yield history[:], ""

    if streamed != assistant_text:
        history[-1]["content"] = assistant_text
        yield history[:], ""


def new_conversation():
    return []


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Rasa Streaming Chat") as demo:
        gr.Markdown("""
        ## Rasa Streaming Chat UI

        This Gradio client streams text responses from your locally running Rasa assistant (REST channel). Adjust `RASA_REST_URL` if your bot is hosted elsewhere.
        """)
        chatbot = gr.Chatbot(elem_id="rasa-chatbot")
        with gr.Row():
            txt = gr.Textbox(show_label=False, placeholder="Ask the bot...", lines=1)
            submit = gr.Button("Send")
        txt.submit(stream_response, [txt, chatbot], [chatbot, txt])
        submit.click(stream_response, [txt, chatbot], [chatbot, txt])
        gr.Button("New conversation").click(lambda: ([], ""), outputs=[chatbot, txt])
    return demo


if __name__ == "__main__":
    css = ".gradio-container { max-width: 900px; margin: auto; }"
    build_interface().launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("UI_PORT", "7860")),
        share=False,
        css=css,
    )
