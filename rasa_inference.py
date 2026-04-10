"""Shared helpers for calling the Rasa REST webhook."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from uuid import uuid4
from typing import Sequence

import requests


def call_rasa(message: str, rasa_url: str, sender: str, timeout: float) -> list[dict]:
    payload = {"sender": sender, "message": message}
    response = requests.post(rasa_url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def make_sender(base: str, suffix: str | None = None) -> str:
    token = suffix or uuid4().hex
    return f"{base}-{token}"


def format_response(response: Sequence[dict]) -> str:
    texts = [text for segment in response if (text := segment.get("text", "").strip())]
    combined = " ".join(texts).strip()
    if not combined:
        return "(no response)"
    return combined


def generate_rasa_predictions(
    examples: list[dict],
    rasa_url: str,
    sender: str,
    timeout: float,
    concurrency: int = 4,
) -> dict[str, str]:
    predictions: dict[str, str] = {}

    def _predict(ex: dict) -> tuple[str, str]:
        question = ex.get("question", "").strip()
        if not question:
            return ex["id"], ""
        try:
            isolated_sender = make_sender(sender, ex.get("id") or None)
            response = call_rasa(question, rasa_url=rasa_url, sender=isolated_sender, timeout=timeout)
            return ex["id"], format_response(response)
        except requests.RequestException:
            return ex["id"], ""

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = [executor.submit(_predict, ex) for ex in examples]
        for future in as_completed(futures):
            ex_id, prediction = future.result()
            predictions[ex_id] = prediction

    return predictions
