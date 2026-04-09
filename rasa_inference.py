"""Shared helpers for calling the Rasa REST webhook."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Sequence

import requests


def call_rasa(message: str, rasa_url: str, sender: str, timeout: float) -> list[dict]:
    payload = {"sender": sender, "message": message}
    response = requests.post(rasa_url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def format_response(response: Sequence[dict]) -> str:
    texts = [segment.get("text", "") for segment in response if segment.get("text")]
    return " ".join(texts).strip() or "(no response)"


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
            response = call_rasa(question, rasa_url=rasa_url, sender=sender, timeout=timeout)
            return ex["id"], format_response(response)
        except requests.RequestException:
            return ex["id"], ""

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = [executor.submit(_predict, ex) for ex in examples]
        for future in as_completed(futures):
            ex_id, prediction = future.result()
            predictions[ex_id] = prediction

    return predictions
