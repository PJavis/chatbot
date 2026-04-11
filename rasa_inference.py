"""Shared helpers for calling the Rasa REST webhook."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Sequence
from urllib.parse import quote, urlsplit, urlunsplit
from uuid import uuid4

import requests


def _load_local_env() -> None:
    """Load local rasa_app/.env values when running without Docker/env export."""
    env_path = os.getenv("RASA_ENV_FILE")
    path = os.path.abspath(env_path) if env_path else os.path.join(os.path.dirname(__file__), "rasa_app", ".env")
    if not os.path.exists(path):
        return

    with open(path, encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_local_env()

TAVILY_SEARCH_URL = os.getenv("TAVILY_SEARCH_URL", "https://api.tavily.com/search")
TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "3"))
TAVILY_SEARCH_DEPTH = os.getenv("TAVILY_SEARCH_DEPTH", "basic")
TAVILY_INCLUDE_ANSWER = os.getenv("TAVILY_INCLUDE_ANSWER", "advanced")
TAVILY_TIMEOUT = float(os.getenv("TAVILY_TIMEOUT", "20"))
ENABLE_TAVILY_FALLBACK = os.getenv("ENABLE_TAVILY_FALLBACK", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

RAG_FAILURE_MARKERS = (
    "(no response)",
    "tôi không biết",
    "tôi không tìm thấy thông tin phù hợp",
    "hiện chưa có dữ liệu tri thức",
    "không có thông tin",
    "không tìm thấy",
)

CONVERSATION_HISTORY_MARKERS = (
    "tôi vừa",
    "tôi đã hỏi",
    "câu hỏi trước",
    "bạn vừa",
    "câu trả lời trước",
    "nhắc lại",
    "lặp lại",
    "cuộc trò chuyện",
    "chủ đề hiện tại",
    "chúng ta đang nói",
)

FLOW_TO_INTENT = {
    "greet": "greet",
    "ask_help": "ask_help",
    "thanks": "thanks",
    "goodbye": "goodbye",
    "chitchat": "chitchat",
    "pattern_search": "ask_knowledge",
}

ACTION_TO_INTENT = {
    "utter_greet": "greet",
    "utter_help_overview": "ask_help",
    "utter_appreciation": "thanks",
    "utter_goodbye": "goodbye",
    "utter_free_chitchat_response": "chitchat",
    "action_trigger_search": "ask_knowledge",
    "utter_no_relevant_answer_found": "ask_out_of_scope",
}


def call_rasa(message: str, rasa_url: str, sender: str, timeout: float) -> list[dict]:
    payload = {"sender": sender, "message": message}
    response = requests.post(rasa_url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def make_rasa_parse_url(rasa_url: str) -> str:
    """Return the Rasa model parse endpoint matching a REST webhook URL."""
    explicit_parse_url = os.getenv("RASA_PARSE_URL", "").strip()
    if explicit_parse_url:
        return explicit_parse_url

    parsed = urlsplit(rasa_url)
    return urlunsplit((parsed.scheme, parsed.netloc, "/model/parse", "", ""))


def make_rasa_tracker_url(rasa_url: str, sender: str) -> str:
    """Return the Rasa tracker endpoint for a sender."""
    parsed = urlsplit(rasa_url)
    path = f"/conversations/{quote(sender, safe='')}/tracker"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def call_rasa_intent(message: str, rasa_url: str, timeout: float) -> dict[str, object]:
    """Return the top intent from Rasa's model parse endpoint."""
    response = requests.post(
        make_rasa_parse_url(rasa_url),
        json={"text": message},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    intent = data.get("intent") or {}
    ranking = data.get("intent_ranking") or []
    if not intent and ranking:
        intent = ranking[0]

    return {
        "name": str(intent.get("name") or "unknown"),
        "confidence": intent.get("confidence"),
        "source": "model_parse",
    }


def _intent_from_command(command: dict) -> str | None:
    command_name = command.get("command")
    if command_name == "start flow":
        flow = str(command.get("flow") or "")
        return FLOW_TO_INTENT.get(flow, flow or None)
    if command_name == "knowledge":
        return "ask_knowledge"
    return None


def _intent_from_tracker(tracker: dict) -> dict[str, object] | None:
    latest_message = tracker.get("latest_message") or {}
    intent = latest_message.get("intent") or {}
    intent_name = intent.get("name")
    if intent_name:
        return {
            "name": str(intent_name),
            "confidence": intent.get("confidence"),
            "source": "tracker_intent",
        }

    commands = latest_message.get("commands") or []
    for command in commands:
        if not isinstance(command, dict):
            continue
        command_intent = _intent_from_command(command)
        if command_intent:
            return {
                "name": command_intent,
                "confidence": None,
                "source": "tracker_command",
                "command": command.get("command"),
            }

    for event in reversed(tracker.get("events") or []):
        if event.get("event") != "action":
            continue
        action_name = str(event.get("name") or "")
        if action_name in ACTION_TO_INTENT:
            return {
                "name": ACTION_TO_INTENT[action_name],
                "confidence": event.get("confidence"),
                "source": "tracker_action",
                "action": action_name,
            }

        metadata = event.get("metadata") or {}
        active_flow = str(metadata.get("active_flow") or "")
        if active_flow in FLOW_TO_INTENT:
            return {
                "name": FLOW_TO_INTENT[active_flow],
                "confidence": event.get("confidence"),
                "source": "tracker_flow",
                "flow": active_flow,
            }

    return None


def _latest_turn_events(tracker: dict) -> list[dict]:
    events = tracker.get("events") or []
    latest_user_idx = None
    for idx, event in enumerate(events):
        if event.get("event") == "user":
            latest_user_idx = idx
    if latest_user_idx is None:
        return []
    return events[latest_user_idx + 1 :]


def _tool_trace_from_tracker(tracker: dict) -> dict[str, object]:
    for event in _latest_turn_events(tracker):
        if event.get("event") != "action":
            continue
        metadata = event.get("metadata") or {}
        message = metadata.get("message") or {}
        search_results = message.get("search_results")
        if isinstance(search_results, list) and search_results:
            return {
                "source": "rasa_enterprise_search",
                "tool_output": "\n\n".join(str(result).strip() for result in search_results if str(result).strip()),
            }
    return {}


def call_rasa_backend_trace(rasa_url: str, sender: str, timeout: float) -> dict[str, object]:
    """Return the actual Rasa dialogue classification and tool trace from the tracker."""
    response = requests.get(make_rasa_tracker_url(rasa_url, sender), timeout=timeout)
    response.raise_for_status()
    tracker = response.json()
    intent = _intent_from_tracker(tracker)
    if intent:
        return {"intent": intent, **_tool_trace_from_tracker(tracker)}
    return {"intent": {"name": "unknown", "confidence": None, "source": "tracker"}, **_tool_trace_from_tracker(tracker)}


def _with_intent_metadata(response: list[dict], intent: dict[str, object] | None) -> list[dict]:
    if not intent:
        return response

    annotated = []
    for segment in response:
        next_segment = dict(segment)
        metadata = dict(next_segment.get("metadata") or {})
        metadata["intent"] = intent
        next_segment["metadata"] = metadata
        annotated.append(next_segment)
    return annotated


def _with_trace_metadata(response: list[dict], **trace: object) -> list[dict]:
    annotated = []
    for segment in response:
        next_segment = dict(segment)
        metadata = dict(next_segment.get("metadata") or {})
        for key, value in trace.items():
            if value is not None and key not in metadata:
                metadata[key] = value
        next_segment["metadata"] = metadata
        annotated.append(next_segment)
    return annotated


def make_sender(base: str, suffix: str | None = None) -> str:
    token = suffix or uuid4().hex
    return f"{base}-{token}"


def format_response(response: Sequence[dict]) -> str:
    texts = [text for segment in response if (text := segment.get("text", "").strip())]
    combined = " ".join(texts).strip()
    if not combined:
        return "(no response)"
    return combined


def is_rag_failure(answer: str) -> bool:
    """Return true when Rasa indicates the vector DB/RAG path found no answer."""
    normalized = " ".join(answer.casefold().split())
    return any(marker in normalized for marker in RAG_FAILURE_MARKERS)


def is_conversation_history_query(message: str) -> bool:
    """Avoid web fallback for questions that should be answered from chat memory."""
    normalized = " ".join(message.casefold().split())
    return any(marker in normalized for marker in CONVERSATION_HISTORY_MARKERS)


def tavily_search(
    query: str,
    api_key: str | None = None,
    timeout: float | None = None,
) -> dict | None:
    """Return the raw Tavily API response for a search query."""
    key = (api_key or os.getenv("TAVILY_API_KEY", "")).strip()
    if not key:
        return None

    payload = {
        "query": query,
        "topic": "general",
        "search_depth": TAVILY_SEARCH_DEPTH,
        "include_answer": TAVILY_INCLUDE_ANSWER,
        "include_raw_content": False,
        "max_results": max(1, TAVILY_MAX_RESULTS),
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        TAVILY_SEARCH_URL,
        json=payload,
        headers=headers,
        timeout=timeout or TAVILY_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()

    answer = str(data.get("answer") or "").strip()
    results = data.get("results") or []
    if not answer and not results:
        return None
    return data


def format_tavily_search_answer(data: dict) -> str | None:
    """Format a raw Tavily response as a chatbot fallback answer."""
    answer = str(data.get("answer") or "").strip()
    results = data.get("results") or []
    if not answer and not results:
        return None

    lines = []
    if answer:
        lines.append(answer)
    else:
        top_result = results[0]
        content = str(top_result.get("content") or "").strip()
        if content:
            lines.append(content)

    source_lines = []
    for idx, result in enumerate(results[: max(1, TAVILY_MAX_RESULTS)], start=1):
        title = str(result.get("title") or "Nguồn web").strip()
        url = str(result.get("url") or "").strip()
        if not url:
            continue
        source_lines.append(f"[Web {idx}] {title}: {url}")

    if source_lines:
        lines.append("Nguồn: " + " | ".join(source_lines))
    return "\n".join(lines).strip()


def tavily_search_answer(
    query: str,
    api_key: str | None = None,
    timeout: float | None = None,
) -> str | None:
    """Search Tavily and format a fallback answer with web citations."""
    data = tavily_search(query, api_key=api_key, timeout=timeout)
    if not data:
        return None
    return format_tavily_search_answer(data)


def call_rasa_with_tavily_fallback(
    message: str,
    rasa_url: str,
    sender: str,
    timeout: float,
) -> list[dict]:
    """Call Rasa first; if RAG cannot answer, fall back to Tavily web search."""
    rasa_response = call_rasa(message, rasa_url=rasa_url, sender=sender, timeout=timeout)
    rasa_text = format_response(rasa_response)
    if (
        not ENABLE_TAVILY_FALLBACK
        or not is_rag_failure(rasa_text)
        or is_conversation_history_query(message)
    ):
        return _with_trace_metadata(rasa_response, source="rasa")

    try:
        tavily_data = tavily_search(message)
    except requests.RequestException:
        return _with_trace_metadata(rasa_response, source="rasa")
    if not tavily_data:
        return _with_trace_metadata(rasa_response, source="rasa")
    fallback_text = format_tavily_search_answer(tavily_data)
    if not fallback_text:
        return _with_trace_metadata(rasa_response, source="rasa")
    raw_tool_output = json.dumps(tavily_data, ensure_ascii=False, indent=2)

    return [
        {
            "recipient_id": sender,
            "text": fallback_text,
            "metadata": {"source": "tavily_fallback", "rag_output": rasa_text, "tool_output": raw_tool_output},
        }
    ]


def call_rasa_with_tavily_fallback_and_intent(
    message: str,
    rasa_url: str,
    sender: str,
    timeout: float,
) -> list[dict]:
    """Call Rasa chat and attach the backend-classified top intent to responses."""
    response = call_rasa_with_tavily_fallback(
        message,
        rasa_url=rasa_url,
        sender=sender,
        timeout=timeout,
    )
    try:
        backend_trace = call_rasa_backend_trace(rasa_url=rasa_url, sender=sender, timeout=timeout)
    except (requests.RequestException, ValueError):
        try:
            backend_trace = {"intent": call_rasa_intent(message, rasa_url=rasa_url, timeout=timeout)}
        except (requests.RequestException, ValueError):
            backend_trace = {"intent": {"name": "unavailable", "confidence": None, "source": "unavailable"}}
    intent = backend_trace.pop("intent", None)
    traced_response = _with_trace_metadata(response, **backend_trace)
    return _with_intent_metadata(traced_response, intent)


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
