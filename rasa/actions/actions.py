"""
Custom Actions cho RASA Bot - Tích hợp RAG API
Chạy bằng: rasa run actions --port 5055
"""

import logging
import os
import requests
from typing import Any, Dict, List, Optional, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, SessionStarted, ActionExecuted

logger = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────
RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")
RAG_TIMEOUT = int(os.getenv("RAG_TIMEOUT", "15"))
RAG_TOP_K   = int(os.getenv("RAG_TOP_K", "3"))


# ─── Helper ──────────────────────────────────────────────────────
def call_rag_api(
    query: str,
    top_k: int = RAG_TOP_K,
    context: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Gọi RAG API và trả về dict:
      {
        "answer": str,
        "sources": list[str],
        "confidence": float
      }
    """
    payload = {
        "query": query,
        "top_k": top_k,
    }
    if context:
        payload["context"] = context

    response = requests.post(
        f"{RAG_API_URL}/query",
        json=payload,
        timeout=RAG_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def format_sources(sources: List[str]) -> str:
    """Format danh sách nguồn thành text đẹp."""
    if not sources:
        return ""
    unique = list(dict.fromkeys(sources))[:3]  # max 3 nguồn
    return "\n📎 *Nguồn:* " + " | ".join(unique)


# ─── Main RAG Action ─────────────────────────────────────────────
class ActionRagSearch(Action):
    """
    Custom Action chính: nhận câu hỏi từ user → gọi RAG API → trả lời.

    Được trigger bởi các intents:
      - ask_information
      - ask_technical
      - ask_faq
    """

    def name(self) -> Text:
        return "action_rag_search"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        # 1. Lấy câu hỏi gốc của user
        user_message = tracker.latest_message.get("text", "").strip()
        if not user_message:
            dispatcher.utter_message(response="utter_ask_rephrase")
            return []

        # 2. Lấy context từ các turns trước (nếu có)
        conversation_context = tracker.get_slot("conversation_context") or ""

        # 3. Lấy entities để bổ sung ngữ cảnh
        entities = tracker.latest_message.get("entities", [])
        entity_info = " ".join(
            f"{e.get('entity')}: {e.get('value')}"
            for e in entities
            if e.get("entity") in ("topic", "product", "error_code")
        )

        # Tạo enhanced query nếu có entities
        query = user_message
        if entity_info:
            query = f"{user_message} [{entity_info}]"

        logger.info(f"RAG query: {query!r} | intent: {tracker.get_intent_of_latest_message()}")

        # 4. Gọi RAG API
        try:
            result = call_rag_api(
                query=query,
                context=conversation_context if conversation_context else None,
            )
        except requests.Timeout:
            logger.error("RAG API timeout")
            dispatcher.utter_message(
                text="⚠️ Hệ thống tìm kiếm đang chậm. Vui lòng thử lại sau."
            )
            return []
        except requests.ConnectionError:
            logger.error("Cannot connect to RAG API at %s", RAG_API_URL)
            dispatcher.utter_message(response="utter_error")
            return []
        except requests.HTTPError as e:
            logger.error("RAG API HTTP error: %s", e)
            dispatcher.utter_message(response="utter_error")
            return []
        except Exception as e:
            logger.exception("Unexpected error calling RAG API: %s", e)
            dispatcher.utter_message(response="utter_error")
            return []

        # 5. Xử lý kết quả
        answer    = result.get("answer", "").strip()
        sources   = result.get("sources", [])
        confidence = result.get("confidence", 0.0)

        logger.info(f"RAG confidence: {confidence:.2f} | answer length: {len(answer)}")

        if not answer or confidence < 0.2:
            dispatcher.utter_message(response="utter_no_result")
            return [SlotSet("rag_answer", None)]

        # 6. Format và gửi câu trả lời
        full_response = answer
        if sources:
            full_response += format_sources(sources)

        # Thêm cảnh báo nếu confidence thấp
        if confidence < 0.5:
            full_response += "\n\n⚠️ *Lưu ý:* Tôi không hoàn toàn chắc chắn về câu trả lời này. Vui lòng xác nhận với tài liệu gốc."

        dispatcher.utter_message(text=full_response)

        # 7. Cập nhật slots để dùng cho turns tiếp theo
        new_context = f"Q: {user_message}\nA: {answer}"
        if conversation_context:
            # Giữ tối đa 3 turns gần nhất
            lines = conversation_context.split("\n")
            lines = lines[-6:] if len(lines) > 6 else lines
            conversation_context = "\n".join(lines)
            new_context = conversation_context + "\n" + new_context

        return [
            SlotSet("user_query", user_message),
            SlotSet("rag_answer", answer),
            SlotSet("conversation_context", new_context),
        ]


# ─── Set Context Action ───────────────────────────────────────────
class ActionSetContext(Action):
    """Reset conversation context (ví dụ khi đổi chủ đề)."""

    def name(self) -> Text:
        return "action_set_context"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        return [SlotSet("conversation_context", None)]


# ─── Human Handoff Action ─────────────────────────────────────────
class ActionHumanHandoff(Action):
    """Chuyển conversation sang human agent."""

    def name(self) -> Text:
        return "action_human_handoff"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        # Lấy context để gửi cho agent
        last_query   = tracker.get_slot("user_query") or "Không có thông tin"
        conversation = tracker.get_slot("conversation_context") or "Chưa có hội thoại"

        # Gọi webhook/ticket system (tuỳ chỉnh theo hệ thống của bạn)
        try:
            requests.post(
                f"{os.getenv('SUPPORT_WEBHOOK_URL', 'http://localhost:9000')}/handoff",
                json={
                    "sender_id": tracker.sender_id,
                    "last_query": last_query,
                    "conversation_summary": conversation[-500:],
                },
                timeout=5,
            )
        except Exception as e:
            logger.warning("Could not notify support system: %s", e)

        dispatcher.utter_message(response="utter_human_handoff")
        return [SlotSet("conversation_context", None)]
