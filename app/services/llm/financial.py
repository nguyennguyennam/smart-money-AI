"""LLM helpers that combine OCR/classification/expense extraction in one call.

These helpers own the prompts and JSON parsing so callers (e.g. the Redis
worker) stay focused on orchestration. All calls go through `LLMService`
against OpenAI's `gpt-5-nano` (vision + text) and `gpt-4o-mini-transcribe`
(audio).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.services.classifer.enums import CATEGORIES, TRANSACTION_TYPES
from app.services.llm.service import LLMProvider, LLMService

logger = logging.getLogger(__name__)


DEFAULT_EXPENSE_VND = 50000
DEFAULT_CATEGORY = "OTHER"
DEFAULT_TYPE = "EXPENSE"


_INSTRUCTION_BLOCK = (
    f"Danh mục cho phép (category): {', '.join(CATEGORIES)}.\n"
    f"Loại giao dịch cho phép (type): {', '.join(TRANSACTION_TYPES)}.\n"
    "Quy tắc:\n"
    "- expense: số nguyên VND, là tổng tiền giao dịch. Nếu không chắc chắn, dùng 50000.\n"
    "- category: chọn DUY NHẤT một nhãn từ danh mục cho phép. Nếu không chắc, dùng OTHER.\n"
    "- type: EXPENSE cho chi tiêu/thanh toán, INCOME cho thu nhập/tiền nhận. Mặc định EXPENSE.\n"
)


_OCR_PROMPT = (
    "Bạn là hệ thống xử lý hóa đơn tiếng Việt từ ảnh.\n"
    "Hãy thực hiện ĐỒNG THỜI ba việc và trả về DUY NHẤT một đối tượng JSON với các khóa: "
    "text, category, type, expense.\n"
    "- text: toàn bộ nội dung văn bản đọc được từ ảnh (giữ nguyên tiếng Việt, các dòng cách nhau bằng \\n).\n"
    f"{_INSTRUCTION_BLOCK}"
    "Chỉ trả về JSON, không thêm chú thích hay markdown."
)


def _classify_extract_prompt(text_vi: str) -> str:
    return (
        "Bạn là hệ thống phân tích giao dịch tài chính từ văn bản tiếng Việt.\n"
        "Trả về DUY NHẤT một đối tượng JSON với các khóa: category, type, expense.\n"
        f"{_INSTRUCTION_BLOCK}"
        "Chỉ trả về JSON, không thêm chú thích hay markdown.\n\n"
        "Văn bản:\n"
        "```\n"
        f"{text_vi}\n"
        "```"
    )


def _parse_expense_number(raw: Any) -> int:
    if raw is None:
        return DEFAULT_EXPENSE_VND
    if isinstance(raw, bool):
        return DEFAULT_EXPENSE_VND
    if isinstance(raw, (int, float)):
        n = int(raw)
        return n if n > 0 else DEFAULT_EXPENSE_VND

    s = str(raw).strip()
    m = re.search(r"\d[\d\s.,]*\d|\d+", s)
    if not m:
        return DEFAULT_EXPENSE_VND

    digits = re.sub(r"\D", "", m.group(0))
    if not digits:
        return DEFAULT_EXPENSE_VND

    try:
        n = int(digits)
    except Exception:
        return DEFAULT_EXPENSE_VND

    return n if n > 0 else DEFAULT_EXPENSE_VND


def _normalize_category(raw: Any) -> str:
    if raw is None:
        return DEFAULT_CATEGORY
    s = str(raw).strip().upper()
    if s in CATEGORIES:
        return s
    for c in CATEGORIES:
        if re.search(rf"\b{re.escape(c)}\b", s):
            return c
    return DEFAULT_CATEGORY


def _normalize_type(raw: Any) -> str:
    if raw is None:
        return DEFAULT_TYPE
    s = str(raw).strip().upper()
    if s in TRANSACTION_TYPES:
        return s
    if "INCOME" in s:
        return "INCOME"
    if "EXPENSE" in s:
        return "EXPENSE"
    return DEFAULT_TYPE


def _strip_json_fence(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```"):
        # Drop leading fence (``` or ```json) and trailing fence
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _parse_json_object(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    s = _strip_json_fence(raw)
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Best-effort: pull the first {...} block out.
    match = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return {}
    return {}


async def ocr_classify_extract(
    llm: LLMService,
    image_bytes: bytes,
    mime_type: str,
) -> dict[str, Any]:
    """One gpt-5-nano vision call returning text + category + type + expense."""
    try:
        raw = await llm.generate_with_image(
            prompt=_OCR_PROMPT,
            image_bytes=image_bytes,
            mime_type=mime_type,
            response_format={"type": "json_object"},
        )
    except TypeError:
        # Older OpenAI SDKs may not accept response_format on chat.completions.
        raw = await llm.generate_with_image(
            prompt=_OCR_PROMPT,
            image_bytes=image_bytes,
            mime_type=mime_type,
        )
    except Exception as e:
        logger.warning("ocr_classify_extract LLM call failed: %s", e)
        return {
            "text": "",
            "category": DEFAULT_CATEGORY,
            "type": DEFAULT_TYPE,
            "expense": DEFAULT_EXPENSE_VND,
            "error": str(e),
        }

    obj = _parse_json_object(raw)
    text = str(obj.get("text") or "").strip()

    return {
        "text": text,
        "category": _normalize_category(obj.get("category")),
        "type": _normalize_type(obj.get("type")),
        "expense": _parse_expense_number(obj.get("expense")),
        "error": None if text else "No readable text content found in the uploaded file",
    }


async def classify_and_extract(
    llm: LLMService,
    text_vi: str,
) -> dict[str, Any]:
    """One gpt-5-nano text call returning category + type + expense for a transcript."""
    if not text_vi or not isinstance(text_vi, str) or not text_vi.strip():
        return {
            "category": DEFAULT_CATEGORY,
            "type": DEFAULT_TYPE,
            "expense": DEFAULT_EXPENSE_VND,
        }

    prompt = _classify_extract_prompt(text_vi)

    try:
        raw = await llm.generate(
            prompt=prompt,
            provider=LLMProvider.OPENAI,
            response_format={"type": "json_object"},
        )
    except TypeError:
        raw = await llm.generate(prompt=prompt, provider=LLMProvider.OPENAI)
    except Exception as e:
        logger.warning("classify_and_extract LLM call failed: %s", e)
        return {
            "category": DEFAULT_CATEGORY,
            "type": DEFAULT_TYPE,
            "expense": DEFAULT_EXPENSE_VND,
        }

    obj = _parse_json_object(raw)
    return {
        "category": _normalize_category(obj.get("category")),
        "type": _normalize_type(obj.get("type")),
        "expense": _parse_expense_number(obj.get("expense")),
    }
