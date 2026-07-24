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
DEFAULT_DESCRIPTION = ""
# Confidence when the model omits it on an otherwise-successful result (neutral,
# not the misleading 1.0 we used before). Error/empty paths use 0.0.
DEFAULT_CONFIDENCE = 0.5


# Describes the shape of each item inside the `transactions` array.
_TRANSACTION_FIELDS_BLOCK = (
    "Mỗi giao dịch là một object với các khóa: category, type, expense, description, confidence.\n"
    f"Danh mục cho phép (category): {', '.join(CATEGORIES)}.\n"
    f"Loại giao dịch cho phép (type): {', '.join(TRANSACTION_TYPES)}.\n"
    
    "Quy tắc cho TỪNG giao dịch:\n"
    "- expense: số nguyên VND, số tiền của RIÊNG giao dịch đó. Nếu không chắc chắn, dùng 50000.\n"
    "- category: chọn DUY NHẤT một nhãn từ danh mục cho phép. Nếu không chắc, dùng OTHER.\n"
    "- type: EXPENSE cho chi tiêu/thanh toán, INCOME cho thu nhập/tiền nhận. Mặc định EXPENSE.\n"
    "- description: mô tả ngắn gọn 2-5 từ về giao dịch, dùng CÙNG ngôn ngữ với văn bản "
    "(tiếng Việt nếu văn bản tiếng Việt, tiếng Anh nếu văn bản tiếng Anh). "
    "Không dấu câu thừa, không markdown.\n"
    "- confidence: số thực 0..1 thể hiện mức độ chắc chắn về category/type/expense "
    "(1 = rất chắc chắn, 0 = không chắc).\n"
    "Nếu văn bản có NHIỀU khoản (ví dụ 'cà phê 50k, bánh mì 20k'), hãy TÁCH thành nhiều giao dịch, "
    "mỗi khoản một object. Nếu chỉ có một khoản, trả về mảng gồm một phần tử.\n"
)


# Single-transaction field rules for OCR (one receipt → one transaction).
_SINGLE_TRANSACTION_FIELDS_BLOCK = (
    f"Danh mục cho phép (category): {', '.join(CATEGORIES)}.\n"
    f"Loại giao dịch cho phép (type): {', '.join(TRANSACTION_TYPES)}.\n"
    "Quy tắc:\n"
    "- expense: số nguyên VND, TỔNG số tiền của hóa đơn. Nếu không chắc chắn, dùng 50000.\n"
    "- category: chọn DUY NHẤT một nhãn từ danh mục cho phép. Nếu không chắc, dùng OTHER.\n"
    "- type: EXPENSE cho chi tiêu/thanh toán, INCOME cho thu nhập/tiền nhận. Mặc định EXPENSE.\n"
    "- description: mô tả ngắn gọn 2-5 từ về hóa đơn, dùng CÙNG ngôn ngữ với văn bản "
    "(tiếng Việt nếu văn bản tiếng Việt, tiếng Anh nếu văn bản tiếng Anh). "
    "Không dấu câu thừa, không markdown.\n"
    "- confidence: số thực 0..1 thể hiện mức độ chắc chắn về category/type/expense "
    "(1 = rất chắc chắn, 0 = không chắc).\n"
)


_OCR_PROMPT = (
    "Bạn là hệ thống xử lý hóa đơn tiếng Việt từ ảnh.\n"
    "Trả về DUY NHẤT một đối tượng JSON với các khóa: text, category, type, expense, description, confidence.\n"
    "- text: toàn bộ nội dung văn bản đọc được từ ảnh (giữ nguyên tiếng Việt, các dòng cách nhau bằng \\n).\n"
    "Toàn bộ hóa đơn được coi là MỘT giao dịch duy nhất (gộp các khoản thành một tổng).\n"
    f"{_SINGLE_TRANSACTION_FIELDS_BLOCK}"
    "Chỉ trả về JSON, không thêm chú thích hay markdown."
)


def _classify_extract_prompt(text_vi: str) -> str:
    return (
        "Bạn là hệ thống phân tích giao dịch tài chính từ văn bản tiếng Việt.\n"
        "Trả về DUY NHẤT một đối tượng JSON với khóa: transactions (một MẢNG/array các giao dịch).\n"
        f"{_TRANSACTION_FIELDS_BLOCK}"
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


def _normalize_description(raw: Any) -> str:
    if raw is None:
        return DEFAULT_DESCRIPTION
    s = re.sub(r"\s+", " ", str(raw).strip().strip("\"'").strip())
    if not s:
        return DEFAULT_DESCRIPTION
    # Guardrail in case the model ignores the 2-5 word request.
    words = s.split(" ")
    if len(words) > 6:
        s = " ".join(words[:6])
    return s


def _normalize_confidence(raw: Any, default: float = DEFAULT_CONFIDENCE) -> float:
    if raw is None:
        return default
    try:
        val = float(str(raw).strip().rstrip("%"))
    except (TypeError, ValueError):
        return default
    if val > 1.0:  # tolerate a 0..100 (percentage) scale
        val = val / 100.0
    return max(0.0, min(1.0, round(val, 4)))


def _default_transaction(confidence: float = 0.0) -> dict[str, Any]:
    return {
        "category": DEFAULT_CATEGORY,
        "type": DEFAULT_TYPE,
        "expense": DEFAULT_EXPENSE_VND,
        "description": DEFAULT_DESCRIPTION,
        "confidence": confidence,
    }


def _normalize_transaction(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        return _default_transaction()
    return {
        "category": _normalize_category(item.get("category")),
        "type": _normalize_type(item.get("type")),
        "expense": _parse_expense_number(item.get("expense")),
        "description": _normalize_description(item.get("description")),
        "confidence": _normalize_confidence(item.get("confidence")),
    }


def _normalize_transactions(obj: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull a clean, non-empty list of transactions out of the model's JSON.

    Tolerates the legacy single-object shape and always returns at least one
    item (a low-confidence default) so downstream never gets an empty array.
    """
    raw = obj.get("transactions") if isinstance(obj, dict) else None

    items: list[dict[str, Any]] = []
    if isinstance(raw, list):
        items = [_normalize_transaction(it) for it in raw if isinstance(it, dict)]
    elif isinstance(obj, dict) and any(k in obj for k in ("category", "type", "expense")):
        # Legacy single-object response shape.
        items = [_normalize_transaction(obj)]

    if not items:
        items = [_default_transaction()]
    return items


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
    """One gpt-5-nano vision call returning text + a single transaction.

    A receipt image is treated as one transaction (the whole-bill total), so
    this returns flat fields (category/type/expense/description/confidence)
    rather than a list.
    """
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
        return {"text": "", "error": str(e), **_default_transaction()}

    obj = _parse_json_object(raw)
    text = str(obj.get("text") or "").strip()

    if not text:
        return {
            "text": "",
            "error": "No readable text content found in the uploaded file",
            **_default_transaction(),
        }

    return {"text": text, "error": None, **_normalize_transaction(obj)}


async def classify_and_extract(
    llm: LLMService,
    text_vi: str,
) -> dict[str, Any]:
    """One gpt-5-nano text call returning a list of transactions for a transcript."""
    if not text_vi or not isinstance(text_vi, str) or not text_vi.strip():
        return {"transactions": [_default_transaction()]}

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
        return {"transactions": [_default_transaction()]}

    obj = _parse_json_object(raw)
    return {"transactions": _normalize_transactions(obj)}

async def classify_notification(
    llm: LLMService,
    text: str,
) -> dict[str, Any]:
    """
    Classify bank notification (SMS / push).
    ALWAYS return exactly ONE transaction.
    """

    if not text or not isinstance(text, str) or not text.strip():
        return {"transactions": [_default_transaction()]}

    prompt = (
        "Bạn là hệ thống phân tích thông báo ngân hàng (SMS/push notification).\n"
        "Đây LUÔN là MỘT giao dịch duy nhất.\n\n"

        "Trả về DUY NHẤT một JSON object với khóa: transaction.\n\n"

        "Quy tắc QUAN TRỌNG:\n"
        "- Chỉ tạo 1 giao dịch duy nhất\n"
        "- KHÔNG được tự tạo thêm giao dịch\n"
        "- expense: lấy từ số tiền thay đổi số dư (ví dụ '-30,000 VND' → 30000)\n"
        "- Nếu có dấu '-' → EXPENSE\n"
        "- Nếu là tiền vào (cộng tiền) → INCOME\n"
        "- description: lấy từ nội dung chính (ví dụ 'nap tien dien thoai')\n"
        "- category: chọn từ danh mục, nếu không chắc → OTHER\n"
        "- confidence: 0..1\n\n"

        f"Danh mục: {', '.join(CATEGORIES)}\n"
        f"Loại: {', '.join(TRANSACTION_TYPES)}\n\n"

        "Chỉ trả về JSON, không markdown.\n\n"

        "Văn bản:\n"
        "```\n"
        f"{text}\n"
        "```"
    )

    try:
        raw = await llm.generate(
            prompt=prompt,
            provider=LLMProvider.OPENAI,
            response_format={"type": "json_object"},
        )
    except TypeError:
        raw = await llm.generate(prompt=prompt, provider=LLMProvider.OPENAI)
    except Exception as e:
        logger.warning("classify_notification LLM failed: %s", e)
        return {"transactions": [_default_transaction()]}

    obj = _parse_json_object(raw)

    # hỗ trợ cả 2 format: {transaction:{}} hoặc flat
    tx = obj.get("transaction") if isinstance(obj, dict) else None
    if not tx and isinstance(obj, dict):
        tx = obj

    normalized = _normalize_transaction(tx)

    return {"transactions": [normalized]}