from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import socket
from dataclasses import dataclass
from typing import Any

import redis.asyncio as redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError

from app.core.config import settings
from app.services.fetcher.cloudinary_fetcher import get_cloudinary_fetcher

logger = logging.getLogger(__name__)


_DEFAULT_EXPENSE_VND = 50000
_CATEGORY_LLM_FALLBACK_THRESHOLD = 0.5


def _parse_expense_number(raw: str | None) -> int:
    if not raw:
        return _DEFAULT_EXPENSE_VND

    s = str(raw).strip()
    # Grab first number-like token (supports 1.234.567 or 1,234,567)
    m = re.search(r"\d[\d\s.,]*\d|\d+", s)
    if not m:
        return _DEFAULT_EXPENSE_VND

    token = m.group(0)
    digits = re.sub(r"\D", "", token)
    if not digits:
        return _DEFAULT_EXPENSE_VND

    try:
        n = int(digits)
    except Exception:
        return _DEFAULT_EXPENSE_VND

    return n if n > 0 else _DEFAULT_EXPENSE_VND


async def _extract_expense_vnd(llm_service, bill_text_vi: str) -> int:
    if not bill_text_vi or not isinstance(bill_text_vi, str):
        return _DEFAULT_EXPENSE_VND

    prompt = (
        "Bạn là hệ thống trích xuất chi phí từ nội dung hóa đơn bằng tiếng Việt.\n"
        "Hãy trả về DUY NHẤT một số nguyên (VND) đại diện cho tổng tiền phải trả.\n"
        "Không giải thích, không thêm chữ, không thêm ký hiệu tiền tệ.\n"
        "Nếu không chắc chắn hoặc không tìm thấy tổng tiền, hãy trả về 50000.\n\n"
        "Nội dung hóa đơn:\n"
        "```\n"
        f"{bill_text_vi}\n"
        "```\n"
        "\nChỉ trả về một số:"
    )

    try:
        out = await llm_service.generate(prompt=prompt, provider="gemini")
    except Exception:
        return _DEFAULT_EXPENSE_VND

    return _parse_expense_number(out)


async def _classify_category_with_llm(llm_service, text_vi: str) -> str | None:
    if not text_vi or not isinstance(text_vi, str):
        return None

    # Import here to avoid pulling classifier deps at module import time.
    from app.services.classifer.enums import CATEGORIES

    categories = ", ".join(CATEGORIES)

    prompt = (
        "Bạn là hệ thống phân loại chi tiêu từ văn bản tiếng Việt.\n"
        "Hãy trả về DUY NHẤT một nhãn danh mục từ danh sách cho phép.\n"
        "Không giải thích, không thêm ký tự khác.\n"
        "Nếu không chắc chắn, trả về OTHER.\n\n"
        f"Danh mục cho phép: {categories}\n\n"
        "Văn bản:\n"
        "```\n"
        f"{text_vi}\n"
        "```\n\n"
        "Chỉ trả về một nhãn danh mục:" 
    )

    try:
        out = await llm_service.generate(prompt=prompt, provider="gemini")
    except Exception:
        return None

    if not out:
        return None

    out_norm = str(out).strip().upper()
    # Try exact match first
    if out_norm in CATEGORIES:
        return out_norm

    # Fallback: find any allowed token inside the response
    for c in CATEGORIES:
        if re.search(rf"\b{re.escape(c)}\b", out_norm):
            return c

    return None


async def _classify_transaction_type(llm_service, text_vi: str) -> str | None:
    """Classify if transaction is EXPENSE or INCOME using LLM."""
    if not text_vi or not isinstance(text_vi, str):
        return None

    prompt = (
        "Bạn là hệ thống phân loại giao dịch tài chính từ văn bản tiếng Việt.\n"
        "Hãy trả về DUY NHẤT một loại giao dịch: EXPENSE hoặc INCOME.\n"
        "EXPENSE: chi tiêu, thanh toán, mua sắm, ...\n"
        "INCOME: thu nhập, lương, thưởng, tiền nhận, ...\n"
        "Không giải thích, không thêm ký tự khác.\n"
        "Nếu không chắc chắn, trả về EXPENSE.\n\n"
        "Văn bản:\n"
        "```\n"
        f"{text_vi}\n"
        "```\n\n"
        "Chỉ trả về EXPENSE hoặc INCOME:"
    )

    try:
        out = await llm_service.generate(prompt=prompt, provider="gemini")
    except Exception:
        return None

    if not out:
        return None

    out_norm = str(out).strip().upper()
    
    # Check for exact match
    if out_norm == "EXPENSE":
        return "EXPENSE"
    if out_norm == "INCOME":
        return "INCOME"
    
    # Fallback: try to find the keyword in response
    if re.search(r"\bINCOME\b", out_norm):
        return "INCOME"
    if re.search(r"\bEXPENSE\b", out_norm):
        return "EXPENSE"
    
    # Default to EXPENSE if unsure
    return "EXPENSE"


def _redis_from_url_checked(url: str, label: str) -> redis.Redis:
    try:
        # Keep connections alive and do periodic health checks to reduce
        # "Connection closed by server" issues (common on managed Redis).
        return redis.from_url(
            url,
            decode_responses=False,
            socket_keepalive=True,
            health_check_interval=30,
            socket_connect_timeout=10,
            socket_timeout=30,
            retry_on_timeout=True,
        )
    except ValueError as e:
        raise RuntimeError(
            f"Invalid {label}='{url}'. Expected formats like: "
            "redis://host:6379/0 or redis://:password@host:6379/0 or rediss://:password@host:6379/0"
        ) from e


async def _close_redis_client(r: redis.Redis) -> None:
    try:
        await r.close()
    except Exception:
        pass
    try:
        await r.connection_pool.disconnect(inuse_connections=True)
    except Exception:
        pass


@dataclass(frozen=True)
class JobEvent:
    job_id: str
    user_id: str | None
    file_url: str
    duty: str
    created_at: str | None


def _decode_stream_fields(fields: dict[Any, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in fields.items():
        if isinstance(k, bytes):
            k = k.decode("utf-8", errors="replace")
        else:
            k = str(k)

        if isinstance(v, bytes):
            v = v.decode("utf-8", errors="replace")
        else:
            v = str(v)

        out[k] = v
    return out

def _detect_transaction_type_rule(text: str) -> str | None:
    if not text:
        return None

    lower = text.lower()

    # ✅ dấu - => EXPENSE
    if re.search(r"[-−]\s?\d", text):
        return "EXPENSE"

    # ✅ từ khóa income
    if any(kw in lower for kw in ["nhận", "chuyển đến", "cộng tiền", "credit", "ghi có"]):
        return "INCOME"

    # ✅ từ khóa expense
    if any(kw in lower for kw in ["trừ", "thanh toán", "chi", "debit", "ghi nợ"]):
        return "EXPENSE"

    return None


def _parse_job(fields: dict[str, str]) -> JobEvent | None:
    if fields.get("init") == "true":
        return None
    
    job_id = fields.get("jobId") or fields.get("job_id")

    if not job_id:
        # 🔥 Bỏ qua message không hợp lệ
        logger.warning(f"Skip invalid message: {fields}")
        return None
    
    raw_data = fields.get("data")
    duty = fields.get("duty")

    if not job_id:
        raise ValueError("Missing jobId")
    if not raw_data and duty and duty.lower() != "notification":
        raise ValueError("Missing data")
    if not duty:
        raise ValueError("Missing duty")

    # ✅ FIX: parse JSON nếu cần
    file_url = raw_data

    if raw_data and raw_data.startswith("{"):
        try:
            parsed = json.loads(raw_data)
            file_url = parsed.get("imageUrl")
        except Exception:
            raise ValueError(f"Invalid JSON data: {raw_data}")

    # ✅ validate URL (skip for notification duty)
    duty_lower = duty.lower() if duty else ""
    if duty_lower != "notification":
        if not file_url or not file_url.startswith("http"):
            raise ValueError(f"Invalid file_url: {file_url}")

    print(f"Parsed jobId={job_id} file_url={file_url} duty={duty}")

    return JobEvent(
        job_id=str(job_id),
        user_id=fields.get("userId") or fields.get("user_id"),
        file_url=file_url,
        duty=str(duty),
        created_at=fields.get("createdAt") or fields.get("created_at"),
    )

async def _ensure_consumer_group(r: redis.Redis, stream_key: str, group: str) -> None:
    try:
        await r.xgroup_create(stream_key, group, id="$", mkstream=True)
        logger.info("Created consumer group %s for stream %s", group, stream_key)
    except ResponseError as e:
        msg = str(e)
        if "BUSYGROUP" in msg:
            return
        raise


async def _read_pending_one(
    r: redis.Redis,
    stream_key: str,
    group: str,
    consumer: str,
    min_idle_ms: int,
) -> tuple[str, dict[str, str]] | None:
    # Prefer retrying pending messages before reading new ones.
    # redis-py returns: (next_start_id, [(message_id, {field: value}), ...], deleted_ids)
    try:
        result = await r.xautoclaim(
            stream_key,
            group,
            consumer,
            min_idle_ms,
            start_id="0-0",
            count=1,
        )
    except ResponseError as e:
        # Older Redis versions may not support XAUTOCLAIM.
        logger.warning("XAUTOCLAIM not available: %s", e)
        return None

    if not result:
        return None

    # Handle both list/tuple and dict shapes defensively.
    if isinstance(result, (list, tuple)) and len(result) >= 2:
        messages = result[1]
    elif isinstance(result, dict):
        messages = result.get("messages") or []
    else:
        messages = []

    if not messages:
        return None

    message_id, fields_any = messages[0]
    if isinstance(message_id, bytes):
        message_id = message_id.decode("utf-8", errors="replace")
    fields = _decode_stream_fields(fields_any)
    return str(message_id), fields


async def _read_new_one(
    r: redis.Redis,
    stream_key: str,
    group: str,
    consumer: str,
    block_ms: int,
) -> tuple[str, dict[str, str]] | None:
    entries = await r.xreadgroup(
        groupname=group,
        consumername=consumer,
        streams={stream_key: ">"},
        count=1,
        block=block_ms,
    )

    if not entries:
        return None

    _stream_name, stream_entries = entries[0]
    if not stream_entries:
        return None

    message_id, fields_any = stream_entries[0]
    if isinstance(message_id, bytes):
        message_id = message_id.decode("utf-8", errors="replace")
    fields = _decode_stream_fields(fields_any)
    return str(message_id), fields


async def _times_delivered(
    r: redis.Redis,
    stream_key: str,
    group: str,
    message_id: str,
) -> int | None:
    try:
        pendings = await r.xpending_range(
            stream_key,
            group,
            min=message_id,
            max=message_id,
            count=1,
        )
    except Exception:
        return None

    if not pendings:
        return None

    p0 = pendings[0]
    if isinstance(p0, dict):
        td = p0.get("times_delivered")
        try:
            return int(td) if td is not None else None
        except Exception:
            return None

    # Some versions return tuples: (message_id, consumer, idle, delivered)
    if isinstance(p0, (list, tuple)) and len(p0) >= 4:
        try:
            return int(p0[3])
        except Exception:
            return None

    return None


async def _dead_letter(
    r: redis.Redis,
    dead_letter_stream_key: str,
    original_stream: str,
    message_id: str,
    fields: dict[str, str],
    error: str,
) -> None:
    if not dead_letter_stream_key:
        return

    payload: dict[str, str] = {
        **fields,
        "_originalStream": original_stream,
        "_originalId": message_id,
        "_error": error,
    }
    await r.xadd(dead_letter_stream_key, payload)

def enhance_cloudinary_url(url: str) -> str:
    return url.replace("/upload/", "/upload/q_100/")


async def _publish_extraction_error(
    input_redis: redis.Redis,
    result_redis: redis.Redis,
    stream_key: str,
    group: str,
    message_id: str,
    job_id: str,
    error_msg: str,
) -> None:
    """Publish an error result for a job whose content cannot be extracted (e.g. blank/blur image).

    Writes to result_stream and the job key so consumers get an immediate response,
    then ACKs the message so it is never retried (content quality won't improve on retry).
    """
    payload = {
        "jobId": job_id,
        "error": error_msg,
        "text": "",
        "expense": _DEFAULT_EXPENSE_VND,
        "type": "EXPENSE",
        "category": "OTHER",
        "confidence": 0.0,
    }
    await result_redis.xadd("result_stream", payload, maxlen=10000, approximate=True)
    result_key = f"job:{job_id}"
    ttl = int(settings.RESULT_TTL_SECONDS)
    if ttl > 0:
        await result_redis.setex(result_key, ttl, json.dumps(payload, ensure_ascii=False))
    else:
        await result_redis.set(result_key, json.dumps(payload, ensure_ascii=False))
    await input_redis.xack(stream_key, group, message_id)
async def _handle_ocr_duty(image_extractor, download) -> str:
    """Extract text from image using OCR."""
    ocr_result = await image_extractor.extract_bytes(download.content)
    if ocr_result.get("error"):
        raise RuntimeError(str(ocr_result.get("error")))
    return str(ocr_result.get("text") or "")


async def _handle_voice_duty(voice_extractor, download) -> str:
    """Extract text from audio using ASR."""
    asr_result = await voice_extractor.extract_bytes(
        download.content, content_type=download.content_type
    )
    if asr_result.get("error"):
        raise RuntimeError(str(asr_result.get("error")))
    return str(asr_result.get("text") or "")


async def _handle_notification_duty(fields: dict[str, str]) -> str:
    """Extract text from notification (text is already provided)."""
    # Check multiple possible field names
    text = (
        fields.get("text") 
        or fields.get("message") 
        or fields.get("data")
        or ""
    )
    if not text:
        raise ValueError(
            f"Missing text/message/data in notification duty. Available fields: {list(fields.keys())}"
        )
    return str(text)

def _extract_amount_from_notification(text: str) -> int | None:
    if not text:
        return None

    normalized = text.lower()

    # 🔥 STEP 1: ưu tiên số có dấu (transaction thật)
    signed_pattern = r"([+-]\d[\d.,]*)\s*(?:vnd|vnđ|đ)\b"
    match = re.search(signed_pattern, normalized)

    if match:
        raw = match.group(1)
        sign = -1 if raw.startswith("-") else 1
        digits = re.sub(r"[^\d]", "", raw)

        if digits:
            return int(digits) * sign

    # 🔥 STEP 2: fallback (nếu không có dấu)
    fallback_patterns = [
        r"(?:gd|so tien|số tiền)[:\s]+([+-]?\d[\d.,]*)",
    ]

    for pattern in fallback_patterns:
        match = re.search(pattern, normalized)
        if match:
            digits = re.sub(r"[^\d]", "", match.group(1))
            if digits:
                return int(digits)

    return None

async def _extract_text_by_duty(
    job: JobEvent,
    download,
    image_extractor,
    voice_extractor,
    fields: dict[str, str],
) -> str:
    """Route to appropriate handler based on duty type."""
    duty_lower = job.duty.lower()
    
    duty_handlers = {
        "ocr": lambda: _handle_ocr_duty(image_extractor, download),
        "voice": lambda: _handle_voice_duty(voice_extractor, download),
        "notification": lambda: _handle_notification_duty(fields),
    }
    
    if duty_lower not in duty_handlers:
        raise ValueError(f"Unsupported duty: {job.duty}")
    
    return await duty_handlers[duty_lower]()


async def _process_one(
    input_redis: redis.Redis,
    result_redis: redis.Redis,
    fetcher,
    classifier,
    image_extractor,
    voice_extractor,
    llm_service,
    stream_key: str,
    group: str,
    message_id: str,
    fields: dict[str, str],
) -> None:
      job = _parse_job(fields)

      if job is None:
          await input_redis.xack(stream_key, group, message_id)
          logger.info("Skipped init message id=%s", message_id)
          return

      download = None

      if job.duty.lower() in ("ocr", "voice"):
          enhanced_url = enhance_cloudinary_url(job.file_url)
          download = await fetcher.fetch(enhanced_url)

      try:
          extracted_text = await _extract_text_by_duty(
              job,
              download,
              image_extractor,
              voice_extractor,
              fields,
          )
      except Exception as e:
          await _publish_extraction_error(
              input_redis,
              result_redis,
              stream_key,
              group,
              message_id,
              job.job_id,
              str(e),
          )
          return

      if not extracted_text.strip():
          await _publish_extraction_error(
              input_redis,
              result_redis,
              stream_key,
              group,
              message_id,
              job.job_id,
              "No readable text content found in the uploaded file",
          )
          return

      if job.duty.lower() == "notification":
          expense = _extract_amount_from_notification(extracted_text)

          if expense is None:
              expense = await _extract_expense_vnd(
                  llm_service,
                  extracted_text,
              )
      else:
          expense = await _extract_expense_vnd(
              llm_service,
              extracted_text,
          )

      transaction_type = _detect_transaction_type_rule(extracted_text)

      if not transaction_type:
          transaction_type = await _classify_transaction_type(llm_service, extracted_text)

      if not transaction_type:
          transaction_type = "EXPENSE"

      final_category = "OTHER"
      confidence = 0.0
      try:
          classification = classifier.classify(extracted_text)
          if classification.error or classification.category is None:
              logger.warning("Classifier returned no category for job %s: %s", job.job_id, classification.error)
              llm_category = await _classify_category_with_llm(llm_service, extracted_text)
              if llm_category:
                  final_category = llm_category
          else:
              final_category = classification.category.value
              confidence = float(classification.confidence)
              if confidence < _CATEGORY_LLM_FALLBACK_THRESHOLD:
                  llm_category = await _classify_category_with_llm(llm_service, extracted_text)
                  if llm_category:
                      final_category = llm_category
      except Exception as e:
          logger.warning("Classifier raised for job %s: %s — falling back to LLM", job.job_id, e)
          llm_category = await _classify_category_with_llm(llm_service, extracted_text)
          if llm_category:
              final_category = llm_category

      result_key = f"job:{job.job_id}"
      payload = {
          "jobId": job.job_id,
          "userId": job.user_id,
          "text": extracted_text,
          "expense": abs(expense) if expense else _DEFAULT_EXPENSE_VND, 
          "type": transaction_type or "EXPENSE",
          "category": final_category,
          "confidence": confidence,
      }


      await result_redis.xadd("result_stream", payload, maxlen=10000, approximate=True)

      ttl = int(settings.RESULT_TTL_SECONDS)
      if ttl > 0:
          await result_redis.setex(
              result_key,
              ttl,
              json.dumps(payload, ensure_ascii=False),
          )
      else:
          await result_redis.set(result_key, json.dumps(payload, ensure_ascii=False))

      await input_redis.xack(stream_key, group, message_id)


async def run_worker_forever() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    consumer_name = settings.REDIS_CONSUMER_NAME
    if consumer_name == "smartmoney-ai-1":
        consumer_name = f"{socket.gethostname()}-{os.getpid()}"

    stream_key = settings.REDIS_STREAM_KEY
    group = settings.REDIS_CONSUMER_GROUP

    async def connect() -> tuple[redis.Redis, redis.Redis]:
        in_r = _redis_from_url_checked(settings.REDIS_STREAM_URL, "REDIS_STREAM_URL")
        out_r = _redis_from_url_checked(settings.REDIS_RESULT_URL, "REDIS_RESULT_URL")
        await in_r.ping()
        await out_r.ping()
        await _ensure_consumer_group(in_r, stream_key, group)
        return in_r, out_r

    input_redis, result_redis = await connect()

    fetcher = get_cloudinary_fetcher()

    from app.services.extractor.voice_extractor import VoiceExtractor

    voice_extractor = VoiceExtractor()  # preloads ASR model at worker startup

    from app.services.llm import get_llm_service

    llm_service = get_llm_service()

    from app.services.classifer.classifier import get_classifier_service
    from app.services.extractor.image_extractor import ImageExtractor

    classifier = get_classifier_service()
    image_extractor = ImageExtractor()

    logger.info(
        "Worker started. stream=%s group=%s consumer=%s",
        stream_key,
        group,
        consumer_name,
    )

    while True:
        try:
            # 1) Try to reclaim and process one idle pending message.
            item = await _read_pending_one(
                input_redis,
                stream_key,
                group,
                consumer_name,
                min_idle_ms=int(settings.REDIS_CLAIM_IDLE_MS),
            )

            # 2) If none pending, read one new message.
            if item is None:
                item = await _read_new_one(
                    input_redis,
                    stream_key,
                    group,
                    consumer_name,
                    block_ms=int(settings.REDIS_BLOCK_MS),
                )

            if item is None:
                continue
        except (RedisConnectionError, OSError) as e:
            logger.warning("Redis connection dropped (%s). Reconnecting...", e)
            await _close_redis_client(input_redis)
            await _close_redis_client(result_redis)

            backoff = 1.0
            while True:
                try:
                    input_redis, result_redis = await connect()
                    logger.info("Reconnected to Redis")
                    break
                except Exception as e2:
                    logger.warning("Reconnect failed (%s). Retrying in %.1fs", e2, backoff)
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2.0, 30.0)
            continue

        message_id, fields = item

        try:
            await _process_one(
                input_redis,
                result_redis,
                fetcher,
                classifier,
                image_extractor,
                voice_extractor,
                llm_service,
                stream_key,
                group,
                message_id,
                fields,
            )
            logger.info("Processed job stream id=%s jobId=%s", message_id, fields.get("jobId"))

        except ValueError as e:

            err = str(e)

            logger.exception(
                "Malformed message stream id=%s error=%s",
                message_id,
                err,
            )

            await _dead_letter(
                input_redis,
                settings.REDIS_DEAD_LETTER_STREAM_KEY,
                stream_key,
                message_id,
                fields,
                err,
            )

            await input_redis.xack(
                stream_key,
                group,
                message_id,
            )

            continue

        except Exception as e:

            err = str(e)
            job_id = fields.get("jobId")

            times_delivered = await _times_delivered(
                input_redis,
                stream_key,
                group,
                message_id,
            )

            logger.exception(
                "Failed processing stream id=%s jobId=%s delivered=%s error=%s",
                message_id,
                job_id,
                times_delivered,
                err,
            )

            max_retries = int(settings.REDIS_MAX_RETRIES)

            if (
                times_delivered is not None
                and max_retries > 0
                and times_delivered >= max_retries
            ):
                await _dead_letter(
                    input_redis,
                    settings.REDIS_DEAD_LETTER_STREAM_KEY,
                    stream_key,
                    message_id,
                    fields,
                    err,
                )

                await input_redis.xack(
                    stream_key,
                    group,
                    message_id,
                )

            await asyncio.sleep(0.5)

def main() -> None:
    asyncio.run(run_worker_forever())


if __name__ == "__main__":
    main()
