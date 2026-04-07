from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
from dataclasses import dataclass
from typing import Any

import redis.asyncio as redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError

from app.core.config import settings
from app.services.fetcher.cloudinary_fetcher import get_cloudinary_fetcher

logger = logging.getLogger(__name__)


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


def _parse_job(fields: dict[str, str]) -> JobEvent:
    job_id = fields.get("jobId") or fields.get("job_id")
    raw_data = fields.get("data")
    duty = fields.get("duty")

    if not job_id:
        raise ValueError("Missing jobId")
    if not raw_data:
        raise ValueError("Missing data")
    if not duty:
        raise ValueError("Missing duty")

    # ✅ FIX: parse JSON nếu cần
    file_url = raw_data

    if raw_data.startswith("{"):
        try:
            parsed = json.loads(raw_data)
            file_url = parsed.get("imageUrl")
        except Exception:
            raise ValueError(f"Invalid JSON data: {raw_data}")

    # ✅ validate URL
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

async def _process_one(
    input_redis: redis.Redis,
    result_redis: redis.Redis,
    fetcher,
    classifier,
    image_extractor,
    voice_extractor,
    stream_key: str,
    group: str,
    message_id: str,
    fields: dict[str, str],
) -> None:
    job = _parse_job(fields)

    enhanced_url = enhance_cloudinary_url(job.file_url)    

    download = await fetcher.fetch(enhanced_url)

    extracted_text = ""
    if job.duty.lower() == "ocr":
        ocr_result = await image_extractor.extract_bytes(download.content)
        if ocr_result.get("error"):
            raise RuntimeError(str(ocr_result.get("error")))
        extracted_text = str(ocr_result.get("text") or "")

    elif job.duty.lower() == "voice":
        asr_result = await voice_extractor.extract_bytes(download.content, content_type=download.content_type)
        if asr_result.get("error"):
            raise RuntimeError(str(asr_result.get("error")))
        extracted_text = str(asr_result.get("text") or "")

    else:
        raise ValueError(f"Unsupported duty: {job.duty}")

    classification = classifier.classify(extracted_text)

    result_key = f"job:{job.job_id}"
    payload = {
        "jobId": job.job_id,
        "text": extracted_text,
        "category": classification.category.value,
        "confidence": classification.confidence,
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

    # Important on Windows: load Faster-Whisper/CTranslate2 before OCR/classifier
    # to avoid native DLL/OpenMP runtime conflicts (WinError 127).
    from app.services.extractor.voice_extractor import VoiceExtractor

    voice_extractor = VoiceExtractor()  # preloads ASR model at worker startup

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
                stream_key,
                group,
                message_id,
                fields,
            )
            logger.info("Processed job stream id=%s jobId=%s", message_id, fields.get("jobId"))

        except Exception as e:
            err = str(e)
            job_id = fields.get("jobId")
            times_delivered = await _times_delivered(input_redis, stream_key, group, message_id)

            logger.exception(
                "Failed processing stream id=%s jobId=%s delivered=%s error=%s",
                message_id,
                job_id,
                times_delivered,
                err,
            )

            max_retries = int(settings.REDIS_MAX_RETRIES)
            if times_delivered is not None and max_retries > 0 and times_delivered >= max_retries:
                await _dead_letter(
                    input_redis,
                    settings.REDIS_DEAD_LETTER_STREAM_KEY,
                    stream_key,
                    message_id,
                    fields,
                    err,
                )
                await input_redis.xack(stream_key, group, message_id)

            # If not dead-lettered, leave unacked so it can be retried later.
            await asyncio.sleep(0.5)


def main() -> None:
    asyncio.run(run_worker_forever())


if __name__ == "__main__":
    main()
