from __future__ import annotations

from dataclasses import dataclass

import httpx

from app.core.config import settings


@dataclass(frozen=True)
class FetchResult:
    content: bytes
    content_type: str | None


class CloudinaryFetcher:
    async def fetch(self, url: str) -> FetchResult:
        if not url or not isinstance(url, str):
            raise ValueError("data is required")

        timeout = httpx.Timeout(settings.DOWNLOAD_TIMEOUT_SECONDS)
        limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)

        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, limits=limits) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                content_type = resp.headers.get("content-type")

                buf = bytearray()
                max_bytes = int(settings.DOWNLOAD_MAX_BYTES)

                async for chunk in resp.aiter_bytes():
                    if not chunk:
                        continue
                    buf.extend(chunk)
                    if max_bytes > 0 and len(buf) > max_bytes:
                        raise ValueError(f"Downloaded file too large: {len(buf)} bytes")

                return FetchResult(content=bytes(buf), content_type=content_type)


def get_cloudinary_fetcher() -> CloudinaryFetcher:
    # lightweight; no need for caching/singleton complexity
    return CloudinaryFetcher()
