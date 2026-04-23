"""Callback URL validation + delivery with retries.

The delivery task is fire-and-forget from the worker's perspective: we
call asyncio.create_task(deliver_callback(...)) after marking a job
done/failed, and the worker immediately picks up the next queued job
without waiting for the webhook to return.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
from typing import Any, Optional
from urllib.parse import urlparse

import httpx

from server.jobstore import Job, JobStore


log = logging.getLogger("server.callbacks")


_BLOCKED_NETS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),   # link-local
    ipaddress.ip_network("10.0.0.0/8"),       # RFC1918
    ipaddress.ip_network("172.16.0.0/12"),    # RFC1918
    ipaddress.ip_network("192.168.0.0/16"),   # RFC1918
    ipaddress.ip_network("0.0.0.0/8"),        # unspecified
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("fc00::/7"),
]

_HOP_BY_HOP = {
    "host", "content-length", "transfer-encoding", "connection",
    "keep-alive", "upgrade", "proxy-connection",
}

_RETRY_DELAYS = (2.0, 10.0, 60.0)  # 4 total attempts (initial + 3 retries)
_REQUEST_TIMEOUT = 30.0
_MAX_HEADERS_BYTES = 4096


class CallbackValidationError(ValueError):
    pass


def redact_url(url: str) -> str:
    """Log-safe representation: scheme + host + short path, no query/fragment."""
    if not url:
        return "<empty>"
    try:
        p = urlparse(url)
        path = p.path or ""
        if len(path) > 30:
            path = path[:30] + "..."
        return f"{p.scheme}://{p.hostname or '?'}{path}"
    except Exception:
        return "<unparseable>"


def validate_callback_url(
    url: str,
    allow_private: bool = False,
) -> None:
    """Raise CallbackValidationError on anything suspicious.

    Checks scheme, resolves the hostname, rejects RFC1918/loopback/link-local
    unless allow_private is set.
    """
    if not url:
        return  # allowed to be absent
    try:
        parsed = urlparse(url)
    except Exception as e:
        raise CallbackValidationError(f"unparseable URL: {e}") from e
    if parsed.scheme not in ("http", "https"):
        raise CallbackValidationError(
            f"scheme must be http or https, got {parsed.scheme!r}"
        )
    host = parsed.hostname
    if not host:
        raise CallbackValidationError("missing hostname")

    if allow_private:
        return

    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise CallbackValidationError(f"DNS resolution failed: {e}") from e
    for family, _, _, _, sockaddr in infos:
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        for net in _BLOCKED_NETS:
            if ip.version == net.version and ip in net:
                raise CallbackValidationError(
                    f"callback host resolves to blocked address {ip} "
                    f"(in {net}); set WAN_DEMO_ALLOW_PRIVATE_CALLBACK=1 "
                    f"to override for internal testing"
                )


def validate_callback_headers(headers: Optional[dict[str, str]]) -> None:
    if not headers:
        return
    total = 0
    for k, v in headers.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise CallbackValidationError(
                "callback_headers keys and values must be strings"
            )
        if k.lower() in _HOP_BY_HOP:
            raise CallbackValidationError(
                f"callback header {k!r} is hop-by-hop; not forwardable"
            )
        total += len(k) + len(v)
    if total > _MAX_HEADERS_BYTES:
        raise CallbackValidationError(
            f"callback_headers too large ({total} > {_MAX_HEADERS_BYTES} bytes)"
        )


def build_callback_payload(
    job: Job,
    video_base_url: str,
) -> dict[str, Any]:
    """Build the POST body sent to the caller's webhook."""
    video_url: Optional[str] = None
    if job.status == "done" and job.video_path:
        video_url = (
            f"{video_base_url.rstrip('/')}/v1/generations/{job.id}/video"
            f"?token={job.video_token}"
        )
    return {
        "job_id": job.id,
        "status": job.status,
        "video_url": video_url,
        "prompt": job.prompt,
        "finished_at": job.finished_at,
        "error": job.error,
        "metadata": job.callback_metadata or {},
    }


async def deliver_callback(
    store: JobStore,
    job: Job,
    video_base_url: str,
) -> None:
    """POST the callback with retries; record final status in the DB.

    Never raises — all errors are caught and logged.
    """
    if not job.callback_url:
        return
    payload = build_callback_payload(job, video_base_url)
    headers = dict(job.callback_headers or {})
    headers.setdefault("Content-Type", "application/json")
    redacted = redact_url(job.callback_url)
    header_keys = sorted(headers.keys())

    last_err: Optional[str] = None
    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
        for attempt in range(1 + len(_RETRY_DELAYS)):
            try:
                resp = await client.post(
                    job.callback_url, json=payload, headers=headers
                )
                if 200 <= resp.status_code < 300:
                    log.info(
                        "callback delivered job=%s attempt=%d status=%d url=%s",
                        job.id, attempt + 1, resp.status_code, redacted,
                    )
                    await store.update_callback(
                        job.id, "delivered", attempt + 1
                    )
                    return
                last_err = f"HTTP {resp.status_code}"
            except httpx.HTTPError as e:
                last_err = f"{type(e).__name__}: {e}"[:200]
            except Exception as e:   # defensive
                last_err = f"{type(e).__name__}: {e}"[:200]

            log.warning(
                "callback attempt failed job=%s attempt=%d url=%s header_keys=%s err=%s",
                job.id, attempt + 1, redacted, header_keys, last_err,
            )
            if attempt < len(_RETRY_DELAYS):
                await asyncio.sleep(_RETRY_DELAYS[attempt])

    log.warning(
        "callback giving up job=%s attempts=%d url=%s err=%s",
        job.id, 1 + len(_RETRY_DELAYS), redacted, last_err,
    )
    await store.update_callback(
        job.id, "failed", 1 + len(_RETRY_DELAYS), last_err
    )
