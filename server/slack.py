"""Post video-generation results into Slack via the bot token.

We use the bot's Web API directly (`chat.postMessage` + `files.upload_v2`)
rather than an incoming webhook because webhooks can't attach files.

Auth: WAN_SLACK_BOT_TOKEN env var (xoxb-*). The same token the bot that
receives the @mention uses — one app, one token, two directions.

Optional safety: WAN_SLACK_ALLOWED_CHANNELS=C0123,C0456 restricts which
channels the server will post to. Empty or unset = allow any channel
that shows up in the callback metadata.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import httpx


log = logging.getLogger("server.slack")

_SLACK = "https://slack.com/api"
_TIMEOUT = 60.0


def _token() -> str:
    t = os.environ.get("WAN_SLACK_BOT_TOKEN", "")
    if not t:
        raise RuntimeError(
            "WAN_SLACK_BOT_TOKEN is not set; cannot post to Slack"
        )
    return t


def _allowed_channels() -> Optional[set[str]]:
    spec = os.environ.get("WAN_SLACK_ALLOWED_CHANNELS", "").strip()
    if not spec:
        return None
    return {c.strip() for c in spec.split(",") if c.strip()}


def channel_allowed(channel: str) -> bool:
    whitelist = _allowed_channels()
    return whitelist is None or channel in whitelist


async def post_message(
    channel: str,
    text: str,
    thread_ts: Optional[str] = None,
) -> dict:
    """Post a plain text message via chat.postMessage."""
    headers = {
        "Authorization": f"Bearer {_token()}",
        "Content-Type": "application/json; charset=utf-8",
    }
    body: dict = {"channel": channel, "text": text}
    if thread_ts:
        body["thread_ts"] = thread_ts
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        r = await client.post(
            f"{_SLACK}/chat.postMessage", headers=headers, json=body
        )
    d = r.json()
    if not d.get("ok"):
        log.warning("chat.postMessage failed: %s", d)
    return d


async def upload_video(
    channel: str,
    video_path: str,
    title: str,
    initial_comment: str,
    thread_ts: Optional[str] = None,
) -> dict:
    """Upload an mp4 to Slack via the files.upload_v2 three-step dance:

    1. files.getUploadURLExternal -> {upload_url, file_id}
    2. POST bytes to upload_url
    3. files.completeUploadExternal -> share to channel
    """
    size = os.path.getsize(video_path)
    filename = os.path.basename(video_path)
    hdr = {"Authorization": f"Bearer {_token()}"}

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        # 1) signed upload URL
        r = await client.post(
            f"{_SLACK}/files.getUploadURLExternal",
            headers=hdr,
            data={"filename": filename, "length": str(size)},
        )
        step1 = r.json()
        if not step1.get("ok"):
            log.warning("files.getUploadURLExternal failed: %s", step1)
            return step1
        upload_url = step1["upload_url"]
        file_id = step1["file_id"]

        # 2) upload bytes (no Slack auth header on this step — it's a
        # presigned URL)
        with open(video_path, "rb") as f:
            r = await client.post(upload_url, content=f.read())
        if r.status_code >= 300:
            err = {"ok": False, "step": "upload", "status": r.status_code,
                   "body": r.text[:200]}
            log.warning("upload POST failed: %s", err)
            return err

        # 3) complete + share
        payload: dict = {
            "files": [{"id": file_id, "title": title[:250] or filename}],
            "channel_id": channel,
            "initial_comment": initial_comment,
        }
        if thread_ts:
            payload["thread_ts"] = thread_ts
        r = await client.post(
            f"{_SLACK}/files.completeUploadExternal",
            headers={**hdr, "Content-Type": "application/json; charset=utf-8"},
            json=payload,
        )
        step3 = r.json()
        if not step3.get("ok"):
            log.warning("files.completeUploadExternal failed: %s", step3)
        return step3


async def post_video_result(
    *,
    channel: str,
    user: str,
    thread_ts: Optional[str],
    prompt: str,
    video_path: Optional[str],
    error: Optional[str],
) -> dict:
    """Top-level entrypoint. Posts failure message OR uploads video."""
    if not channel_allowed(channel):
        log.warning(
            "refusing to post to channel=%s (not in "
            "WAN_SLACK_ALLOWED_CHANNELS)", channel,
        )
        return {"ok": False, "error": "channel not allowed"}

    if error or not video_path or not os.path.exists(video_path):
        msg = (
            f":x: <@{user}> generation failed: `{(error or 'missing video')[:250]}`"
            if error
            else f":x: <@{user}> generation completed but the video file is missing"
        )
        return await post_message(channel, msg, thread_ts=thread_ts)

    comment = f":tada: <@{user}> your video is ready: *{prompt[:250]}*"
    return await upload_video(
        channel=channel,
        video_path=video_path,
        title=prompt,
        initial_comment=comment,
        thread_ts=thread_ts,
    )
