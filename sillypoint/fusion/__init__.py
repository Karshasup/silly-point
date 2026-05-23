"""
sillypoint/fusion/__init__.py
──────────────────────────────
Calls the LLM (Claude by default) with the prompt built in fusion/prompt.py
and parses the structured JSON response into typed BallEvent objects.

Public API
----------
    from sillypoint.fusion import fuse
    balls = fuse(transcript, visual_events)
"""

from __future__ import annotations

import json
import os

import anthropic

from sillypoint.fusion.prompt import build_messages
from sillypoint.types import BallEvent, TranscriptSegment, VisualEvent

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    return _client


def fuse(
    transcript: list[TranscriptSegment],
    visual_events: list[VisualEvent],
) -> list[BallEvent]:
    """
    Fuse audio transcript + visual detections into structured ball events.

    Sends both evidence streams to the LLM and parses the JSON array it
    returns. Pydantic validation on BallEvent will raise if the model
    returns malformed data.
    """
    messages = build_messages(transcript, visual_events)
    model    = os.getenv("FUSION_MODEL", "claude-opus-4-6")

    response = _get_client().messages.create(
        model=model,
        max_tokens=4096,
        system=messages[0]["content"],
        messages=messages[1:],
    )

    raw = json.loads(response.content[0].text)
    return [BallEvent(**ball) for ball in raw]
