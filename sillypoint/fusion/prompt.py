"""
sillypoint/fusion/prompt.py
────────────────────────────
Builds the LLM prompt that fuses audio transcription + visual events into
structured ball-by-ball cricket events.

The design principle: give the model *evidence* and ask it to *reason*,
not pattern-match.  Each piece of evidence carries its own timestamp so
the model can align transcript fragments with visual detections.

Usage
-----
    from sillypoint.fusion.prompt import build_messages
    from sillypoint.types import TranscriptSegment, VisualEvent

    messages = build_messages(transcript=segments, visual_events=events)
    # → pass directly to anthropic.messages.create(messages=messages, ...)
"""

from __future__ import annotations

import json
from typing import Sequence

from sillypoint.types import TranscriptSegment, VisualEvent


# ──────────────────────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a senior cricket analyst and rules expert.  Your job is to receive
two streams of evidence from a video clip — a timestamped speech transcript
from the on-field umpire(s) and a sequence of visual detections (ball
position, batsman pose, umpire arm signals) — and produce a structured,
ball-by-ball event log.

## Rules you must follow

1. **Ground every claim in evidence.**  For each ball event you emit, cite
   the specific transcript excerpt and/or visual detection that supports it.
   Do not invent signals or outcomes.

2. **Resolve conflicts explicitly.**  When the transcript and visual evidence
   disagree (e.g. umpire says "four" but ball trajectory shows it hit the
   boundary rope), note the conflict in the `reasoning` field and favour the
   visual evidence, unless the transcript is unusually clear.

3. **Use standard cricket terminology** for outcomes:
   dot_ball | single | two | three | four | six | wide | no_ball |
   wicket_bowled | wicket_lbw | wicket_caught | wicket_runout |
   wicket_stumped | dead_ball | penalty_runs | unknown

4. **Confidence scores** reflect how much unambiguous evidence supports your
   classification.  A ball with a clear umpire signal + clean transcript
   excerpt should be ≥ 0.90.  Ambiguous or silent balls should be ≤ 0.60.

5. **Emit one JSON object per delivery** (including wides, no-balls, and
   dead balls).  Do not merge deliveries.

6. **If you cannot identify a ball boundary** (e.g. the clip starts mid-over
   and you cannot tell where one delivery ends and the next begins), emit a
   single event with outcome "unknown" and explain in `reasoning`.

## Output schema

Return a JSON array.  Each element must conform to:

```json
{
  "ball_number": 1,
  "timestamp": "MM:SS.s",
  "outcome": "<outcome_term>",
  "runs": 0,
  "umpire_signal": "<signal_or_null>",
  "transcript_excerpt": "<verbatim words or null>",
  "ball_trajectory": [
    {"t": 0.0, "x": 0.41, "y": 0.72, "confidence": 0.91}
  ],
  "reasoning": "<one or two sentences explaining the classification>",
  "confidence": 0.87
}
```

Field notes:
- `ball_trajectory`: normalised (0–1) frame coordinates, sampled at ~10 fps.
  May be an empty list if ball tracking failed.
- `umpire_signal`: one of  no_ball | wide | four | six | out | not_out |
  dead_ball | penalty | null
- `runs`: integer.  For a wicket delivery, record runs scored before the
  wicket fell (usually 0).
- `timestamp`: the moment the ball is *bowled* (not when it lands or the
  umpire signals).

Return ONLY the JSON array.  No markdown fences, no preamble.
"""


# ──────────────────────────────────────────────────────────────
# User-turn builder
# ──────────────────────────────────────────────────────────────

_EVIDENCE_TEMPLATE = """\
## Transcript
{transcript_block}

## Visual events
{visual_block}

Analyse the evidence above and return the ball-by-ball JSON array.
"""


def _format_transcript(segments: Sequence[TranscriptSegment]) -> str:
    """Render transcript segments as a timestamped list."""
    if not segments:
        return "(no transcript available)"
    lines = []
    for seg in segments:
        start = _fmt_ts(seg.start)
        end = _fmt_ts(seg.end)
        lines.append(f"[{start} → {end}]  {seg.text.strip()}")
    return "\n".join(lines)


def _format_visual(events: Sequence[VisualEvent]) -> str:
    """Render visual detections as compact JSON lines, one per event."""
    if not events:
        return "(no visual events detected)"
    # One JSON object per line keeps the context window lean.
    return "\n".join(json.dumps(e.model_dump(), separators=(",", ":")) for e in events)


def _fmt_ts(seconds: float) -> str:
    """Convert float seconds → MM:SS.s display string."""
    minutes = int(seconds) // 60
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:04.1f}"


def build_messages(
    transcript: Sequence[TranscriptSegment],
    visual_events: Sequence[VisualEvent],
) -> list[dict]:
    """
    Build the `messages` list ready for the Anthropic or Gemini chat API.

    Parameters
    ----------
    transcript:
        Word/segment-level output from the Whisper transcription step.
    visual_events:
        Frame-level detections from the YOLO + pose pipeline.

    Returns
    -------
    list[dict]
        A two-element list: the system turn and the user turn, structured
        for ``anthropic.messages.create(messages=...)``.

    Example
    -------
    >>> import anthropic, os
    >>> client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    >>> messages = build_messages(transcript=segs, visual_events=events)
    >>> response = client.messages.create(
    ...     model=os.environ.get("FUSION_MODEL", "claude-opus-4-6"),
    ...     max_tokens=4096,
    ...     system=messages[0]["content"],
    ...     messages=messages[1:],
    ... )
    >>> ball_events = json.loads(response.content[0].text)
    """
    transcript_block = _format_transcript(transcript)
    visual_block = _format_visual(visual_events)

    user_content = _EVIDENCE_TEMPLATE.format(
        transcript_block=transcript_block,
        visual_block=visual_block,
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ──────────────────────────────────────────────────────────────
# Prompt introspection helper (useful during development)
# ──────────────────────────────────────────────────────────────

def estimate_tokens(transcript: Sequence[TranscriptSegment], visual_events: Sequence[VisualEvent]) -> int:
    """
    Rough token estimate (≈ chars / 4) so callers can check they won't blow
    the context window before making the API call.
    """
    messages = build_messages(transcript, visual_events)
    total_chars = sum(len(m["content"]) for m in messages)
    return total_chars // 4
