"""
sillypoint/types.py
────────────────────
Shared Pydantic models used across all pipeline stages.
These are the canonical data contracts between audio/, vision/, and fusion/.
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────
# Audio pipeline output
# ──────────────────────────────────────────────────────────────

class TranscriptSegment(BaseModel):
    """One segment (word or phrase) from the Whisper transcription."""

    start: float = Field(..., description="Segment start time in seconds")
    end: float = Field(..., description="Segment end time in seconds")
    text: str = Field(..., description="Transcribed text for this segment")
    speaker: Optional[str] = Field(
        None,
        description="Speaker label if diarisation is available (e.g. 'umpire_leg')",
    )
    word_confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Per-word confidence from Whisper (if available)",
    )


# ──────────────────────────────────────────────────────────────
# Vision pipeline output
# ──────────────────────────────────────────────────────────────

class BallPosition(BaseModel):
    """Ball detection at a single frame."""

    t: float = Field(..., description="Timestamp in seconds")
    x: float = Field(..., ge=0.0, le=1.0, description="Normalised x coordinate (0 = left)")
    y: float = Field(..., ge=0.0, le=1.0, description="Normalised y coordinate (0 = top)")
    confidence: float = Field(..., ge=0.0, le=1.0)


UmpireSignalLabel = Literal[
    "no_ball", "wide", "four", "six", "out", "not_out",
    "dead_ball", "penalty", "unknown",
]

UmpirePoseLabel = Literal[
    "arms_crossed",      # out
    "arm_extended",      # wide or no-ball
    "both_arms_raised",  # six
    "arm_swept_low",     # four
    "index_raised",      # one short
    "neutral",
    "unknown",
]


class VisualEvent(BaseModel):
    """
    A single visual event detected in the video.
    One event typically covers a single delivery window.
    """

    frame_start: float = Field(..., description="Start of detection window (seconds)")
    frame_end: float = Field(..., description="End of detection window (seconds)")
    ball_trajectory: list[BallPosition] = Field(
        default_factory=list,
        description="Ball detections across the delivery window",
    )
    umpire_signal: Optional[UmpireSignalLabel] = Field(
        None,
        description="Umpire arm-signal classification from pose model",
    )
    umpire_pose: Optional[UmpirePoseLabel] = Field(
        None,
        description="Umpire body-pose label from MediaPipe",
    )
    batsman_hit: Optional[bool] = Field(
        None,
        description="Whether the ball made contact with the bat",
    )
    boundary_crossed: Optional[bool] = Field(
        None,
        description="Whether the ball crossed the boundary rope",
    )
    stump_disturbed: Optional[bool] = Field(
        None,
        description="Whether stumps were disturbed (bowled / run-out candidate)",
    )
    raw_detections: Optional[dict] = Field(
        None,
        description="Raw YOLO detection output for debugging",
    )


# ──────────────────────────────────────────────────────────────
# Fusion pipeline output
# ──────────────────────────────────────────────────────────────

OutcomeLabel = Literal[
    "dot_ball", "single", "two", "three", "four", "six",
    "wide", "no_ball",
    "wicket_bowled", "wicket_lbw", "wicket_caught",
    "wicket_runout", "wicket_stumped",
    "dead_ball", "penalty_runs", "unknown",
]


class BallEvent(BaseModel):
    """Structured result for a single delivery — the final pipeline output."""

    ball_number: int = Field(..., ge=1)
    timestamp: str = Field(..., description="MM:SS.s of the delivery")
    outcome: OutcomeLabel
    runs: int = Field(..., ge=0)
    umpire_signal: Optional[UmpireSignalLabel] = None
    transcript_excerpt: Optional[str] = None
    ball_trajectory: list[BallPosition] = Field(default_factory=list)
    reasoning: str = Field(..., description="LLM reasoning for the classification")
    confidence: float = Field(..., ge=0.0, le=1.0)


class MatchLog(BaseModel):
    """Top-level output written to output.json."""

    match_id: str
    clip_path: str
    balls: list[BallEvent]
