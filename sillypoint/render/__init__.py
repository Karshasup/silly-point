"""
sillypoint/render/__init__.py
──────────────────────────────
Burns ball-event overlays onto the video using OpenCV.

Each BallEvent is shown as a text overlay for OVERLAY_DURATION seconds
starting from the ball's timestamp. White text with a black shadow for
legibility on any background.

Public API
----------
    from sillypoint.render import annotate
    annotate("clip.mp4", match_log, "output_annotated.mp4")
"""

from __future__ import annotations

import cv2

from sillypoint.types import BallEvent, MatchLog

OVERLAY_DURATION = 3.0  # seconds to display each ball-event overlay


def _ts_to_sec(ts: str) -> float:
    """Convert 'MM:SS.s' → float seconds."""
    m, s = ts.split(":")
    return int(m) * 60 + float(s)


def _draw_overlay(frame, event: BallEvent) -> None:
    """Draw a 2-3 line event summary in the bottom-left of the frame."""
    h = frame.shape[0]
    lines = [
        f"Ball {event.ball_number}  {event.outcome.upper()}  +{event.runs}r  ({event.confidence:.0%})",
    ]
    if event.umpire_signal:
        lines.append(f"Signal: {event.umpire_signal}")
    if event.transcript_excerpt:
        lines.append(f'"{event.transcript_excerpt}"')

    y = h - (len(lines) * 28) - 16
    for line in lines:
        # Shadow then white text
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        y += 28


def annotate(video_path: str, log: MatchLog, output_path: str) -> None:
    """
    Write an annotated copy of video_path to output_path.

    Each delivery's overlay is shown for OVERLAY_DURATION seconds starting
    at the ball's timestamp. Events are looked up by nearest timestamp —
    the most recent event that started within OVERLAY_DURATION seconds is used.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Pre-compute (start_sec, event) sorted ascending so we can scan cheaply
    timed_events = sorted(
        (((_ts_to_sec(b.timestamp)), b) for b in log.balls),
        key=lambda x: x[0],
    )

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_idx / fps

        # Most recent event whose window [t_start, t_start + OVERLAY_DURATION] covers t
        active: BallEvent | None = None
        for t_start, event in reversed(timed_events):
            if t_start <= t <= t_start + OVERLAY_DURATION:
                active = event
                break

        if active:
            _draw_overlay(frame, active)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
