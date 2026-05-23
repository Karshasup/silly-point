"""
sillypoint/vision/__init__.py
──────────────────────────────
Frame sampling + object detection (YOLOv8) + umpire pose classification (MediaPipe).

Samples the video at SAMPLE_FPS, runs YOLO per frame for ball detection,
runs MediaPipe Pose per frame for umpire signal classification, then
groups detections into per-delivery windows.

A new delivery window begins when ball detection has been absent for
DELIVERY_GAP seconds. Each window is extended by SIGNAL_WINDOW seconds
after the last ball detection to capture the umpire's post-delivery signal.

Public API
----------
    from sillypoint.vision import detect
    events = detect("clip.mp4")
"""

from __future__ import annotations

import os

import cv2
import mediapipe as mp
from ultralytics import YOLO

from sillypoint.types import BallPosition, UmpirePoseLabel, UmpireSignalLabel, VisualEvent

# Seconds without ball detection that marks a boundary between deliveries.
DELIVERY_GAP: float = float(os.getenv("DELIVERY_GAP", "2.0"))

# Seconds after last ball detection included in the window (for umpire signal).
SIGNAL_WINDOW: float = float(os.getenv("SIGNAL_WINDOW", "3.0"))

# Frame rate to sample at (lower = faster, still sufficient for cricket).
SAMPLE_FPS: float = float(os.getenv("SAMPLE_FPS", "10.0"))

# COCO class index for "sports ball".
_BALL_CLASS = 32

_yolo: YOLO | None = None


def _get_yolo() -> YOLO:
    global _yolo
    if _yolo is None:
        _yolo = YOLO(os.getenv("YOLO_MODEL", "yolov8n.pt"))
    return _yolo


# ──────────────────────────────────────────────────────────────
# Pose classification
# ──────────────────────────────────────────────────────────────

_POSE_TO_SIGNAL: dict[UmpirePoseLabel, UmpireSignalLabel | None] = {
    "both_arms_raised": "six",
    "arms_crossed":     "out",
    "arm_extended":     "wide",
    "arm_swept_low":    "four",
    "index_raised":     "no_ball",
    "neutral":          None,
    "unknown":          None,
}


def _classify_pose(landmarks: list) -> UmpirePoseLabel:
    """
    Heuristic pose classification from MediaPipe landmark list.
    Threshold values are tuned for a standing umpire facing the camera.
    """
    L_SHO, R_SHO = 11, 12
    L_ELB, R_ELB = 13, 14
    L_WRI, R_WRI = 15, 16
    L_HIP, R_HIP = 23, 24

    lm = landmarks

    l_wrist_y = lm[L_WRI].y
    r_wrist_y = lm[R_WRI].y
    l_sho_y   = lm[L_SHO].y
    r_sho_y   = lm[R_SHO].y

    # Both wrists above respective shoulders → six
    if l_wrist_y < l_sho_y and r_wrist_y < r_sho_y:
        return "both_arms_raised"

    center_x  = (lm[L_HIP].x + lm[R_HIP].x) / 2
    sho_y_avg = (l_sho_y + r_sho_y) / 2
    hip_y_avg = (lm[L_HIP].y + lm[R_HIP].y) / 2

    # Both wrists near body centre → out (arms crossed)
    if abs(lm[L_WRI].x - center_x) < 0.15 and abs(lm[R_WRI].x - center_x) < 0.15:
        return "arms_crossed"

    # Either wrist extended horizontally at shoulder height → wide / no-ball
    l_ext = abs(l_wrist_y - sho_y_avg) < 0.15 and lm[L_WRI].x < lm[L_SHO].x - 0.2
    r_ext = abs(r_wrist_y - sho_y_avg) < 0.15 and lm[R_WRI].x > lm[R_SHO].x + 0.2
    if l_ext or r_ext:
        return "arm_extended"

    # Either wrist below hip → four (sweeping arm low)
    if l_wrist_y > hip_y_avg or r_wrist_y > hip_y_avg:
        return "arm_swept_low"

    return "neutral"


# ──────────────────────────────────────────────────────────────
# Delivery window segmentation
# ──────────────────────────────────────────────────────────────

# Internal frame record: (timestamp, ball_or_none, pose_label)
_Frame = tuple[float, BallPosition | None, UmpirePoseLabel]


def _to_events(frames: list[_Frame]) -> list[VisualEvent]:
    """
    Cluster frame detections into per-delivery VisualEvents.

    Strategy:
      1. Find indices of frames that contain a ball detection.
      2. Split those indices into clusters where the gap between consecutive
         detections exceeds DELIVERY_GAP.
      3. For each cluster, extend the time window by SIGNAL_WINDOW to
         capture the umpire's post-delivery signal.
      4. Collect ball trajectory and dominant pose from that extended window.
    """
    ball_frames = [(i, t) for i, (t, ball, _) in enumerate(frames) if ball is not None]

    if not ball_frames:
        return []

    # Split ball_frames into per-delivery clusters
    clusters: list[list[int]] = []
    current: list[int] = [ball_frames[0][0]]

    for (_, t_prev), (i_curr, t_curr) in zip(ball_frames, ball_frames[1:]):
        if t_curr - t_prev > DELIVERY_GAP:
            clusters.append(current)
            current = [i_curr]
        else:
            current.append(i_curr)
    clusters.append(current)

    events: list[VisualEvent] = []
    for cluster in clusters:
        t_start = frames[cluster[0]][0]
        t_ball_end = frames[cluster[-1]][0]
        t_window_end = t_ball_end + SIGNAL_WINDOW

        window_frames = [f for f in frames if t_start <= f[0] <= t_window_end]

        balls  = [b for _, b, _ in window_frames if b is not None]
        poses  = [p for _, _, p in window_frames]
        dominant = max(set(poses), key=poses.count)

        events.append(VisualEvent(
            frame_start=t_start,
            frame_end=t_window_end,
            ball_trajectory=balls,
            umpire_pose=dominant,
            umpire_signal=_POSE_TO_SIGNAL.get(dominant),
        ))

    return events


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def detect(video_path: str) -> list[VisualEvent]:
    """
    Sample frames from a video, run YOLO + MediaPipe, return delivery windows.

    Each VisualEvent covers one delivery: ball trajectory + umpire pose signal.
    """
    yolo = _get_yolo()
    cap  = cv2.VideoCapture(video_path)

    native_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step  = max(1, int(native_fps / SAMPLE_FPS))

    frames: list[_Frame] = []

    with mp.solutions.pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose_model:

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step == 0:
                t = frame_idx / native_fps
                h, w = frame.shape[:2]

                # Ball detection
                ball: BallPosition | None = None
                for box in yolo(frame, verbose=False)[0].boxes:
                    if int(box.cls[0]) == _BALL_CLASS:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        ball = BallPosition(
                            t=t,
                            x=(x1 + x2) / 2 / w,
                            y=(y1 + y2) / 2 / h,
                            confidence=float(box.conf[0]),
                        )
                        break  # highest-confidence ball only

                # Pose classification
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose_model.process(rgb)
                pose   = (
                    _classify_pose(result.pose_landmarks.landmark)
                    if result.pose_landmarks
                    else "unknown"
                )

                frames.append((t, ball, pose))

            frame_idx += 1

    cap.release()
    return _to_events(frames)
