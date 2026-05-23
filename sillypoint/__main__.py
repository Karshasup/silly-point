"""
sillypoint/__main__.py
───────────────────────
CLI entry point.  Invoked by:  python -m sillypoint <video>

Pipeline:
  1. audio.transcribe + vision.detect run in parallel (independent)
  2. fusion.fuse combines both into structured BallEvents
  3. render.annotate burns overlays onto a copy of the video
  4. output.json + output_annotated.mp4 written to --output-dir
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import typer
from dotenv import load_dotenv

from sillypoint import audio, vision
from sillypoint.fusion import fuse
from sillypoint.render import annotate
from sillypoint.types import MatchLog

load_dotenv()

app = typer.Typer(add_completion=False)


@app.command()
def main(
    video: Path = typer.Argument(..., help="Path to cricket video clip"),
    output_dir: Path = typer.Option(Path("."), "--output-dir", "-o", help="Directory for output files"),
) -> None:
    """Analyse a cricket video clip → ball-by-ball JSON + annotated video."""
    if not video.exists():
        typer.echo(f"Error: {video} not found", err=True)
        raise typer.Exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    match_id = video.stem

    typer.echo(f"Processing {video.name} ...")

    typer.echo("  [1/4] Running audio transcription + vision detection in parallel ...")
    with ThreadPoolExecutor(max_workers=2) as pool:
        audio_fut  = pool.submit(audio.transcribe, str(video))
        vision_fut = pool.submit(vision.detect, str(video))
        transcript     = audio_fut.result()
        visual_events  = vision_fut.result()

    typer.echo(f"  [2/4] Got {len(transcript)} transcript segments, {len(visual_events)} delivery windows")

    typer.echo("  [3/4] Fusing with LLM ...")
    balls = fuse(transcript, visual_events)
    log   = MatchLog(match_id=match_id, clip_path=str(video), balls=balls)

    json_out = output_dir / "output.json"
    json_out.write_text(log.model_dump_json(indent=2))
    typer.echo(f"  [4/4] Wrote {json_out}  ({len(balls)} ball events)")

    video_out = output_dir / "output_annotated.mp4"
    typer.echo(f"        Rendering annotated video → {video_out} ...")
    annotate(str(video), log, str(video_out))
    typer.echo("Done.")


if __name__ == "__main__":
    app()
