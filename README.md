# silly-point

> Multimodal AI cricket umpiring. Drop in a video clip, get back a transcribed, ball-tracked, ball-by-ball breakdown with reasoning.

<!-- HERO -->

Most cricket CV work stops at "detect the ball" or "classify the umpire's pose." silly-point fuses both signals — plus what the umpire actually *says* — into structured match events.

## How it works

```
video clip ──┬─► whisper ──────► timestamped umpire speech
             │
             ├─► frame sampler ─► YOLO (ball, bat, stumps, batsman)
             │                  └► umpire pose classification
             │
             └─► LLM fusion ──► structured ball-by-ball JSON + annotated video
```

The fusion layer is the interesting part: timestamped transcript + visual events + pose labels go to an LLM that produces a ball-by-ball log with reasoning per event.

## Quick start

```bash
git clone https://github.com/Karshasup/silly-point
cd silly-point
pip install -r requirements.txt
cp .env.example .env  # add your API keys
python -m sillypoint examples/sample_over.mp4
```

Outputs `output.json` and `output_annotated.mp4`.

## Output format

```json
{
  "match_id": "sample_over",
  "balls": [
    {
      "ball_number": 1,
      "timestamp": "00:02.4",
      "outcome": "no_ball",
      "runs": 1,
      "umpire_signal": "no_ball",
      "transcript_excerpt": "no ball, one run",
      "ball_trajectory": [...],
      "reasoning": "Umpire's outstretched arm + verbal 'no ball' call matches the front-foot fault visible at 00:02.1",
      "confidence": 0.87
    }
  ]
}
```

## Architecture

| Module | Does | Stack |
|---|---|---|
| `audio/` | Speech-to-text on full clip with word-level timestamps | faster-whisper |
| `vision/` | Frame sampling + object detection + pose classification | ultralytics YOLOv8, mediapipe |
| `fusion/` | Combines audio + vision into structured events | Anthropic / Gemini API |
| `render/` | Burns JSON onto video as overlay | ffmpeg |

## Roadmap

- [x] v0.1 — single-over clip → ball-by-ball JSON
- [ ] v0.2 — full match support, scorecard generation
- [ ] v0.3 — LBW trajectory projection
- [ ] v0.4 — live stream input
- [ ] v0.5 — fine-tuned pose model on the SNOW dataset

## Why "silly-point"

Silly point is a fielding position absurdly close to the batter — close enough to see everything happen. Felt right.

## Contributing

Issues and PRs welcome. The audio pipeline is the most novel piece and the most under-tested — sample clips with clean umpire audio are particularly useful.

## License

MIT
