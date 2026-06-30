# video2prompt

Turn a video into text prompts for AI video generation models (Sora, Runway, Veo).

`video2prompt` analyzes a source video locally — shot detection, object/subject
detection, camera motion, color/lighting, on-screen text (OCR), and optional
speech transcription — then uses an LLM to turn that structured analysis into
one of two output types, depending on what kind of video you're working with.

**Two modes:**

- **`video_prompt` (default)** — dense visual-description prompts per shot,
  meant to be fed into a video-generation model (Sora, Runway, Veo) to
  recreate or approximate a camera-shot video.
- **`breakdown`** — a general-purpose scene-by-scene breakdown of *any*
  video — software demos, tutorials, vlogs, narrative camera footage,
  presentations, screen recordings, anything. Each scene gets a narrative
  role label inferred from context (not a fixed template), a description of
  what happens, any on-screen text quoted in order, and the whole video gets
  a one-sentence "core" summary. Use this mode when you want to understand
  and re-describe a video's structure and content rather than recreate it
  visually.

It also works in two cost/network modes orthogonal to the above:

- **Hybrid (default):** local CV + an LLM (Claude) reasoning over the structured
  data and sampled frames to write higher-quality, more natural prompts.
- **Fully offline (`--no-llm`):** local CV only, with a deterministic template
  filling in the output. No API key, no network calls, ever.

See [`SPEC.md`](./SPEC.md) for the full implementation spec and design rationale.

## Install

```bash
git clone https://github.com/lewisluc87-hub/video-to-prompt.git
cd video-to-prompt
pip install -e ".[all]"      # everything: LLM, object detection, transcription
# or pick extras individually:
pip install -e ".[llm]"       # Claude reasoning layer
pip install -e ".[objects]"   # YOLOv8n subject detection (better than the built-in fallback)
pip install -e ".[transcribe]" # faster-whisper speech-to-text context
pip install -e ".[ocr]"        # pytesseract on-screen text extraction (breakdown mode)
```

`ffmpeg`/`ffprobe` must also be installed and on your `PATH` (e.g. `apt install
ffmpeg` or `brew install ffmpeg`). The `ocr` extra also requires the
`tesseract` binary on your `PATH` (e.g. `apt install tesseract-ocr` or
`brew install tesseract`).

For the LLM reasoning layer, set your API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Quickstart

```bash
video2prompt clip.mp4
```

Writes a markdown file (or prints to stdout) with one prompt per detected shot.

```bash
# Target Runway's prompt conventions, merge into one master prompt, save as JSON
video2prompt clip.mp4 --target runway --merge --format json -o prompts.json

# Fully offline, no API calls, no network
video2prompt clip.mp4 --no-llm

# Verify what the CV stage actually detected before trusting the prompt
video2prompt clip.mp4 --debug

# Scene-by-scene breakdown of any video (demo, tutorial, vlog, etc.)
video2prompt my_video.mp4 --mode breakdown
```

`--debug` writes annotated keyframes (bounding boxes, optical-flow vectors,
composition grid, detected palette/lighting/motion caption) plus an
`index.md` you can scroll through, so you can sanity-check the pipeline's
understanding of the video before relying on the generated prompt. See
`SPEC.md` §6 for details.

## CLI options

```
video2prompt <input> [options]

  --output, -o PATH         Output file (default: stdout)
  --format [json|md|txt]    Output format (default: md)
  --mode [video_prompt|breakdown]  video_prompt: dense per-shot prompts for
                             Sora/Runway/Veo (default). breakdown: scene-by-scene
                             breakdown of a screen-recorded demo/marketing video.
  --target [generic|sora|runway|veo]  Prompt style, video_prompt mode only (default: generic)
  --merge                   Also produce one merged master prompt (video_prompt mode only)
  --no-llm                  Force template-only mode (no API calls)
  --no-transcribe           Skip audio transcription
  --no-ocr                   Skip on-screen text (OCR) extraction
  --whisper-model SIZE      tiny|base|small|medium (default: base)
  --provider [anthropic]    LLM provider (default: anthropic)
  --max-shots N              Cap number of shots processed (cost/time control)
  --keep-frames              Don't delete extracted keyframe images after run
  --debug                    Generate annotated keyframe overlays for verification
  --verbose                  Verbose logging
```

## How it works

```
input.mp4 → scene detection → keyframe sampling → local CV analysis
          → (optional) speech transcription, aligned to shots
          → structured per-shot JSON
          → LLM reasoning layer (or template fallback)
          → per-target formatter (generic / sora / runway / veo)
          → prompts.md / prompts.json
```

Full architecture, data schema, and design decisions are in [`SPEC.md`](./SPEC.md).

## Notes on accuracy

This is a descriptive aid, not a lossless encoder — it won't perfectly
reconstruct a video, and it isn't meant to. Use `--debug` to check the CV
layer's output on a new video before trusting the prompts it produces, and
periodically round-trip a generated prompt through the actual target model
to compare against the source clip. See `SPEC.md` for more on verification.

Object detection quality depends on whether `ultralytics` (YOLOv8n) is
installed; without it, the tool falls back to a much cruder contour-based
"largest moving blob" heuristic so the pipeline still runs end-to-end.

Vendor prompt conventions (Sora/Runway/Veo) change over time — the
formatters in `src/video2prompt/formatters/` encode a snapshot of current
guidance and should be rechecked against each vendor's docs periodically.

## License

MIT — see [`LICENSE`](./LICENSE).
