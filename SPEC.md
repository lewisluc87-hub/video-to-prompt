# Video-to-Prompt Generator — Implementation Spec

## 1. Purpose

A CLI tool that ingests a video file and outputs a text prompt (or set of prompts) suitable for feeding into AI video generation models (Sora, Runway Gen-3/4, Veo, Kling, etc.) to recreate or closely approximate the source video's visual content, motion, and style.

Pipeline: local CV extracts objective, deterministic signal from the video (scenes, motion, composition, color, objects). An LLM reasoning layer then synthesizes that structured signal into natural-language prompt(s), optionally per-model-formatted.

## 2. Goals & Non-Goals

**Goals**
- Single command: `video2prompt input.mp4` → prompt text(s) on stdout / file.
- Hybrid pipeline so the tool works (degraded) with zero API calls, and works well with an LLM key configured.
- Multi-shot videos get broken into per-shot prompts, with an optional merged "master" prompt.
- Output formatted for at least one target model's known prompt conventions, with a model-agnostic fallback.
- Reasonably fast on consumer hardware (no GPU required for the local CV stage; LLM stage is API-based).

**Non-Goals (v1)**
- Sound-design / SFX/music prompting (flag as future work). Speech transcription is in scope as a *context-gathering* input (see §6a) — not as literal text reproduced in the output prompt.
- Frame-perfect reconstruction guarantees — this is a descriptive aid, not a lossless encoder.
- Training or fine-tuning any models.
- A hosted web service (open-source CLI/library only for now).

## 3. High-Level Architecture

```
input.mp4
   │
   ▼
[1] Ingest & Probe ─── ffprobe metadata (fps, duration, resolution, codec)
   │
   ▼
[2] Shot/Scene Segmentation ─── PySceneDetect (content-aware cuts)
   │
   ▼
[3] Per-Shot Frame Sampling ─── keyframes: first, middle, last (+ extras for long shots)
   │
   ▼
[3a] Audio Transcription (whole video, parallel to CV) ─── faster-whisper, local/offline
   │      Output: timestamped transcript segments, aligned back to shot time ranges
   │
   ▼
[4] Local CV Analysis (per shot)
   ├─ Object/subject detection      (YOLOv8n or similar, lightweight)
   ├─ Composition/framing heuristics (rule-of-thirds box, shot-size estimate from subject bbox)
   ├─ Camera motion estimation       (optical flow / RAFT-lite or simple sparse flow via OpenCV)
   ├─ Color/lighting analysis        (dominant palette, brightness, contrast)
   └─ Shot duration & pacing
   │
   ▼
[5] Structured Shot Record (JSON) ── one per shot, schema in §5 (includes transcript_segment field)
   │
   ▼
[6] LLM Reasoning Layer (optional but default-on)
   ├─ Input: structured JSON + sampled keyframe thumbnails (base64, low-res) + aligned transcript text
   ├─ Task: use transcript ONLY to infer intent/tone/context; describe subject, action,
   │         style, camera move in natural visual language — never quote or restate dialogue
   └─ Output: prose prompt per shot
   │
   ▼
[7] Prompt Assembly & Formatting
   ├─ Per-shot prompts
   ├─ Master/merged prompt (if requested)
   └─ Model-specific formatting (Sora / Runway / Veo / generic)
   │
   ▼
output: prompts.json + prompts.md (or stdout)
```

If no LLM API key is configured, step 6 is replaced by a template-based sentence generator that fills a fixed prompt template from the structured JSON fields directly (lower quality, zero cost, zero network).

## 4. Tech Stack

- **Language:** Python 3.11+
- **Video I/O / probing:** `ffmpeg-python` + system `ffmpeg`/`ffprobe`
- **Scene detection:** `PySceneDetect`
- **Object detection:** `ultralytics` (YOLOv8n, CPU-friendly) — optional extra; tool runs without it using coarser heuristics
- **Optical flow / motion:** OpenCV (`cv2.calcOpticalFlowFarneback` as default; pluggable for RAFT later)
- **Color analysis:** OpenCV + `colorthief` or k-means on pixel samples
- **Audio transcription:** `faster-whisper` (CTranslate2-based, runs locally on CPU, no API key/network required)
- **LLM client:** Anthropic SDK (`anthropic` pkg), pluggable provider interface so OpenAI/local models can be swapped in
- **CLI framework:** `typer` (or `argparse` to minimize deps)
- **Config:** `pydantic` models + optional `config.yaml`
- **Packaging:** `pyproject.toml`, published as installable via `pip install -e .` and eventually to PyPI
- **Testing:** `pytest`, sample fixture videos (a few seconds, checked into `tests/fixtures/` or pulled via a download script if too large for git)

## 5. Structured Shot Record Schema

```json
{
  "shot_index": 0,
  "start_time": 0.0,
  "end_time": 3.2,
  "duration": 3.2,
  "resolution": "1920x1080",
  "fps": 30,
  "subjects": [
    {"label": "person", "confidence": 0.91, "bbox_norm": [0.3, 0.2, 0.6, 0.9], "track_id": 1}
  ],
  "shot_size_estimate": "medium close-up",
  "composition": "rule-of-thirds, subject left",
  "camera_motion": {
    "type": "pan",
    "direction": "left-to-right",
    "magnitude": "slow"
  },
  "color": {
    "dominant_palette": ["#1a2b3c", "#d8c39a", "#445566"],
    "brightness": "low-key",
    "contrast": "high"
  },
  "pacing_note": "single continuous shot, no internal cuts",
  "transcript_segment": "...so I just kept walking until I saw the gate...",
  "keyframe_paths": ["shot0_f0.jpg", "shot0_f1.jpg", "shot0_f2.jpg"]
}
```

`transcript_segment` is the whisper output text whose timestamps fall within `[start_time, end_time]` for that shot. It may be empty/null for silent or music-only shots. This field is consumed by the LLM stage for context only — formatters in §7 must never copy it verbatim into the final prompt.

This JSON is the contract between the CV stage and the LLM stage — it's also useful standalone for debugging or for users who want to write prompts by hand from structured data.

## 6. Debug / Verification Mode

The single highest-leverage tool for trusting this pipeline's output is being able to *see* what the CV stage thinks it saw, next to the actual frame. `--debug` turns this on.

**What it produces**, per sampled keyframe, written to `<output_dir>/debug/`:
- The original keyframe with overlays drawn on top: subject bounding boxes (labeled with class + confidence), a rule-of-thirds grid with the estimated composition anchor marked, and sparse optical-flow vectors arrowed over the frame to show detected camera/subject motion direction and magnitude.
- A small text annotation burned into the corner of the image (or a sidecar `.json`) showing the shot's classified `shot_size_estimate`, `camera_motion`, and `dominant_palette` swatch — so the overlay is self-contained and shareable without needing to cross-reference the main JSON output.
- An `index.md` in the debug folder that lays out all shots in order with their overlay thumbnails inline, so a full video's CV output can be eyeballed in one scroll rather than opening files one by one.

**Cost:** this stage only draws on data already computed in §4 (CV analysis) — it adds negligible runtime (simple OpenCV drawing calls) and no extra model inference. `--debug` implies `--keep-frames` since there's nothing to overlay onto otherwise.

**Intended use:** run with `--debug` whenever testing on a new video or after changing a CV module, scan `debug/index.md`, and confirm shot boundaries, detected subjects, and motion direction look right *before* trusting the LLM-generated prompt for that shot. If the CV layer is feeding the LLM bad signal, this is where it'll be visible first — much cheaper to catch here than by noticing the final prompt or generated video is off.

**Known limitation, fixed during testing**: early versions of the motion estimator reported confident pan/tilt/zoom classifications on content with no real camera motion — UI scrolling, animated burned-in captions, and noisy low-texture footage (e.g. star fields) all produced scattered, high-magnitude flow vectors that got misread as deliberate camera movement. The fix added a directional-coherence check (`motion.py`): real camera motion produces flow vectors that broadly agree in direction, while scroll/caption/noise motion doesn't, so incoherent flow now reports as `"unknown"` rather than a confident wrong answer. The object-detection confidence threshold was also raised from 0.35 to 0.5 after observing false-positive detections (e.g. "cell phone," "train") on flat UI screens and abstract art. `--debug` is what surfaced both issues — exactly its intended purpose.



## 7. Audio Transcription (Context Stage)

Runs once on the full audio track, in parallel with the CV stage (not per-shot, since speech doesn't respect shot boundaries).

- **Tool:** `faster-whisper`, default model size `small` or `base` (configurable; `--whisper-model` flag) for a good speed/accuracy tradeoff on consumer CPUs. No API key, no network call — keeps the "fully offline" mode genuinely offline.
- **Output:** timestamped segments (start, end, text), which are then sliced and attached to whichever shot(s) they overlap (`transcript_segment` field in §5's schema). A segment spanning a cut gets attached to both shots it touches.
- **Explicit non-use:** the transcript is never inserted into the final prompt text or formatter output. Its only consumer is the LLM reasoning layer's system prompt, which is instructed to use it strictly for inferring intent, tone, relationships between subjects, and setting/context clues — not to quote or describe it as dialogue, since video-gen models don't take spoken text as a meaningful signal.
- **Graceful degradation:** if the video has no audio track, or `--no-transcribe` is passed, this stage is skipped and `transcript_segment` stays null for all shots — the rest of the pipeline behaves exactly as before this change.

## 8. LLM Reasoning Layer

- Input per shot: the JSON record + 1–3 low-res keyframe thumbnails (sent as image content blocks).
- System prompt instructs the model to: describe subject and action concretely, name shot size and camera movement using standard cinematography vocabulary, describe lighting/color/mood, and avoid hallucinating details not visible/inferable (e.g., don't invent dialogue or off-screen elements).
- Output: a single dense paragraph (40–80 words), optimized for video-gen model conventions (subject → action → setting → camera → style/lighting).
- Batch shots in parallel (async) with a concurrency cap; merge results in original order.
- If `--merge` is passed, do a second LLM call: feed all per-shot prompts plus overall video metadata, ask for one cohesive master prompt (for single-shot regeneration tools that only accept one prompt).

## 9. Prompt Formatting / Model Targets


Implement formatter plugins, each taking the generic prose prompt + structured data and emitting model-flavored output:

- **Generic** — plain descriptive paragraph (default, works everywhere).
- **Sora-style** — emphasizes narrative/scene description, supports the multi-sentence cinematic format Sora favors.
- **Runway-style** — shorter, comma-separated descriptor format (subject, action, camera move, style) matching Gen-3/4 conventions.
- **Veo-style** — similar to generic but with explicit camera-direction syntax Google's docs recommend.

Formatters live in `formatters/` as small, independently testable functions so new targets can be added without touching the core pipeline. Document each formatter's assumptions clearly since these conventions shift as vendors update their docs — note in the README that formatting guidance should be periodically rechecked against current vendor prompt guides.

## 9a. Breakdown Mode (general scene-by-scene breakdown, any video)

`--mode breakdown` is a second, parallel output mode alongside the default `video_prompt` mode described above. It produces a structured scene-by-scene breakdown of a video — narrative role per scene, a plain-prose description of what happens, the on-screen text quoted in order, and a one-sentence summary of the whole video. It is general-purpose: it works on screen-recorded software demos, tutorials, vlogs, narrative camera footage, presentations, or anything else, not just marketing content. The role labels and descriptions are inferred per-video rather than forced into a fixed marketing-funnel template (Hook/CTA/etc. show up naturally when the video actually has that structure, but the LLM isn't instructed to assume it).

**Why a separate mode, not a tuned version of `video_prompt`:** `video_prompt` mode is built around recreating a shot visually (camera motion, shot size, composition) for a video-generation model. Breakdown mode is built around understanding and re-describing what a video communicates — its structure, on-screen content, and purpose — which is a fundamentally different task with a different output shape, regardless of the source video's genre. Testing this against screen-recorded UI content specifically (see §6's debug-mode findings) was what surfaced the need for it, but the mode itself isn't limited to that content type.

**Pipeline differences in breakdown mode:**
- An OCR stage (`ocr.py`, via `pytesseract`, extra: `video2prompt[ocr]`) extracts on-screen/burned-in text per shot, for any video that has it — captions, UI labels, lower-thirds, slide text, subtitles burned into frame, etc. Unlike the audio transcript (context only, never quoted), OCR text is *primary content* in this mode — it's quoted directly in the output, since it's literally part of what the video shows.
- The LLM reasoning layer uses a different system prompt (`BREAKDOWN_SYSTEM_PROMPT` in `llm/base.py`) that asks for a `scene_role` label inferred from context (not a fixed enum — could be "Hook," "Establishing Shot," "Demonstration," "Conclusion," or anything else that fits), a `scene_description` of what happens (UI interaction for screen content, action/setting for camera footage), and the ordered `on_screen_text` list, returned as JSON per shot.
- A final `generate_core_prompt` call produces a one-sentence summary of the whole video's subject and purpose/outcome.
- Camera-motion/shot-size/composition fields are still computed (the CV stage doesn't know the mode ahead of time) and can still inform `scene_description` for camera-shot footage; they're just not separately surfaced as a "camera move" field the way `video_prompt` mode does.

**Output format**: see `cli.py`'s `_render_breakdown` — numbered scenes with timestamp ranges and role labels, a one/two-sentence description, an arrow-separated quote of on-screen text in order, and a closing one-sentence core prompt.

**Offline/template fallback**: `TemplateProvider.generate_scene_breakdown` provides a best-effort breakdown using OCR text, detected subjects, and shot-index heuristics (no LLM reasoning) when `--no-llm` is set — lower quality, but keeps the fully-offline guarantee intact.

## 10. CLI Design

```
video2prompt <input> [options]

Options:
  --output, -o PATH        Output file (default: stdout)
  --format [json|md|txt]   Output format (default: md)
  --mode [video_prompt|breakdown]  video_prompt: dense visual-description prompts
                            for Sora/Runway/Veo (default). breakdown: scene-by-scene
                            breakdown of a screen-recorded demo/marketing video,
                            with on-screen text and a one-sentence core summary.
  --target [generic|sora|runway|veo]   Prompt style, video_prompt mode only (default: generic)
  --merge                  Also produce one merged master prompt (video_prompt mode only)
  --no-llm                 Force template-only mode (no API calls)
  --no-transcribe           Skip audio transcription entirely
  --no-ocr                  Skip on-screen text (OCR) extraction
  --whisper-model SIZE      tiny|base|small|medium (default: base)
  --provider [anthropic|openai]  LLM provider (default: anthropic)
  --max-shots N             Cap number of shots processed (cost/time control)
  --keep-frames             Don't delete extracted keyframe images after run
  --debug                   Generate annotated keyframe overlays (bboxes, flow vectors,
                             shot-size/composition guides) for visual verification of the
                             CV stage; implies --keep-frames; written to <output_dir>/debug/
  --verbose                 Debug logging
```

Example:
```
video2prompt clip.mp4 --target runway --merge -o prompts.md
```

## 11. Repo Structure

```
video-to-prompt/
├── pyproject.toml
├── README.md
├── LICENSE (MIT or Apache-2.0)
├── src/video2prompt/
│   ├── __init__.py
│   ├── cli.py
│   ├── ingest.py          # ffprobe wrapper, validation
│   ├── segment.py          # PySceneDetect wrapper
│   ├── sample.py            # keyframe extraction
│   ├── audio.py              # faster-whisper transcription + shot alignment
│   ├── ocr.py                 # pytesseract on-screen text extraction (breakdown mode)
│   ├── cv_analysis/
│   │   ├── objects.py
│   │   ├── motion.py
│   │   ├── color.py
│   │   ├── composition.py
│   │   └── debug_overlay.py  # draws bboxes/flow vectors/composition guides for --debug
│   ├── schema.py            # pydantic models (ShotRecord etc.)
│   ├── llm/
│   │   ├── base.py          # provider interface
│   │   ├── anthropic_provider.py
│   │   └── template_fallback.py
│   ├── formatters/
│   │   ├── generic.py
│   │   ├── sora.py
│   │   ├── runway.py
│   │   └── veo.py
│   └── assemble.py          # orchestrates the pipeline end-to-end
├── tests/
│   ├── fixtures/
│   └── test_*.py
└── examples/
    └── sample_output.md
```

## 12. Configuration & Secrets

- API key read from `ANTHROPIC_API_KEY` env var (never hardcoded, never logged).
- Optional `config.yaml` for defaults (target format, max shots, concurrency limit).
- `.env.example` checked in; real `.env` gitignored.
- README documents that running with `--no-llm` requires no key and no network access at all — this includes the transcription stage, which runs fully locally via `faster-whisper` regardless of `--no-llm`.

## 13. Performance & Cost Considerations

- Default keyframe count per shot kept low (3) to bound both compute and LLM image-token cost.
- `--max-shots` flag protects against runaway cost on long videos; tool warns if estimated shot count exceeds a threshold (e.g., >40) before proceeding, with a confirmation prompt.
- Local CV stage designed to run on CPU in real time or faster for typical short clips (<2 min); YOLOv8n chosen specifically for this.
- Async/concurrent LLM calls with a configurable concurrency cap (default 4) to avoid rate-limit errors.

## 14. Testing Strategy

- Unit tests per module (segment boundaries on a known fixture, motion classification on a synthetic pan, color palette extraction, transcript-to-shot time alignment on a fixture with known speech timestamps).
- Golden-file tests for template fallback mode (deterministic, no network) to catch regressions in generated prose.
- LLM-stage tests mocked (no real API calls in CI) plus an optional manual/integration test script gated behind an env flag for local runs with a real key.
- A couple of short, rights-cleared sample clips (or synthetically generated test video, e.g. colored shapes moving) bundled as fixtures so the suite runs without external downloads.

## 15. Open-Source Readiness Checklist

- LICENSE file (MIT recommended for max reuse).
- README with install instructions, quickstart example, architecture diagram (can reuse §3 ASCII diagram), and a clear note on API costs/keys required for full functionality.
- CONTRIBUTING.md covering how to add a new formatter or CV module.
- GitHub Actions CI: lint (ruff), type-check (mypy optional), pytest on push/PR.
- Issue templates for bug reports and "add a new model formatter" feature requests.
- Versioned releases via tags; changelog.

## 16. Roadmap (post-v1)

- Sound-design/SFX/music cue detection and prompting (separate from speech transcription, which is now in v1).
- RAFT-based optical flow for higher-fidelity camera motion classification.
- Style-transfer/visual-style fingerprinting (e.g., detect "anime," "film noir," "VHS") via a lightweight classifier or LLM-vision pass.
- Web UI wrapper around the same core library.
- Direct API integrations to submit the generated prompt straight to a video-gen provider's API, where available.
