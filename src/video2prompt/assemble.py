"""End-to-end orchestration of the pipeline described in SPEC.md section 3."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from . import audio as audio_stage
from . import ingest, ocr, sample, segment
from .cv_analysis import color, composition, debug_overlay, motion, objects
from .formatters import format_shot
from .llm import get_provider
from .schema import ShotRecord, VideoAnalysis


def run_pipeline(
    input_path: str,
    *,
    mode: str = "video_prompt",  # "video_prompt" | "breakdown"
    use_llm: bool = True,
    provider_name: str = "anthropic",
    transcribe: bool = True,
    whisper_model: str = "base",
    ocr_text: bool = True,
    target_format: str = "generic",
    merge: bool = False,
    max_shots: int | None = None,
    debug: bool = False,
    keep_frames: bool = False,
    work_dir: str | None = None,
) -> tuple[VideoAnalysis, Path]:
    video = ingest.probe(input_path)

    base_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="video2prompt_"))
    frames_dir = base_dir / "frames"
    debug_dir = base_dir / "debug"

    shots = segment.detect_shots(video)
    if max_shots:
        shots = shots[:max_shots]

    transcript_segments = []
    if transcribe:
        transcript_segments = audio_stage.transcribe(video, model_size=whisper_model)

    shot_records: list[ShotRecord] = []
    debug_overlays_by_shot: dict[int, list[Path]] = {}

    for shot in shots:
        keyframes = sample.sample_keyframes(video, shot, frames_dir)
        if not keyframes:
            continue

        subjects = objects.detect_subjects(keyframes[len(keyframes) // 2])
        shot_size = composition.estimate_shot_size(subjects)
        comp = composition.estimate_composition(subjects)
        color_info = color.analyze_color(keyframes[len(keyframes) // 2])

        cam_motion = None
        flow = None
        if len(keyframes) >= 2:
            cam_motion, flow = motion.estimate_motion(keyframes[0], keyframes[-1])
        else:
            from .schema import CameraMotion

            cam_motion = CameraMotion()

        transcript_text = audio_stage.attach_to_shot(
            transcript_segments, shot.start_time, shot.end_time
        )

        on_screen_text = None
        if ocr_text:
            extracted = ocr.extract_text_for_shot(keyframes)
            on_screen_text = extracted or None

        record = ShotRecord(
            shot_index=shot.index,
            start_time=shot.start_time,
            end_time=shot.end_time,
            duration=shot.duration,
            resolution=video.resolution,
            fps=video.fps,
            subjects=subjects,
            shot_size_estimate=shot_size,
            composition=comp,
            camera_motion=cam_motion,
            color=color_info,
            pacing_note=(
                "single continuous shot, no internal cuts" if shot.duration > 0 else ""
            ),
            transcript_segment=transcript_text,
            on_screen_text=on_screen_text,
            keyframe_paths=[str(p) for p in keyframes],
        )

        if debug:
            overlays = []
            for kf in keyframes:
                out_path = debug_dir / f"shot{shot.index}_{kf.stem}_overlay.jpg"
                debug_overlay.render_debug_overlay(
                    kf, out_path, subjects, cam_motion, color_info, shot_size,
                    flow=flow if kf == keyframes[-1] else None,
                )
                overlays.append(out_path)
            record.debug_overlay_paths = [str(p) for p in overlays]
            debug_overlays_by_shot[shot.index] = overlays

        shot_records.append(record)

    if debug and debug_overlays_by_shot:
        debug_overlay.write_debug_index(debug_dir, debug_overlays_by_shot)

    llm_provider = get_provider(use_llm, provider_name)

    if mode == "breakdown":
        for record in shot_records:
            kf_paths = [Path(p) for p in record.keyframe_paths]
            breakdown = llm_provider.generate_scene_breakdown(record, kf_paths)
            record.scene_role = breakdown.get("scene_role")
            record.scene_description = breakdown.get("scene_description")
            ocr_lines = breakdown.get("on_screen_text") or []
            record.on_screen_text = "\n".join(ocr_lines) if ocr_lines else record.on_screen_text
    else:
        for record in shot_records:
            kf_paths = [Path(p) for p in record.keyframe_paths]
            record.generated_prompt = llm_provider.generate_shot_prompt(record, kf_paths)

    analysis = VideoAnalysis(
        source_path=str(video.path),
        duration=video.duration,
        fps=video.fps,
        resolution=video.resolution,
        shots=shot_records,
    )

    if mode == "breakdown" and shot_records:
        scenes = [
            {
                "scene_role": r.scene_role,
                "scene_description": r.scene_description,
                "on_screen_text": (r.on_screen_text or "").splitlines(),
            }
            for r in shot_records
        ]
        analysis.core_prompt = llm_provider.generate_core_prompt(
            scenes, {"duration": video.duration, "resolution": video.resolution}
        )
    elif merge and shot_records:
        prompts = [r.generated_prompt or "" for r in shot_records]
        analysis.master_prompt = llm_provider.generate_master_prompt(
            prompts, {"duration": video.duration, "resolution": video.resolution}
        )

    if not keep_frames and not debug:
        shutil.rmtree(frames_dir, ignore_errors=True)

    return analysis, base_dir


def format_analysis(analysis: VideoAnalysis, target: str, mode: str = "video_prompt") -> dict:
    if mode == "breakdown":
        formatted = {
            "source": analysis.source_path,
            "duration": analysis.duration,
            "scenes": [
                {
                    "shot_index": shot.shot_index,
                    "start_time": shot.start_time,
                    "end_time": shot.end_time,
                    "scene_role": shot.scene_role,
                    "scene_description": shot.scene_description,
                    "on_screen_text": (shot.on_screen_text or "").splitlines(),
                }
                for shot in analysis.shots
            ],
        }
        if analysis.core_prompt:
            formatted["core_prompt"] = analysis.core_prompt
        return formatted

    formatted = {
        "source": analysis.source_path,
        "duration": analysis.duration,
        "shots": [
            {
                "shot_index": shot.shot_index,
                "start_time": shot.start_time,
                "end_time": shot.end_time,
                "prompt": format_shot(shot, target),
            }
            for shot in analysis.shots
        ],
    }
    if analysis.master_prompt:
        formatted["master_prompt"] = analysis.master_prompt
    return formatted
