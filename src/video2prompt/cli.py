from __future__ import annotations

import json
import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

# Optional personal-use override -- .env.personal (gitignored) can set
# VAP_AUTH_MODE=subscription + CLAUDE_CODE_OAUTH_TOKEN for personal runs
# without touching the shared production .env. override=True lets it win
# over any ANTHROPIC_API_KEY already loaded above, and also strips a
# stray ANTHROPIC_API_KEY explicitly in case .env.personal doesn't set one
# at all (mirrors the belt-and-suspenders approach in VAP's personal_run.py).
if Path(".env.personal").exists():
    load_dotenv(".env.personal", override=True)
    if os.environ.get("VAP_AUTH_MODE") == "subscription":
        os.environ.pop("ANTHROPIC_API_KEY", None)

from .assemble import format_analysis, run_pipeline

app = typer.Typer(add_completion=False, help="Turn a video into prompts for AI video generation models.")
console = Console()


@app.command()
def main(
    input: str = typer.Argument(..., help="Path to the input video file."),
    output: str | None = typer.Option(None, "--output", "-o", help="Output file path (default: stdout)."),
    format: str = typer.Option("md", "--format", help="Output format: json | md | txt"),
    mode: str = typer.Option(
        "video_prompt", "--mode",
        help="video_prompt: dense visual-description prompts for Sora/Runway/Veo. "
             "breakdown: scene-by-scene breakdown of any video (demos, tutorials, "
             "vlogs, narrative footage, screen recordings, etc.), with on-screen "
             "text and a one-sentence core summary.",
    ),
    target: str = typer.Option("generic", "--target", help="Prompt style (video_prompt mode only): generic | sora | runway | veo"),
    merge: bool = typer.Option(False, "--merge", help="Also produce one merged master prompt (video_prompt mode only)."),
    no_llm: bool = typer.Option(False, "--no-llm", help="Force template-only mode (no API calls)."),
    no_transcribe: bool = typer.Option(False, "--no-transcribe", help="Skip audio transcription."),
    no_ocr: bool = typer.Option(False, "--no-ocr", help="Skip on-screen text (OCR) extraction."),
    whisper_model: str = typer.Option("base", "--whisper-model", help="tiny|base|small|medium"),
    provider: str = typer.Option("anthropic", "--provider", help="LLM provider: anthropic"),
    max_shots: int | None = typer.Option(None, "--max-shots", help="Cap number of shots processed."),
    keep_frames: bool = typer.Option(False, "--keep-frames", help="Don't delete extracted keyframes."),
    debug: bool = typer.Option(False, "--debug", help="Generate annotated keyframe overlays for verification."),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose logging."),
):
    """Analyze INPUT and generate video-gen prompts or a scene breakdown."""
    if debug:
        keep_frames = True

    with console.status("[bold green]Analyzing video..."):
        try:
            analysis, work_dir = run_pipeline(
                input,
                mode=mode,
                use_llm=not no_llm,
                provider_name=provider,
                transcribe=not no_transcribe,
                whisper_model=whisper_model,
                ocr_text=not no_ocr,
                target_format=target,
                merge=merge,
                max_shots=max_shots,
                debug=debug,
                keep_frames=keep_frames,
            )
        except Exception as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")
            raise typer.Exit(code=1) from exc

    result = format_analysis(analysis, target, mode)

    rendered = _render(result, format, mode)

    if output:
        Path(output).write_text(rendered, encoding="utf-8")
        console.print(f"[bold green]Wrote prompts to {output}[/bold green]")
    else:
        print(rendered)

    if debug:
        console.print(f"[bold cyan]Debug overlays:[/bold cyan] {work_dir / 'debug' / 'index.md'}")
    elif keep_frames:
        console.print(f"[bold cyan]Keyframes kept at:[/bold cyan] {work_dir / 'frames'}")


def _render(result: dict, fmt: str, mode: str = "video_prompt") -> str:
    if mode == "breakdown":
        return _render_breakdown(result, fmt)

    if fmt == "json":
        return json.dumps(result, indent=2)
    if fmt == "txt":
        lines = [s["prompt"] for s in result["shots"]]
        if "master_prompt" in result:
            lines.append("")
            lines.append("--- MASTER PROMPT ---")
            lines.append(result["master_prompt"])
        return "\n\n".join(lines)
    # default: md
    lines = [f"# Prompts for {result['source']}", ""]
    for shot in result["shots"]:
        lines.append(f"## Shot {shot['shot_index']} ({shot['start_time']:.1f}s - {shot['end_time']:.1f}s)")
        lines.append("")
        lines.append(shot["prompt"])
        lines.append("")
    if "master_prompt" in result:
        lines.append("## Master Prompt")
        lines.append("")
        lines.append(result["master_prompt"])
    return "\n".join(lines)


def _render_breakdown(result: dict, fmt: str) -> str:
    if fmt == "json":
        return json.dumps(result, indent=2)

    lines = [f"Prompt Breakdown of {Path(result['source']).name} (~{result['duration']:.0f} seconds)"]
    for scene in result["scenes"]:
        start_str = f"{int(scene['start_time'] // 60)}:{int(scene['start_time'] % 60):02d}"
        end_str = f"{int(scene['end_time'] // 60)}:{int(scene['end_time'] % 60):02d}"
        header = f"Scene {scene['shot_index'] + 1} ({start_str}–{end_str})"
        if scene.get("scene_role"):
            header += f" — {scene['scene_role']}"
        lines.append("")
        lines.append(header)
        if scene.get("scene_description"):
            lines.append(scene["scene_description"])
        texts = scene.get("on_screen_text") or []
        if texts:
            quoted = " → ".join(f'"{t}"' for t in texts)
            lines.append(f"On-screen text: {quoted}")

    if result.get("core_prompt"):
        lines.append("")
        lines.append("Core prompt in one sentence:")
        lines.append(f"> {result['core_prompt']}")

    return "\n".join(lines)


if __name__ == "__main__":
    app()