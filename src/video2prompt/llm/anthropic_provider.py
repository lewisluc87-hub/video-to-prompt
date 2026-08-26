"""Anthropic-backed LLM provider.

Two auth modes, controlled by VAP_AUTH_MODE env var (mirrors the pattern
used in Viral Analyzer Pro's generator.py):

  VAP_AUTH_MODE=api_key (default) — standard Anthropic API key billing.
  Install: pip install video2prompt[llm]
  Set env: ANTHROPIC_API_KEY=your_key

  VAP_AUTH_MODE=subscription — PERSONAL USE ONLY. Routes requests through
  the Claude Agent SDK using a personal Claude Pro/Max subscription OAuth
  token instead of API billing. Anthropic's policy: "Unless previously
  approved, Anthropic does not allow third party developers to offer
  claude.ai login or rate limits for their products." This mode is fine
  for your own personal, individual runs of this tool (exactly what
  `claude setup-token` is for) — never expose it to other users of this
  tool, and never make it a toggle in any shared/product UI.

  Setup (one-time, in your own terminal):
    npm install -g @anthropic-ai/claude-code
    claude setup-token          # opens a browser, logs into your Pro/Max account
  Then set env for your personal runs:
    CLAUDE_CODE_OAUTH_TOKEN=<the token from setup-token>
    VAP_AUTH_MODE=subscription
  Do NOT also have ANTHROPIC_API_KEY set in that same environment — it will
  shadow the OAuth token and silently fall back to API billing.

  Install: pip install claude-agent-sdk

  NOTE: unlike VAP's generator.py (text-only prompts), this tool sends
  keyframe images in generate_shot_prompt() and generate_scene_breakdown().
  The subscription path below sends images as extra content blocks in the
  Agent SDK query — this exact path wasn't exercised in the reference
  code, so test those two calls first after switching modes.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path

from ..schema import ShotRecord
from .base import BREAKDOWN_SYSTEM_PROMPT, CORE_PROMPT_SYSTEM_PROMPT, SYSTEM_PROMPT, LLMProvider

_MODEL = "claude-sonnet-4-6"


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = _MODEL):
        self._model = model
        self._mode = os.environ.get("VAP_AUTH_MODE", "api_key")
        self._client = None

        if self._mode == "subscription":
            # Validated lazily on first call (see _call_subscription), so
            # constructing the provider doesn't require the SDK/token to be
            # present until a call is actually made -- matches generator.py.
            return

        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "anthropic package not installed. Run `pip install video2prompt[llm]`, "
                "or use --no-llm for template-only mode."
            ) from exc

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Export it, or run with --no-llm "
                "for fully offline template mode."
            )
        self._client = anthropic.Anthropic(api_key=api_key)

    def generate_shot_prompt(self, shot: ShotRecord, keyframe_paths: list[Path]) -> str:
        content = [{"type": "text", "text": self._shot_context_text(shot)}]
        for p in keyframe_paths[:3]:
            content.append(_image_block(p))
        return self._call(system=SYSTEM_PROMPT, content=content, max_tokens=300).strip()

    def generate_master_prompt(self, shot_prompts: list[str], video_meta: dict) -> str:
        joined = "\n\n".join(f"Shot {i}: {p}" for i, p in enumerate(shot_prompts))
        prompt = (
            "Combine the following per-shot descriptions of a single source video "
            "into ONE cohesive prompt (under 120 words) for a video generation model "
            "that only accepts a single prompt. Preserve the overall narrative arc, "
            "dominant subject, setting, and visual style. Do not list shots separately "
            f"in your output -- write one continuous description.\n\nVideo metadata: "
            f"{json.dumps(video_meta)}\n\n{joined}"
        )
        return self._call(system=SYSTEM_PROMPT, content=prompt, max_tokens=300).strip()

    def generate_scene_breakdown(self, shot: ShotRecord, keyframe_paths: list[Path]) -> dict:
        content = [{"type": "text", "text": self._breakdown_context_text(shot)}]
        for p in keyframe_paths[:3]:
            content.append(_image_block(p))
        text = self._call(system=BREAKDOWN_SYSTEM_PROMPT, content=content, max_tokens=1000).strip()
        return _parse_breakdown_json(text)

    def generate_core_prompt(self, scenes: list[dict], video_meta: dict) -> str:
        lines = []
        for s in scenes:
            role = s.get("scene_role", "")
            desc = s.get("scene_description", "")
            texts = "; ".join(s.get("on_screen_text", []))
            lines.append(f"- {role}: {desc} (on-screen text: {texts})")
        prompt = (
            "Here are the per-scene descriptions of a screen-recorded demo video:\n\n"
            + "\n".join(lines)
            + f"\n\nVideo duration: {video_meta.get('duration', '?')}s"
        )
        return self._call(system=CORE_PROMPT_SYSTEM_PROMPT, content=prompt, max_tokens=150).strip()

    def _call(self, system: str, content, max_tokens: int) -> str:
        """Unified entry point -- routes to API-key billing or personal-
        subscription auth based on VAP_AUTH_MODE, same switch as VAP's
        generator.py::_call_claude(). `content` may be a plain string
        (text-only) or a list of Messages-API-style content blocks
        (text + images)."""
        if self._mode == "subscription":
            return self._call_subscription(system=system, content=content, max_tokens=max_tokens)

        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
        )
        return _extract_text(response)

    def _call_subscription(self, system: str, content, max_tokens: int) -> str:
        """PERSONAL USE ONLY -- see module docstring. Routes through the
        Claude Agent SDK using personal subscription OAuth instead of API
        billing. Mirrors generator.py::_call_claude_subscription()."""
        try:
            from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock
        except ImportError as exc:
            raise RuntimeError(
                "claude-agent-sdk not installed. Run: pip install claude-agent-sdk"
            ) from exc
        if os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is set -- it will shadow your OAuth token and "
                "silently bill your API credits instead of your subscription. "
                "Unset it before using VAP_AUTH_MODE=subscription."
            )
        if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
            raise RuntimeError(
                "CLAUDE_CODE_OAUTH_TOKEN not set. Run 'claude setup-token' once "
                "in your terminal to generate one -- see module docstring."
            )

        # Text-only content goes through as a plain string prompt (identical
        # to generator.py's usage). Content with image blocks is sent as a
        # single-turn message list in Messages-API shape -- UNTESTED path,
        # verify output quality on generate_shot_prompt/generate_scene_breakdown.
        prompt = content if isinstance(content, str) else [
            {"role": "user", "content": content}
        ]

        async def _run() -> str:
            text_parts = []
            async for msg in query(
                prompt=prompt,
                options=ClaudeAgentOptions(
                    system_prompt=system,
                    allowed_tools=[],   # plain text completion, no file/bash/agentic tool use
                    max_turns=5,
                    max_budget_usd=None,
                ),
            ):
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            text_parts.append(block.text)
            return "".join(text_parts)

        return asyncio.run(_run())

    @staticmethod
    def _breakdown_context_text(shot: ShotRecord) -> str:
        text = (
            f"Shot {shot.shot_index}, {shot.start_time:.1f}s - {shot.end_time:.1f}s "
            f"(duration {shot.duration:.1f}s)."
        )
        if shot.on_screen_text:
            text += f"\n\nOCR-extracted on-screen text for this shot:\n{shot.on_screen_text}"
        else:
            text += "\n\nNo on-screen text detected for this shot."
        if shot.transcript_segment:
            text += (
                f"\n\nSpoken transcript overlapping this shot (context only, "
                f"do not quote): {shot.transcript_segment}"
            )
        return text

    @staticmethod
    def _shot_context_text(shot: ShotRecord) -> str:
        payload = shot.model_dump(exclude={"keyframe_paths", "debug_overlay_paths", "generated_prompt"})
        text = f"Structured shot data:\n{json.dumps(payload, indent=2)}"
        if shot.transcript_segment:
            text += (
                f"\n\nTranscript overlapping this shot (CONTEXT ONLY, do not quote): "
                f"{shot.transcript_segment}"
            )
        return text


def _image_block(path: Path) -> dict:
    media_type = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }


def _extract_text(response) -> str:
    return "".join(block.text for block in response.content if getattr(block, "type", None) == "text")


def _parse_breakdown_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    data = _try_parse_json(cleaned)
    if data is None:
        # Likely truncated mid-response (hit max_tokens). Try closing it off.
        data = _try_parse_json(_attempt_json_repair(cleaned))

    if data is None:
        return {
            "scene_role": "Unlabeled",
            "scene_description": "(LLM response was truncated or malformed for this shot; "
                                  "raw on-screen text is still available below.)",
            "on_screen_text": [],
        }

    return {
        "scene_role": data.get("scene_role", "Unlabeled"),
        "scene_description": data.get("scene_description", ""),
        "on_screen_text": data.get("on_screen_text", []) or [],
    }


def _try_parse_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _attempt_json_repair(text: str) -> str:
    """Best-effort repair for a JSON object truncated mid-string/mid-array
    (happens when a response hits max_tokens before finishing). Closes any
    open string, then pads with closing brackets/braces to match what's open.
    """
    repaired = text
    if repaired.count('"') % 2 == 1:
        repaired += '"'
    open_braces = repaired.count("{") - repaired.count("}")
    open_brackets = repaired.count("[") - repaired.count("]")
    repaired += "]" * max(0, open_brackets)
    repaired += "}" * max(0, open_braces)
    return repaired