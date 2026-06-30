from __future__ import annotations

from .base import LLMProvider
from .template_fallback import TemplateProvider


def get_provider(use_llm: bool, provider_name: str = "anthropic") -> LLMProvider:
    if not use_llm:
        return TemplateProvider()

    if provider_name == "anthropic":
        from .anthropic_provider import AnthropicProvider

        try:
            return AnthropicProvider()
        except RuntimeError as exc:
            import sys

            print(f"[video2prompt] {exc}\nFalling back to template mode.", file=sys.stderr)
            return TemplateProvider()

    raise ValueError(f"Unknown LLM provider: {provider_name}")


__all__ = ["LLMProvider", "TemplateProvider", "get_provider"]
