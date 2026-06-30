from __future__ import annotations

from ..schema import ShotRecord
from . import generic, runway, sora, veo

_REGISTRY = {
    "generic": generic.format_prompt,
    "sora": sora.format_prompt,
    "runway": runway.format_prompt,
    "veo": veo.format_prompt,
}


def format_shot(shot: ShotRecord, target: str = "generic") -> str:
    fn = _REGISTRY.get(target)
    if fn is None:
        raise ValueError(f"Unknown format target: {target}. Options: {list(_REGISTRY)}")
    return fn(shot)


__all__ = ["format_shot"]
