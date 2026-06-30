# Contributing

Thanks for considering a contribution. This is a small personal/open-source
project, so the bar is mostly "does it work and is it tested," not perfection.

## Setup

```bash
pip install -e ".[dev,all]"
pytest
```

## Adding a new prompt-format target (e.g. for a new video-gen model)

1. Add a module under `src/video2prompt/formatters/`, e.g. `kling.py`, with a
   single `format_prompt(shot: ShotRecord) -> str` function.
2. Register it in `formatters/__init__.py`'s `_REGISTRY` dict.
3. Add a test in `tests/test_formatters.py` covering at least one
   representative shot.
4. Note in your module's docstring that vendor prompt conventions change over
   time and should be periodically rechecked against current docs (see
   `SPEC.md` §9 for the existing pattern).

## Adding/improving a CV module

CV analysis modules live in `src/video2prompt/cv_analysis/`. Each one takes a
frame path (or paths) and returns a typed object from `schema.py`. If you add
a new signal (e.g. a better motion estimator), keep the existing function
signature where possible so `assemble.py` doesn't need restructuring, and add
a corresponding field to `ShotRecord` in `schema.py` if it's new structured
data.

## Tests

- Keep CV/formatter/template tests deterministic and dependency-light — they
  should run in CI without `ffmpeg`, API keys, or GPU models installed.
- LLM-backed code paths (`llm/anthropic_provider.py`) aren't unit tested
  against the real API in CI; if you change that module, test manually with a
  real key locally.

## Code style

`ruff` is configured as a dev dependency; run `ruff check .` before opening a
PR. No strict formatting enforcement beyond that for now.
