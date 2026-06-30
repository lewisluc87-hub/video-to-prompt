# Example: `--mode video_prompt` output (Runway target)

Generated using `video2prompt clip.mp4 --target runway --merge`, on a short
multi-shot clip containing a person holding a sign, a close-up of an object,
and a wide moving train shot. Demonstrates the default visual-description
mode aimed at recreating footage with a video generation model, as opposed
to `--mode breakdown`'s structured scene-by-scene breakdown (see
`sample_breakdown_output.md`).

The `runway` formatter produces short, comma-separated descriptors (subject,
action, camera move, style) matching Runway's Gen-3/4 prompt conventions; the
`generic`, `sora`, and `veo` targets are available too and reformat the same
underlying per-shot description differently (see `SPEC.md` section 9 and
`src/video2prompt/formatters/`).

---

## Shot 0 (0.0s - 2.5s)

person, stop sign, close-up, camera zoom out slow, mid lighting, #282926, #d5d1cb

## Shot 1 (2.5s - 3.4s)

backpack, medium close-up, camera tilt down-to-up fast, mid lighting, #181c21, #63737f

## Shot 2 (12.0s - 13.4s)

train, close-up, camera tilt up-to-down medium, low-key lighting, #141122, #b22528

## Master Prompt

A close-up shot zooms slowly out from a person holding a stop sign against a
neutral backdrop, cutting to a fast upward tilt revealing a backpack in
medium close-up, then to a close-up of a train under low-key lighting with a
slow downward tilt -- muted, slightly desaturated color palette throughout,
naturalistic mid-to-low lighting, handheld-feeling camera movement between
each shot.

---

### Same shot, different targets (for comparison)

**`generic`** (plain descriptive paragraph, default):
> Medium close-up of a person holding a red stop sign, positioned right of
> frame against a neutral gray background. The camera performs a slow zoom
> out, gradually revealing more of the scene. Lighting is flat and even,
> mid-key, with a muted color palette of dark grays and a single red accent
> from the sign.

**`sora`** (cinematic narrative framing):
> A cinematic shot: A person stands holding a red stop sign, framed in a
> medium close-up against a plain gray backdrop. The camera slowly zooms
> out, widening the frame and revealing more context around the subject,
> under soft, even lighting.

**`veo`** (explicit camera-direction clause):
> A person holds a red stop sign in a medium close-up against a neutral
> background, evenly lit with a muted gray and red color palette. Camera:
> slow zoom out.
