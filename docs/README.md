[← back to README](../README.md)

# Documentation index

The top-level [README](../README.md) covers install, quickstart, and positioning.
Everything below is organised by topic.

- [`agent-loop.md`](agent-loop.md) — the full look/edit/verify/undo loop: looking tools,
  `ImageWorkspace`, multi-region calls, and what none of it fixes
- [`evals.md`](evals.md) — the eighteen evals, the quality eval across 18 images, and what
  each one found and fixed
- [`photo-demo.md`](photo-demo.md) — the five-photo demo and the notes behind it
- [`colour-operations.md`](colour-operations.md) — "more"/"less" refinement, and named
  colours vs. region references in `replace_colour`
- [`colour-spaces.md`](colour-spaces.md) — `space='hsv'` vs `space='oklab'`, and why the
  default stayed HSV
- [`rebrand.md`](rebrand.md) — `rebrand.py`: colormap remap and OCR-based font swap
- [`grounding.md`](grounding.md) — `grounding.py`: naming a thing instead of drawing a box
- [`session.md`](session.md) — `ImageSession`: multi-turn plain-language editing
- [`legend-cleanup.md`](legend-cleanup.md) — cleaning a semi-transparent legend/annotation box
- [`composite.md`](composite.md) — `compose_within`: the hard mask-composite guarantee
- [`verification.md`](verification.md) — `verify_photo.py` / `evals/eval_suite.py`
- [`advanced.md`](advanced.md) — `figsurgeon.advanced`: wrappers around other people's models
- [`objects.md`](objects.md) — `objects.py`: object-level editing, points vs. boxes,
  completeness verdicts
- [`compose-engine.md`](compose-engine.md) — the chart engine's three layers, plain-language
  edits, applying to a new figure
- [`limits.md`](limits.md) — known rough edges
- [`mcp.md`](mcp.md) — the MCP server and the `figsurgeon` command line
