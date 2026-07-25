# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

HaroFRAME: an image2video generation system with identity (face) preservation. The
identity-preservation module (`app/identity/`) is fully built; the actual video
generation pipeline (`app/generation/`) is an empty stub and not yet implemented.

## Local vs. vast.ai execution

**Real runs (anything touching torch/diffusers/actual model weights) happen on a
vast.ai GPU rental instance, never on the local dev machine.** The local `.venv`
here is intentionally minimal (only `pydantic`/`pydantic-settings` installed) —
`numpy`, `pillow`, `torch`, `diffusers`, `insightface`, `opencv`, `pytest`, etc. are
deliberately **not** installed locally. Don't `pip install` the full dependency set
locally unless explicitly asked; it's several GB and isn't how this project is meant
to run. Deployment is via the root `Dockerfile` — see "Deployment" below.

## Commands

- Syntax-check a file without any deps installed: `python -m py_compile <file>`
- Run tests: `python -m pytest` / a single test: `python -m pytest tests/test_fusion.py::test_fuse_mean_normalizes_average`
  - Only `tests/test_config.py` can actually execute in the local `.venv` today (needs
    just pydantic). `tests/test_fusion.py` needs `numpy`; `tests/test_factory_wiring.py`
    needs `torch`/`diffusers`/`pillow` — these will only run inside the Docker/vast.ai
    environment until those are installed.
- Install full stack (inside the Docker image / on vast.ai, not locally): `pip install -e ".[gpu]"`
- Install the optional face-restoration extra (basicsr has no working build on
  Python 3.13 — sdist bug, no PyPI wheel — hence it's excluded from the Docker image
  by default): `pip install -e ".[restoration]"`
- Manual end-to-end smoke test (needs real deps + reference photos, run on vast.ai):
  `python scripts/smoke_test_identity.py ref1.jpg [ref2.jpg ...] [--driving driving.jpg]`
- Build the vast.ai image: `docker build -t <registry-user>/haroframe:latest .` (see header comment in `Dockerfile` for the full build/push/run workflow and required env vars)

## Config

`app/core/config.py` uses `pydantic-settings`. `Settings` reads env vars prefixed
`HAROFRAME_` with nested delimiter `__` (e.g. `HAROFRAME_IDENTITY__DEVICE=cpu`,
`HAROFRAME_IDENTITY__IPADAPTER__ENABLED=true`), plus an optional `.env` file (none
committed). `get_settings()` is an `lru_cache`d singleton — always go through it
rather than constructing `Settings()` directly outside of tests.

`IdentityConfig` bundles five independent sub-configs (`face`, `ipadapter`,
`instantid`, `controlnet`, `restoration`) plus top-level `device`/`dtype`/
`cache_dir`/`hf_token`/`base_sdxl_model`.

## Identity module architecture (`app/identity/`)

Protocol-based provider design defined in `app/identity/interfaces.py`:
`FaceEmbedding` → `IdentityReference` (one or more ref photos + their embeddings +
a fused embedding) → `StructureHint` (pose/depth driving image, deliberately
separate from the identity reference — may come from an unrelated video frame) →
`IdentityConditioning` (the pipeline-ready kwargs bundle a provider produces).

**Exactly one face adapter at a time.** `app.identity.ipadapter` and
`app.identity.instantid` both satisfy the `FaceConditioningProvider` protocol, so
`IdentityEngine` holds a single `face_adapter` slot regardless of which is
configured. Enabling both `config.ipadapter.enabled` and `config.instantid.enabled`
raises `ConflictingAdapterConfigError` (see `app/identity/factory.py`) rather than
silently picking one.

- **`face/`** — `FaceAnalyzer` protocol + `InsightFaceAnalyzer` (lazy-loaded
  InsightFace `FaceAnalysis`, model only loads on first `analyze()` call) +
  `fusion.py` with three strategies for combining multiple reference photos'
  embeddings into one (`mean`, `weighted_by_det_score`, `best_quality`).
- **`ipadapter/`** — `IdentityAdapter` protocol with two providers dispatched by
  `config.variant` via `factory.build_ipadapter_provider()`: `clip_ipadapter.py`
  (plain CLIP-image conditioning — style/appearance only, weak identity fidelity)
  and `faceid_sdxl.py` (conditions on the ArcFace embedding — used as the default,
  stronger identity fidelity).
- **`controlnet/`** — `StructureConditioner` protocol with `pose_dwpose.py` (DWPose,
  falls back to OpenPose if DWPose ONNX weights can't load) and `depth.py`
  (Midas or Zoe depth estimation). `factory.build_structure_conditioners()` builds
  0, 1, or 2 conditioners from config flags. Deliberately independent of the face
  adapter — the driving image for structure can differ from the identity reference.
- **`instantid/`** — hybrid provider combining an ArcFace embedding (via ip-adapter
  cross-attention) with a facial-keypoint IdentityNet ControlNet.
  `vendor/` is **frozen vendored code** from instantX-research/InstantID
  (Apache-2.0) — do not edit except for compatibility patches, and note any patch
  inline where made. `pipeline.py` builds the special
  `StableDiffusionXLInstantIDPipeline` (not a plain diffusers pipeline) that
  `provider.py`'s `InstantIdProvider.load()`/`build_conditioning()` require.
- **`restoration/`** — optional post-hoc face restoration (GFPGAN) run on an
  already-generated frame; not part of identity conditioning itself.
  `gfpgan_restorer.py` defers and guards its `gfpgan`/`basicsr`/`facexlib` imports,
  raising `ModelLoadError` with a clear message until they're actually installed
  (see the Python 3.13/basicsr note above).
- **`engine.py` / `factory.py`** — top-level orchestration.
  `IdentityEngine.prepare_reference()` runs face analysis + fusion in place on an
  `IdentityReference`. `build_conditioning()` merges the face adapter's kwargs with
  any structure conditioners' ControlNet kwargs — using diffusers' multi-controlnet
  list convention when two conditioners are combined — and *skips* separate
  structure conditioners when the face adapter already carries its own structure
  signal (InstantID's keypoint ControlNet), to avoid colliding `controlnet`/
  `control_image` kwargs. `build_identity_engine(config)` wires the whole thing from
  `IdentityConfig`.

## Deployment

The root `Dockerfile` targets vast.ai: `python:3.11-slim` base (not local's 3.13 —
this sidesteps the basicsr wheel issue), installs `pip install -e ".[gpu]"` only
(the `restoration` extra is intentionally excluded from the baked image), and its
`CMD` is `sleep infinity` — the container stays idle and is accessed via vast.ai's
own exec/instance-portal feature (no bundled SSH server). HF token and feature-flag
env vars (`HAROFRAME_IDENTITY__...`) are meant to be set at vast.ai
instance-creation time, never baked into the image.
