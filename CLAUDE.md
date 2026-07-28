# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

HaroFRAME: an image2video generation system with identity (face) preservation.
`app/identity/` (identity-preserving conditioning) and `app/generation/` (the
frame-by-frame video generation pipeline built on top of it) are both fully built.

## Local vs. vast.ai execution

**Real runs that need actual model weights or a GPU still happen on a vast.ai GPU
rental instance, never on the local dev machine** (`torch.cuda.is_available()` is
`False` locally — CPU-only). The local `.venv` does, however, have the full
dependency set installed (`torch`, `diffusers`, `numpy`, `pillow`, `opencv`,
`insightface`, `pytest`, etc.) for CPU-only import/logic verification — so
`python -m pytest` actually runs and exercises real wiring locally (mocking out
only the parts that need real GPU-side model weights, e.g. `diffusers`
`from_pretrained()` calls, InstantID/SDXL pipeline `__call__`s, and real depth/LoRA
downloads). Don't assume this holds for a fresh clone/environment, though — if
`pytest` fails with `ModuleNotFoundError`, the deps genuinely aren't installed
there and installing them is `pip install -e ".[gpu]"` (several GB). Deployment is
via the root `Dockerfile` — see "Deployment" below.

## Commands

- Run tests: `python -m pytest` / a single test: `python -m pytest tests/test_fusion.py::test_fuse_mean_normalizes_average`
- Syntax-check a file without importing it: `python -m py_compile <file>`
- Install full stack (inside the Docker image / on vast.ai, or locally if asked): `pip install -e ".[gpu]"`
- Install the optional face-restoration extra (basicsr has no working build on
  Python 3.13 — sdist bug, no PyPI wheel — hence it's excluded from the Docker image
  by default): `pip install -e ".[restoration]"`
- Manual end-to-end smoke tests (need real GPU/model weights, run on vast.ai):
  - Identity module only: `python scripts/smoke_test_identity.py ref1.jpg [ref2.jpg ...] [--driving driving.jpg]`
  - Generation, frames only (no video mux, dumps PNGs for visual QA): `python scripts/smoke_test_generation.py ref.jpg "a person smiling" --frames 16 --out outputs/smoke`
  - Full pipeline, real video file (image2video): `python scripts/generate_video.py ref.jpg "a person smiling, gentle breeze" --out outputs/clip.mp4`
  - Single transformed image (image2image, no motion/video): `python scripts/generate_image.py ref.jpg "anime style portrait" --out outputs/ref_img2img.png`
  - Batch over every photo in `test_images/` (gitignored — see `test_images/README.md`), one failure doesn't stop the rest: `python scripts/test_real_images.py --prompt "..."`
  - Interactive (`input()`-based, no new deps), install checkpoints/LoRAs then submit i2v/i2i jobs to a background queue: `python scripts/interactive_generate.py` — see "Interactive CLI" below
- Build the vast.ai image: `docker build -t <registry-user>/haroframe:latest .` — full step-by-step deploy/run walkthrough (instance creation, env vars, getting photos on/videos off the instance, troubleshooting) is in `VAST_GUIDE.md`, not repeated here
- Local Docker Desktop testing (not how vast.ai itself is configured): `docker compose up -d` after `cp .env.example .env` — see `docker-compose.yml`

## Config

`app/core/config.py` uses `pydantic-settings`. `Settings` reads env vars prefixed
`HAROFRAME_` with nested delimiter `__` (e.g. `HAROFRAME_IDENTITY__DEVICE=cpu`,
`HAROFRAME_IDENTITY__IPADAPTER__ENABLED=true`), plus an optional `.env` file (none
committed). `get_settings()` is an `lru_cache`d singleton — always go through it
rather than constructing `Settings()` directly outside of tests.

`IdentityConfig` bundles five independent sub-configs (`face`, `ipadapter`,
`instantid`, `controlnet`, `restoration`) plus top-level `device`/`dtype`/
`cache_dir`/`hf_token`/`base_sdxl_model`. `GenerationConfig` (sibling field
`Settings.generation`) bundles `motion`, `render`, `lora`, `temporal`, `output`
sub-configs for the video pipeline — see "Generation module architecture" below.

`base_sdxl_model` can be either a Hugging Face repo_id/local diffusers
directory *or* a single-file checkpoint path (e.g. a `.safetensors` downloaded
from Civitai/a direct link via `app/generation/source_resolver.py`'s
`resolve_model_source()`) — both renderers build their pipeline through
`app/identity/sdxl_pipeline_loader.py:load_sdxl_pipeline()`, which dispatches
to `from_single_file()` vs `from_pretrained()` based on whether the value is
an existing local file. `resolve_model_source()` is shared by LoRA and
checkpoint installation alike — same four source kinds either way (local
path, direct URL, Civitai model-page/download URL, bare repo_id) — see
`scripts/interactive_generate.py`.

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

## Generation module architecture (`app/generation/`)

Motion is **synthesized, not driven** — there's no external driving video/pose
sequence, just a text prompt plus optional camera-motion params. Every video is
frame-by-frame SDXL generation (no AnimateDiff/SVD/video-diffusion backbone):
a `CameraMotionPlanner` plans a pan/zoom trajectory, each frame gets warped from
the source photo along that trajectory, then re-rendered through the
identity-preserving stack with a **fixed seed across all frames** for temporal
continuity, then optionally smoothed, then muxed into a video file.

**The frame renderer branches on which face adapter `IdentityEngine` is
holding** (dispatched once, in `app/generation/factory.py` — nowhere else
branches on adapter type): the vendored InstantID pipeline has no img2img/
`strength` entry point at all (`vendor/pipeline_stable_diffusion_xl_instantid.py`
always denoises from pure noise), so it can't share one renderer implementation
with the IP-Adapter family.

- **`motion/`** — `CameraMotionPlanner` implementations. `KenBurns2DPlanner` is
  **face-aware**: it clamps zoom/pan per frame (exact geometry, not an
  approximation) so the face — from `FaceEmbedding.bbox` — never leaves the
  crop, even under aggressive pan/zoom. `StaticMotionPlanner` is a zero-motion
  baseline. `DepthParallaxPlanner` reuses `KenBurns2DPlanner`'s trajectory
  unchanged — `mode="depth_parallax"` only changes the *warper*
  (`DepthParallaxWarper`, which displaces pixels by scene depth from
  `app/identity/controlnet/depth_estimator.py`'s shared `DepthEstimator` before
  the usual crop/zoom), not the trajectory. `factory.py` has two dispatch
  functions: `build_motion_planner()` and `build_frame_warper()`.
- **`renderer/`** — `Img2ImgFrameRenderer` (IP-Adapter/FaceID-SDXL branch): warped
  frame as `StableDiffusionXLImg2ImgPipeline` init image, low-moderate
  `strength`, reuses `IdentityEngine.build_conditioning()` directly.
  `InstantIdFrameRenderer`: no img2img possible, so **re-detects facial
  landmarks on the warped frame itself** each render (reusing
  `InsightFaceAnalyzer`) rather than transforming the reference photo's
  landmarks through the warp matrix — robust to non-affine warps (depth
  parallax) and self-correcting frame to frame. The **identity embedding always
  stays locked to the original reference photo**; only the landmark *layout*
  (via `dataclasses.replace`) comes from the warped frame. Falls back to the
  reference's own landmarks (`face_detected=False` on the result) if
  redetection fails on a given frame, rather than failing the frame outright.
- **`lora/`** — `PeftLoraManager`: stacks multiple *named* style/aesthetic
  LoRAs (`GenerationConfig.lora.entries`, capped by `max_active_loras`, default
  3 — validated at config-parse time) on top of whichever pipeline a renderer
  builds, via diffusers' PEFT multi-adapter API
  (`load_lora_weights(adapter_name=...)` + one combined `set_adapters()` call
  that always includes every already-attached adapter, e.g. the reserved
  `"faceid"` companion LoRA from `FaceIdSdxlProvider`, since `set_adapters()`
  only guarantees the weights of adapters named in that specific call).
  Source resolution itself lives in `app/generation/source_resolver.py`
  (`resolve_model_source()`, shared with SDXL checkpoint installation) —
  accepts a local path, a direct download URL, a Civitai model-page URL
  (resolved via Civitai's API, optionally authenticated with
  `GenerationConfig.lora.civitai_api_key`), or a bare HF repo_id (left
  untouched — diffusers resolves those itself).
- **`temporal/`** — `NullTemporalSmoother` is the default (`TemporalConfig.method
  = "none"`). `EmaFrameSmoother` motion-compensates via opencv dense optical
  flow (Farneback) before blending — naive (non-motion-compensated) blending
  would smear/ghost under the camera pan every clip already has. Not the
  default until validated against real output (see its docstring — plain
  blending can trade flicker for blur, which may not be a better trade).
- **`encode/`** — `ImageioVideoEncoder`, backed by `imageio`/`imageio-ffmpeg`
  (bundled ffmpeg binary, no system install needed) — a **core** dependency,
  not optional, since video output is this module's whole job.
- **`pipeline.py` / `factory.py`** — `GenerationPipeline` (the `IdentityEngine`
  analog): plans motion → warps+renders each frame with one fixed seed →
  optional temporal smoothing → optional encoding to `output_path`. Never
  branches on adapter/motion-mode type itself — takes an already-built
  `IdentityEngine` plus already-dispatched planner/warper/renderer/smoother/
  encoder. `build_generation_pipeline(generation_config, identity_config,
  identity_engine)` is the one place all the `isinstance`/mode dispatch happens.
  `generate()` takes an optional `progress_callback(frames_done, total_frames)`,
  invoked after each frame — used by the interactive CLI's queue status view,
  not otherwise wired to anything. `factory.build_frame_renderer()` (same
  isinstance dispatch, minus motion/encoding) is public on its own too, for
  callers that want a single rendered frame with no video pipeline at all —
  `scripts/generate_image.py` (image2image: renders once, directly from the
  reference photo, no warp) is the one that does this.

## Interactive CLI (`scripts/interactive_generate.py`)

Install-then-select-then-queue, not configure-then-run-once: (0) enter
Hugging Face/Civitai API keys for the session (hidden via `getpass`, itself
guarded by a manual `sys.stdin.isatty()` check — `getpass.getpass()` on
Windows reads straight from the console via `msvcrt` and hangs instead of
falling back when stdin is piped/redirected, unlike POSIX); (1)/(2) install
one or more SDXL checkpoints and LoRAs into a *pool* (downloaded immediately,
same four source kinds as above) without selecting what's used yet; then a
looping main menu: submit a job (photo, prompt, i2v/i2i mode, mode-specific
overrides, then *select* which pooled checkpoint/LoRA(s) to use for this job),
check queue status, or exit.

`GenerationQueueManager` runs submitted `GenerationJob`s one at a time on a
single background daemon thread (`queue.Queue` + `threading.Thread`) so
submitting returns to the menu immediately instead of blocking; status shows
`queued`/`running` (with `frame X/Y` for i2v, via `progress_callback`
above)/`done`/`failed` per job. **Each job builds its own `IdentityEngine`**
rather than sharing one across jobs in the queue — `FaceIdSdxlProvider.load()`
(and similarly-shaped `load()` methods) track "already loaded" as a bool on
the adapter instance itself, not per-pipeline, so reusing one `IdentityEngine`
across jobs whose selected checkpoint/LoRA differ (hence a fresh pipeline
object each time) would silently skip re-attaching the adapter to later jobs'
pipelines. Rebuilding per job is cheap (no model weights load at construction
time). Exiting with jobs still queued/running warns first, since the daemon
thread (and anything it's mid-render on) is dropped when the process exits.

## Deployment

The root `Dockerfile` targets vast.ai: `python:3.11-slim` base (not local's 3.13 —
this sidesteps the basicsr wheel issue), installs `pip install -e ".[gpu]"` only
(the `restoration` extra is intentionally excluded from the baked image), and its
`CMD` is `sleep infinity` — the container stays idle and is accessed via vast.ai's
own exec/instance-portal feature (no bundled SSH server). HF token and feature-flag
env vars (`HAROFRAME_IDENTITY__...`) are meant to be set at vast.ai
instance-creation time, never baked into the image.
