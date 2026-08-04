"""Manual smoke test for the generation module's per-frame rendering.

Loads a reference photo, runs the full frame-by-frame pipeline (motion planning
-> warp -> identity-preserving render), and dumps each rendered frame as a PNG
for manual visual QA -- no video encoding yet (see scripts/generate_video.py
once app/generation/encode/ exists). Meant to be run by hand on vast.ai against
real model weights, not as part of the automated pytest suite.

Generation is two-stage by default (generation.inpaint.enabled), but this script
is about eyeballing per-frame rendering, so --no-inpaint is usually what you
want here; otherwise supply the stage-1 prompt with --inpaint-prompt.

Usage:
    python scripts/smoke_test_generation.py ref.jpg "a person smiling" --frames 16 --no-inpaint --out outputs/smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from app.core.config import (
	INPAINT_PROMPT_REQUIRED_MESSAGE,
	apply_inpaint_overrides,
	get_settings,
	inpaint_prompt_missing,
)
from app.identity.exceptions import IdentityModuleError
from app.identity.factory import build_identity_engine
from app.identity.interfaces import IdentityReference
from app.generation.exceptions import GenerationModuleError
from app.generation.factory import build_generation_pipeline
from app.generation.interfaces import CameraMotionSpec, GenerationRequest


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("reference_image", type=Path, help="reference photo of the person")
	parser.add_argument("prompt", type=str, help="text prompt describing the scene/motion")
	parser.add_argument("--negative-prompt", type=str, default="")
	parser.add_argument("--frames", type=int, default=None, help="override GenerationConfig.motion.num_frames")
	parser.add_argument("--seed", type=int, default=None)
	parser.add_argument("--out", type=Path, default=Path("outputs/smoke"), help="directory to dump PNG frames into")
	parser.add_argument(
		"--inpaint-prompt",
		type=str,
		default=None,
		help="stage-1 prompt: what the masked garment/body region should become "
		"(default: generation.inpaint.prompt; required unless --no-inpaint)",
	)
	parser.add_argument(
		"--no-inpaint",
		action="store_true",
		help="skip stage 1 -- the usual choice here, since this script exists to eyeball per-frame rendering",
	)
	return parser.parse_args()


def main() -> int:
	args = _parse_args()

	settings = get_settings()
	print(f"device={settings.identity.device} dtype={settings.identity.dtype}")

	identity_engine = build_identity_engine(settings.identity)
	if identity_engine.face_adapter is None:
		print("FAIL: no face adapter enabled (identity.ipadapter.enabled / identity.instantid.enabled)")
		return 1

	generation_config = apply_inpaint_overrides(
		settings.generation, prompt=args.inpaint_prompt, disabled=args.no_inpaint
	)
	if inpaint_prompt_missing(generation_config):
		print(f"FAIL: {INPAINT_PROMPT_REQUIRED_MESSAGE}")
		return 1

	generation_pipeline = build_generation_pipeline(generation_config, settings.identity, identity_engine)

	reference = IdentityReference(images=[Image.open(args.reference_image).convert("RGB")])
	motion = generation_config.motion
	spec = CameraMotionSpec(
		mode=motion.mode,
		direction=motion.direction,
		zoom_range=motion.zoom_range,
		pan_fraction=motion.pan_fraction,
		easing=motion.easing,
		num_frames=args.frames if args.frames is not None else generation_config.output.fps * int(
			generation_config.output.duration_seconds
		),
		fps=generation_config.output.fps,
	)
	request = GenerationRequest(
		reference=reference,
		prompt=args.prompt,
		negative_prompt=args.negative_prompt,
		motion=spec,
		seed=args.seed,
	)

	try:
		result = generation_pipeline.generate(request)
	except (IdentityModuleError, GenerationModuleError) as exc:
		print(f"FAIL: {exc}")
		return 1

	args.out.mkdir(parents=True, exist_ok=True)
	for frame in result.frames:
		frame_path = args.out / f"frame_{frame.frame_index:04d}.png"
		frame.image.save(frame_path)
	print(f"rendered {len(result.frames)} frames (fps={result.fps}) -> {args.out}")
	print("OK")
	return 0


if __name__ == "__main__":
	sys.exit(main())
