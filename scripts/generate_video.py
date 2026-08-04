"""User-facing CLI entry point for image2video generation.

Wires Settings -> IdentityEngine -> GenerationPipeline -> an actual video file
on disk. Requires real model weights and a GPU-capable environment (run on
vast.ai, see CLAUDE.md) -- not runnable on the local dev machine.

Generation is two-stage by default (generation.inpaint.enabled): a SAM-masked
inpaint pass first changes the person's clothing or generates body regions --
once, before any motion planning -- and the whole clip is then animated from the
edited photo. So there are two prompts: --inpaint-prompt for stage 1, the
positional prompt for stage 2. Pass --no-inpaint for a single-stage run.

Usage:
    python scripts/generate_video.py ref.jpg "gentle breeze" --inpaint-prompt "a red hoodie"
    python scripts/generate_video.py ref.jpg "a person smiling, gentle breeze" --no-inpaint --out outputs/clip.mp4
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
	parser.add_argument(
		"--out", type=Path, default=None, help="output video path (default: <output_dir>/<reference stem>.<format>)"
	)
	parser.add_argument(
		"--frames", type=int, default=None, help="override frame count (default: fps * duration_seconds)"
	)
	parser.add_argument("--seed", type=int, default=None)
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
		help="skip stage 1 and animate straight from the reference photo",
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
	output_cfg = generation_config.output
	motion_cfg = generation_config.motion
	spec = CameraMotionSpec(
		mode=motion_cfg.mode,
		direction=motion_cfg.direction,
		zoom_range=motion_cfg.zoom_range,
		pan_fraction=motion_cfg.pan_fraction,
		easing=motion_cfg.easing,
		num_frames=args.frames if args.frames is not None else int(output_cfg.fps * output_cfg.duration_seconds),
		fps=output_cfg.fps,
	)
	request = GenerationRequest(
		reference=reference,
		prompt=args.prompt,
		negative_prompt=args.negative_prompt,
		motion=spec,
		seed=args.seed,
		# --inpaint-prompt already went into generation_config above; leaving this
		# unset keeps one source of truth and lets InpaintConfig.prompt apply.
	)

	output_path = args.out
	if output_path is None:
		output_cfg.output_dir.mkdir(parents=True, exist_ok=True)
		output_path = output_cfg.output_dir / f"{args.reference_image.stem}.{output_cfg.format}"

	try:
		result = generation_pipeline.generate(request, output_path=output_path)
	except (IdentityModuleError, GenerationModuleError) as exc:
		print(f"FAIL: {exc}")
		return 1

	if result.output_path is None:
		print("FAIL: video encoding did not run (no video encoder configured)")
		return 1

	print(f"rendered {len(result.frames)} frames (fps={result.fps}) -> {result.output_path}")
	print("OK")
	return 0


if __name__ == "__main__":
	sys.exit(main())
