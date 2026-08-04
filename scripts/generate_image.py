"""CLI entry point for single-image identity-preserving image2image generation.

Unlike scripts/generate_video.py (which animates the reference photo into a
video via synthesized camera motion), this renders exactly one output image
directly from the reference photo itself -- no motion planning, no video
encoding. The same photo is both the identity source and the image being
transformed: "restyle/re-scene this photo" rather than "animate this photo".

Reuses the exact same identity engine and adapter-appropriate FrameRenderer
(Img2ImgFrameRenderer for IP-Adapter, InstantIdFrameRenderer for InstantID) as
the video pipeline -- see app.generation.factory.build_frame_renderer.

Generation is two-stage by default (generation.inpaint.enabled): a SAM-masked
inpaint pass first changes the person's clothing or generates body regions, and
its output becomes the photo this img2img render works from. So there are two
prompts -- --inpaint-prompt for stage 1, the positional prompt for stage 2. Pass
--no-inpaint for a single-stage run straight from the reference photo.

Needs real model weights and a GPU -- run this on vast.ai (see VAST_GUIDE.md),
not on the local dev machine.

Usage:
    python scripts/generate_image.py ref.jpg "standing on a beach" --inpaint-prompt "sleeveless summer top"
    python scripts/generate_image.py ref.jpg "anime style portrait, studio lighting" --no-inpaint --out outputs/ref_img2img.png
"""

from __future__ import annotations

import argparse
import random
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
from app.generation.factory import build_frame_renderer, build_source_editor

_MAX_SEED = 2**31 - 1


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("reference_image", type=Path, help="photo to both take identity from and transform")
	parser.add_argument("prompt", type=str, help="text prompt describing the desired result")
	parser.add_argument("--negative-prompt", type=str, default="")
	parser.add_argument("--out", type=Path, default=None, help="output image path (default: outputs/<stem>_img2img.png)")
	parser.add_argument("--seed", type=int, default=None)
	parser.add_argument(
		"--strength",
		type=float,
		default=None,
		help="img2img strength override (IP-Adapter branch only; ignored -- accepted-and-unused -- by InstantID)",
	)
	parser.add_argument(
		"--inpaint-prompt",
		type=str,
		default=None,
		help="stage-1 prompt: what the masked garment/body region should become "
		"(default: generation.inpaint.prompt; required unless --no-inpaint)",
	)
	parser.add_argument(
		"--inpaint-strength",
		type=float,
		default=None,
		help="stage-1 inpaint strength override (default: generation.inpaint.strength)",
	)
	parser.add_argument(
		"--no-inpaint",
		action="store_true",
		help="skip stage 1 and render straight from the reference photo",
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

	try:
		renderer = build_frame_renderer(identity_engine, settings.identity, generation_config)
		source_editor = build_source_editor(settings.identity, generation_config)
	except GenerationModuleError as exc:
		print(f"FAIL: {exc}")
		return 1

	source_image = Image.open(args.reference_image).convert("RGB")
	# Identity comes from the *original* photo; only the image being transformed
	# is replaced by the stage-1 edit below.
	reference = IdentityReference(images=[source_image])
	seed = args.seed if args.seed is not None else random.randint(0, _MAX_SEED)

	try:
		identity_engine.prepare_reference(reference)
		render_source = source_image
		if source_editor is not None:
			print(f"stage 1: inpainting garment/body region -- {generation_config.inpaint.prompt!r}")
			render_source = source_editor.edit(
				source_image,
				negative_prompt=args.negative_prompt,
				seed=seed,
				strength=args.inpaint_strength,
			)
			# Free stage 1's pipeline before stage 2 builds its own.
			source_editor.release()
		rendered = renderer.render(
			render_source,
			reference=reference,
			prompt=args.prompt,
			negative_prompt=args.negative_prompt,
			seed=seed,
			frame_index=0,
			strength=args.strength,
		)
	except (IdentityModuleError, GenerationModuleError) as exc:
		print(f"FAIL: {exc}")
		return 1

	output_path = args.out or Path("outputs") / f"{args.reference_image.stem}_img2img.png"
	output_path.parent.mkdir(parents=True, exist_ok=True)
	rendered.image.save(output_path)

	print(f"rendered 1 image (seed={seed}) -> {output_path}")
	print("OK")
	return 0


if __name__ == "__main__":
	sys.exit(main())
