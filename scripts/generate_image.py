"""CLI entry point for single-image identity-preserving image2image generation.

Unlike scripts/generate_video.py (which animates the reference photo into a
video via synthesized camera motion), this renders exactly one output image
directly from the reference photo itself -- no motion planning, no video
encoding. The same photo is both the identity source and the image being
transformed: "restyle/re-scene this photo" rather than "animate this photo".

Reuses the exact same identity engine and adapter-appropriate FrameRenderer
(Img2ImgFrameRenderer for IP-Adapter, InstantIdFrameRenderer for InstantID) as
the video pipeline -- see app.generation.factory.build_frame_renderer.

Needs real model weights and a GPU -- run this on vast.ai (see VAST_GUIDE.md),
not on the local dev machine.

Usage:
    python scripts/generate_image.py ref.jpg "anime style portrait, studio lighting" --out outputs/ref_img2img.png
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from PIL import Image

from app.core.config import get_settings
from app.identity.exceptions import IdentityModuleError
from app.identity.factory import build_identity_engine
from app.identity.interfaces import IdentityReference
from app.generation.exceptions import GenerationModuleError
from app.generation.factory import build_frame_renderer

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
	return parser.parse_args()


def main() -> int:
	args = _parse_args()
	settings = get_settings()
	print(f"device={settings.identity.device} dtype={settings.identity.dtype}")

	identity_engine = build_identity_engine(settings.identity)
	if identity_engine.face_adapter is None:
		print("FAIL: no face adapter enabled (identity.ipadapter.enabled / identity.instantid.enabled)")
		return 1

	try:
		renderer = build_frame_renderer(identity_engine, settings.identity, settings.generation)
	except GenerationModuleError as exc:
		print(f"FAIL: {exc}")
		return 1

	source_image = Image.open(args.reference_image).convert("RGB")
	reference = IdentityReference(images=[source_image])
	seed = args.seed if args.seed is not None else random.randint(0, _MAX_SEED)

	try:
		identity_engine.prepare_reference(reference)
		rendered = renderer.render(
			source_image,
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
