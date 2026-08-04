"""Fastest path from a photo to a result: reference image, the two prompts the
two pipeline stages need, and a negative prompt -- nothing else to decide.

Everything the other entry points ask about (mode, strength, guidance, steps,
seed, checkpoint, LoRA, ControlNet toggles, output filename) comes from
Settings/config defaults. That makes this the script for iterating on prompts;
reach for scripts/generate_image.py or scripts/interactive_generate.py when you
actually want to turn those knobs.

Two prompts, because generation is two stages by default:
  -i/--inpaint-prompt  what the masked garment/body region becomes (stage 1)
  prompt (positional)  what the finished image looks like (stage 2)

What it decides for you, deliberately:
  - **Always image2image** -- one image out, no motion, no video encoding. One
    render instead of dozens of frames is what makes prompt iteration cheap.
    Use scripts/generate_video.py when you want video.
  - **Stage-1 inpaint follows config** (HAROFRAME_GENERATION__INPAINT__ENABLED),
    which is **on** by default. Pass --no-inpaint for a single-stage run, which
    also drops the SAM checkpoint requirement.
  - **Random seed each run**, printed on the way out, so a result worth keeping
    can be reproduced with `generate_image.py --seed`.
  - **Auto-named output** into generation.output.output_dir, never overwriting:
    <stem>_quick.png, then <stem>_quick_2.png, and so on.

Needs real model weights and a GPU -- run this on vast.ai (see VAST_GUIDE.md),
not on the local dev machine.

Usage:
    python scripts/quick_generate.py ref.jpg "anime style portrait, studio lighting" -i "a red hoodie"
    python scripts/quick_generate.py ref.jpg "portrait on a beach" -i "sleeveless summer top" -n "blurry, extra fingers"
    python scripts/quick_generate.py ref.jpg "anime style portrait" --no-inpaint
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from PIL import Image

from app.core.config import (
	INPAINT_PROMPT_REQUIRED_MESSAGE,
	GenerationConfig,
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
	parser = argparse.ArgumentParser(
		description="One photo + the two stage prompts + a negative prompt -> one image. Everything else is a config default."
	)
	parser.add_argument("reference_image", type=Path, help="photo to both take identity from and transform")
	parser.add_argument("prompt", type=str, help="stage-2 prompt: what the finished image should look like")
	parser.add_argument(
		"-i",
		"--inpaint-prompt",
		type=str,
		default=None,
		help="stage-1 prompt: what the masked garment/body region should become "
		"(default: generation.inpaint.prompt; required unless --no-inpaint)",
	)
	parser.add_argument(
		"-n",
		"--negative",
		type=str,
		default="",
		help="negative prompt: what to keep out of the result (default: generation.render.negative_prompt)",
	)
	parser.add_argument(
		"--no-inpaint",
		action="store_true",
		help="skip stage 1 and render straight from the photo (single-stage, no SAM checkpoint needed)",
	)
	return parser.parse_args()


def _unique_output_path(output_dir: Path, stem: str) -> Path:
	"""<stem>_quick.png, bumping a counter rather than overwriting -- consecutive
	runs on one photo are the normal way this script gets used."""
	candidate = output_dir / f"{stem}_quick.png"
	counter = 2
	while candidate.exists():
		candidate = output_dir / f"{stem}_quick_{counter}.png"
		counter += 1
	return candidate


def _print_resolved(settings, generation_config: GenerationConfig, adapter_name: str, seed: int, args) -> None:
	"""Everything this script chose on the user's behalf, so it isn't a black box."""
	render = generation_config.render
	inpaint = generation_config.inpaint
	print("--- using (all from config; see VAST_GUIDE.md to change) ---")
	print(f"photo    : {args.reference_image}")
	print(f"stage 1  : {inpaint.prompt if inpaint.enabled else '(skipped)'}")
	print(f"stage 2  : {args.prompt}")
	print(f"negative : {args.negative or render.negative_prompt or '(none)'}")
	print(f"model    : {settings.identity.base_sdxl_model}")
	print(f"adapter  : {adapter_name}")
	print(f"render   : strength={render.strength} guidance={render.guidance_scale} steps={render.num_inference_steps}")
	print(f"seed     : {seed}")
	print("---")


def main() -> int:
	args = _parse_args()

	if not args.reference_image.is_file():
		print(f"FAIL: photo not found: {args.reference_image}")
		return 1

	settings = get_settings()
	identity_engine = build_identity_engine(settings.identity)
	if identity_engine.face_adapter is None:
		print(
			"FAIL: no face adapter enabled -- set HAROFRAME_IDENTITY__IPADAPTER__ENABLED=true "
			"(or HAROFRAME_IDENTITY__INSTANTID__ENABLED=true), then run again"
		)
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
	# Identity comes from the original photo; stage 1, if it runs, only replaces
	# the image being transformed. Same ordering GenerationPipeline uses.
	reference = IdentityReference(images=[source_image])
	seed = random.randint(0, _MAX_SEED)

	_print_resolved(settings, generation_config, type(identity_engine.face_adapter).__name__, seed, args)

	try:
		identity_engine.prepare_reference(reference)
		render_source = source_image
		if source_editor is not None:
			print("stage 1/2: inpainting garment/body region ...")
			render_source = source_editor.edit(source_image, negative_prompt=args.negative, seed=seed)
			print("stage 2/2: rendering ...")
		rendered = renderer.render(
			render_source,
			reference=reference,
			prompt=args.prompt,
			negative_prompt=args.negative,
			seed=seed,
			frame_index=0,
		)
	except (IdentityModuleError, GenerationModuleError) as exc:
		print(f"FAIL: {exc}")
		return 1

	output_dir = generation_config.output.output_dir
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = _unique_output_path(output_dir, args.reference_image.stem)
	rendered.image.save(output_path)

	inpaint_flag = (
		f' --inpaint-prompt "{generation_config.inpaint.prompt}"' if generation_config.inpaint.enabled else " --no-inpaint"
	)
	print(f"-> {output_path}")
	print(
		f"OK (seed={seed}; reproduce with: python scripts/generate_image.py "
		f'{args.reference_image} "{args.prompt}"{inpaint_flag} --seed {seed})'
	)
	return 0


if __name__ == "__main__":
	sys.exit(main())
