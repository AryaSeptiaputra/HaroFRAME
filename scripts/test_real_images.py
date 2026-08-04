"""Batch-test the generation pipeline against real reference photos.

Scans a directory (default: test_images/) for image files (.jpg/.jpeg/.png),
optionally paired with a same-named .txt file containing a per-image prompt,
and runs each through the full identity + generation pipeline, saving a real
video file per image. Needs actual model weights and a GPU -- run this on
vast.ai (see VAST_GUIDE.md), not on the local dev machine. A failure on one image
is reported but does not stop the rest of the batch.

Generation is two-stage by default (generation.inpaint.enabled). The .txt
sidecar / --prompt is the stage-2 prompt and varies per image; --inpaint-prompt
is the stage-1 prompt and applies to the whole batch. Pass --no-inpaint for
single-stage runs.

Usage:
    python scripts/test_real_images.py [--dir test_images] [--prompt "a person, natural lighting"] --no-inpaint
    python scripts/test_real_images.py --inpaint-prompt "a plain white t-shirt" [--out outputs/test_real_images]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
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

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass
class _BatchResult:
	name: str
	ok: bool
	detail: str


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--dir", type=Path, default=Path("test_images"), help="directory to scan for reference photos")
	parser.add_argument(
		"--prompt",
		type=str,
		default="a person, natural lighting, subtle motion",
		help="fallback prompt for images without a matching .txt sidecar file",
	)
	parser.add_argument("--negative-prompt", type=str, default="")
	parser.add_argument(
		"--frames", type=int, default=None, help="override frame count (default: fps * duration_seconds)"
	)
	parser.add_argument("--seed", type=int, default=None)
	parser.add_argument(
		"--out", type=Path, default=Path("outputs/test_real_images"), help="directory to write output videos into"
	)
	parser.add_argument(
		"--inpaint-prompt",
		type=str,
		default=None,
		help="stage-1 prompt applied to every image in the batch: what the masked garment/body "
		"region should become (default: generation.inpaint.prompt; required unless --no-inpaint)",
	)
	parser.add_argument(
		"--no-inpaint",
		action="store_true",
		help="skip stage 1 for the whole batch and animate straight from each photo",
	)
	return parser.parse_args()


def _find_images(directory: Path) -> list[Path]:
	if not directory.is_dir():
		return []
	return sorted(p for p in directory.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS)


def _prompt_for(image_path: Path, default_prompt: str) -> str:
	sidecar = image_path.with_suffix(".txt")
	if sidecar.exists():
		return sidecar.read_text(encoding="utf-8").strip()
	return default_prompt


def main() -> int:
	args = _parse_args()
	images = _find_images(args.dir)
	if not images:
		print(f"no .jpg/.jpeg/.png files found in {args.dir} -- drop some in there first (see {args.dir}/README.md)")
		return 1

	settings = get_settings()
	print(f"device={settings.identity.device} dtype={settings.identity.dtype}")
	print(f"found {len(images)} image(s) in {args.dir}")

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

	output_cfg = generation_config.output
	motion_cfg = generation_config.motion
	args.out.mkdir(parents=True, exist_ok=True)

	results: list[_BatchResult] = []
	for image_path in images:
		print(f"--- {image_path.name} ---")
		try:
			reference = IdentityReference(images=[Image.open(image_path).convert("RGB")])
			spec = CameraMotionSpec(
				mode=motion_cfg.mode,
				direction=motion_cfg.direction,
				zoom_range=motion_cfg.zoom_range,
				pan_fraction=motion_cfg.pan_fraction,
				easing=motion_cfg.easing,
				num_frames=(
					args.frames if args.frames is not None else int(output_cfg.fps * output_cfg.duration_seconds)
				),
				fps=output_cfg.fps,
			)
			request = GenerationRequest(
				reference=reference,
				prompt=_prompt_for(image_path, args.prompt),
				negative_prompt=args.negative_prompt,
				motion=spec,
				seed=args.seed,
			)
			output_path = args.out / f"{image_path.stem}.{output_cfg.format}"
			result = generation_pipeline.generate(request, output_path=output_path)
			if result.output_path is None:
				results.append(_BatchResult(image_path.name, False, "video encoding did not run"))
				print("FAIL: video encoding did not run")
				continue
			results.append(_BatchResult(image_path.name, True, str(result.output_path)))
			print(f"OK -> {result.output_path}")
		except (IdentityModuleError, GenerationModuleError) as exc:
			results.append(_BatchResult(image_path.name, False, str(exc)))
			print(f"FAIL: {exc}")

	print("\n=== summary ===")
	ok_count = sum(1 for r in results if r.ok)
	for r in results:
		status = "OK  " if r.ok else "FAIL"
		print(f"{status} {r.name}: {r.detail}")
	print(f"{ok_count}/{len(results)} succeeded")

	return 0 if ok_count == len(results) else 1


if __name__ == "__main__":
	sys.exit(main())
