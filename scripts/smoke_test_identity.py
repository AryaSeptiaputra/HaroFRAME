"""Manual smoke test for the identity module's wiring.

Loads real reference photos, runs face analysis + embedding fusion, and builds
adapter conditioning kwargs through whichever face adapter is enabled in config
-- without running an actual diffusion pipeline. This is meant to be run by hand
against real images/model weights, not as part of the automated pytest suite
(see tests/ for those).

Usage:
    python scripts/smoke_test_identity.py ref1.jpg [ref2.jpg ...] [--driving driving.jpg]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from app.core.config import get_settings
from app.identity.exceptions import IdentityModuleError
from app.identity.factory import build_identity_engine
from app.identity.interfaces import IdentityReference, StructureHint


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("reference_images", nargs="+", type=Path, help="one or more reference photos of the same person")
	parser.add_argument("--driving", type=Path, default=None, help="optional driving frame for pose/depth/kps structure conditioning")
	return parser.parse_args()


def main() -> int:
	args = _parse_args()

	settings = get_settings()
	identity_config = settings.identity
	print(f"device={identity_config.device} dtype={identity_config.dtype}")

	engine = build_identity_engine(identity_config)
	if engine.face_adapter is None:
		print(
			"no face adapter enabled (identity.ipadapter.enabled / identity.instantid.enabled "
			"are both false) -- conditioning will not be built, only face analysis + fusion."
		)

	images = [Image.open(path).convert("RGB") for path in args.reference_images]
	reference = IdentityReference(images=images)

	try:
		engine.prepare_reference(reference)
	except IdentityModuleError as exc:
		print(f"FAIL: face analysis/fusion: {exc}")
		return 1

	print(f"analyzed {len(reference.embeddings)}/{len(images)} reference image(s):")
	for embedding in reference.embeddings:
		print(f"  source={embedding.source_image_id} det_score={embedding.det_score:.3f}")
	fused = reference.fused_embedding
	print(f"fused embedding: dim={fused.vector.shape[0]} det_score={fused.det_score:.3f}")

	if engine.face_adapter is None:
		return 0

	structure = None
	if args.driving is not None:
		driving_image = Image.open(args.driving).convert("RGB")
		structure = StructureHint(pose_image=driving_image, depth_image=driving_image, source="driving_frame")

	try:
		conditioning = engine.build_conditioning(reference, structure=structure)
	except IdentityModuleError as exc:
		print(f"FAIL: build_conditioning: {exc}")
		return 1

	print(f"conditioning built: applied_adapters={conditioning.applied_adapters}")
	print(f"  used_structure_conditioning={conditioning.used_structure_conditioning}")
	print(f"  adapter_kwargs keys={sorted(conditioning.adapter_kwargs.keys())}")
	print("OK")
	return 0


if __name__ == "__main__":
	sys.exit(main())
