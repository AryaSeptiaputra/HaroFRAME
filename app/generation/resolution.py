from __future__ import annotations

import math

# SDXL is trained around 1024x1024. Rendering much above that costs VRAM
# superlinearly and tends to duplicate limbs and faces.
SDXL_PIXEL_BUDGET = 1024 * 1024


def sdxl_working_size(size: tuple[int, int]) -> tuple[int, int]:
	"""Render size for a source image: its own aspect ratio, scaled down to SDXL's
	native pixel budget, each side a multiple of 8. Never scales *up*.

	Both stages need this, for different reasons.

	Stage 1 must pass it to the pipeline explicitly: diffusers no longer defaults
	``height``/``width`` to the UNet's sample size, and preprocesses the init
	image and the control image through separate processors -- so leaving both
	None lets the init image keep the photo's own size while the pose control
	image keeps whatever controlnet_aux emitted (512px), and the two disagree
	inside the ControlNet::

	    RuntimeError: The size of tensor a (228) must match the size of
	    tensor b (64) at non-singleton dimension 3

	Stage 2 has no such parameters at all -- SDXL img2img takes its resolution
	from the init image alone -- so there the image itself is resized first.
	Either way the point is the same: a 1824px phone photo denoised at full size
	is what turns a 24GB card into ``torch.OutOfMemoryError``.
	"""
	width, height = size
	scale = min(1.0, math.sqrt(SDXL_PIXEL_BUDGET / float(width * height)))
	# Round *down* to the grid, so the budget is an actual ceiling -- rounding to
	# nearest can push a scaled-down image back over it.
	return (
		max(8, int(width * scale) // 8 * 8),
		max(8, int(height * scale) // 8 * 8),
	)
