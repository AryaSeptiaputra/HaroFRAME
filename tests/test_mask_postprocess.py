from __future__ import annotations

import numpy as np
from PIL import Image

from app.identity.segmentation.mask_postprocess import dilate_mask, feather_mask


def _small_square_mask() -> Image.Image:
	array = np.zeros((50, 50), dtype=np.uint8)
	array[20:30, 20:30] = 255
	return Image.fromarray(array, mode="L")


def test_dilate_mask_grows_white_region():
	mask = _small_square_mask()
	original_count = np.array(mask).sum()

	dilated = dilate_mask(mask, dilation_px=5)

	assert np.array(dilated).sum() > original_count
	assert dilated.size == mask.size
	assert dilated.mode == "L"


def test_dilate_mask_grows_monotonically_with_dilation_px():
	mask = _small_square_mask()

	small = dilate_mask(mask, dilation_px=2)
	large = dilate_mask(mask, dilation_px=8)

	assert np.array(large).sum() > np.array(small).sum()


def test_dilate_mask_noop_for_zero_dilation():
	mask = _small_square_mask()

	result = dilate_mask(mask, dilation_px=0)

	assert np.array_equal(np.array(result), np.array(mask))


def test_feather_mask_preserves_size_and_mode():
	mask = _small_square_mask()

	feathered = feather_mask(mask, feather_px=3)

	assert feathered.size == mask.size
	assert feathered.mode == "L"


def test_feather_mask_softens_edges():
	mask = _small_square_mask()

	feathered = feather_mask(mask, feather_px=5)
	array = np.array(feathered)

	# A soft edge introduces intermediate gray values that a hard-edged mask lacks.
	unique_values = np.unique(array)
	assert len(unique_values) > 2
