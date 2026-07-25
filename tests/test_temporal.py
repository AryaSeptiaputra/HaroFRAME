from __future__ import annotations

import pytest
from PIL import Image

from app.core.config import TemporalConfig
from app.generation.exceptions import GenerationModuleError
from app.generation.temporal.ema_blend import EmaFrameSmoother
from app.generation.temporal.factory import build_temporal_smoother
from app.generation.temporal.passthrough import NullTemporalSmoother


class TestNullTemporalSmoother:
	def test_returns_frames_unchanged(self):
		frames = [Image.new("RGB", (8, 8), color=(i, i, i)) for i in range(3)]

		result = NullTemporalSmoother().smooth(frames)

		assert result == frames
		assert result is not frames


class TestEmaFrameSmoother:
	def test_rejects_invalid_smoothing_strength(self):
		with pytest.raises(GenerationModuleError):
			EmaFrameSmoother(smoothing_strength=1.5)

	def test_single_frame_returned_unchanged(self):
		frames = [Image.new("RGB", (8, 8), color=(1, 2, 3))]

		result = EmaFrameSmoother().smooth(frames)

		assert len(result) == 1

	def test_preserves_frame_count_and_size(self):
		frames = [Image.new("RGB", (16, 16), color=(i * 10, i * 10, i * 10)) for i in range(4)]

		result = EmaFrameSmoother(smoothing_strength=0.5).smooth(frames)

		assert len(result) == 4
		for frame in result:
			assert frame.size == (16, 16)

	def test_zero_strength_keeps_current_frame_dominant(self):
		frames = [
			Image.new("RGB", (16, 16), color=(0, 0, 0)),
			Image.new("RGB", (16, 16), color=(200, 200, 200)),
		]

		result = EmaFrameSmoother(smoothing_strength=0.0).smooth(frames)

		r, _, _ = result[1].getpixel((8, 8))
		assert r == pytest.approx(200, abs=2)


class TestBuildTemporalSmoother:
	def test_none_method_returns_null_smoother(self):
		smoother = build_temporal_smoother(TemporalConfig(method="none"))

		assert isinstance(smoother, NullTemporalSmoother)

	def test_ema_method_returns_ema_smoother_with_configured_strength(self):
		smoother = build_temporal_smoother(TemporalConfig(method="ema", smoothing_strength=0.3))

		assert isinstance(smoother, EmaFrameSmoother)
		assert smoother._alpha == pytest.approx(0.3)
