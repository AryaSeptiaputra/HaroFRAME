from __future__ import annotations

from PIL import Image

from app.generation.interfaces import FrameTransform
from app.generation.motion.parallax_warp import DepthParallaxWarper


class _FakeDepthEstimator:
	def __init__(self, depth_image):
		self._depth_image = depth_image
		self.estimate_calls = 0

	def estimate(self, image):
		self.estimate_calls += 1
		return self._depth_image


def test_warp_output_size_matches_source():
	depth_image = Image.new("L", (32, 24), color=128)
	estimator = _FakeDepthEstimator(depth_image)
	warper = DepthParallaxWarper(depth_estimator=estimator)
	source = Image.new("RGB", (32, 24), color=(10, 20, 30))
	transform = FrameTransform(scale=1.1, translate_x=2.0, translate_y=1.0, frame_index=0)

	result = warper.warp(source, transform)

	assert result.size == (32, 24)


def test_warp_reuses_cached_depth_map_for_same_source_image():
	depth_image = Image.new("L", (16, 16), color=200)
	estimator = _FakeDepthEstimator(depth_image)
	warper = DepthParallaxWarper(depth_estimator=estimator)
	source = Image.new("RGB", (16, 16), color=(5, 5, 5))
	transform_a = FrameTransform(scale=1.0, translate_x=0.0, translate_y=0.0, frame_index=0)
	transform_b = FrameTransform(scale=1.05, translate_x=1.0, translate_y=0.0, frame_index=1)

	warper.warp(source, transform_a)
	warper.warp(source, transform_b)

	assert estimator.estimate_calls == 1


def test_warp_recomputes_depth_map_for_a_different_source_image():
	depth_image = Image.new("L", (16, 16), color=200)
	estimator = _FakeDepthEstimator(depth_image)
	warper = DepthParallaxWarper(depth_estimator=estimator)
	source_a = Image.new("RGB", (16, 16), color=(5, 5, 5))
	source_b = Image.new("RGB", (16, 16), color=(9, 9, 9))
	transform = FrameTransform(scale=1.0, translate_x=0.0, translate_y=0.0, frame_index=0)

	warper.warp(source_a, transform)
	warper.warp(source_b, transform)

	assert estimator.estimate_calls == 2
