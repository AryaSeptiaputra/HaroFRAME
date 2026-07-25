from __future__ import annotations

import pytest
from PIL import Image

from app.generation.exceptions import MotionPlanError
from app.generation.interfaces import CameraMotionSpec
from app.generation.motion.factory import build_motion_planner
from app.generation.motion.ken_burns import KenBurns2DPlanner
from app.generation.motion.static import StaticMotionPlanner
from app.generation.motion.warp import AffineFrameWarper

SOURCE_SIZE = (1000, 800)


def _crop_box(source_size, transform):
	width, height = source_size
	crop_w, crop_h = width / transform.scale, height / transform.scale
	cx, cy = width / 2.0 + transform.translate_x, height / 2.0 + transform.translate_y
	return (cx - crop_w / 2.0, cy - crop_h / 2.0, cx + crop_w / 2.0, cy + crop_h / 2.0)


def _assert_within_bounds(source_size, transform):
	x1, y1, x2, y2 = _crop_box(source_size, transform)
	width, height = source_size
	assert x1 >= -1e-6
	assert y1 >= -1e-6
	assert x2 <= width + 1e-6
	assert y2 <= height + 1e-6


class TestKenBurns2DPlanner:
	def test_frame_count_and_fps(self):
		planner = KenBurns2DPlanner()
		spec = CameraMotionSpec(num_frames=24, fps=12)

		plan = planner.plan(SOURCE_SIZE, spec)

		assert plan.num_frames == 24
		assert plan.fps == 12
		assert len(plan.transforms) == 24
		assert [t.frame_index for t in plan.transforms] == list(range(24))

	def test_all_frames_stay_within_source_bounds(self):
		planner = KenBurns2DPlanner()
		spec = CameraMotionSpec(direction="right", zoom_range=(1.0, 1.2), pan_fraction=(0.0, 0.3), num_frames=10)

		plan = planner.plan(SOURCE_SIZE, spec)

		for transform in plan.transforms:
			_assert_within_bounds(SOURCE_SIZE, transform)

	def test_direction_in_zooms_in_monotonically(self):
		planner = KenBurns2DPlanner()
		spec = CameraMotionSpec(direction="in", zoom_range=(1.0, 1.15), easing="linear", num_frames=8)

		scales = [t.scale for t in planner.plan(SOURCE_SIZE, spec).transforms]

		assert scales == sorted(scales)
		assert scales[0] == pytest.approx(1.0)
		assert scales[-1] == pytest.approx(1.15)

	def test_direction_out_zooms_out_monotonically(self):
		planner = KenBurns2DPlanner()
		spec = CameraMotionSpec(direction="out", zoom_range=(1.0, 1.15), easing="linear", num_frames=8)

		scales = [t.scale for t in planner.plan(SOURCE_SIZE, spec).transforms]

		assert scales == sorted(scales, reverse=True)
		assert scales[0] == pytest.approx(1.15)
		assert scales[-1] == pytest.approx(1.0)

	@pytest.mark.parametrize(
		"direction,attr,expected_sign",
		[("right", "translate_x", 1.0), ("left", "translate_x", -1.0), ("down", "translate_y", 1.0), ("up", "translate_y", -1.0)],
	)
	def test_pan_direction_sign(self, direction, attr, expected_sign):
		planner = KenBurns2DPlanner()
		spec = CameraMotionSpec(direction=direction, zoom_range=(1.0, 1.3), pan_fraction=(0.0, 0.2), easing="linear", num_frames=6)

		last = planner.plan(SOURCE_SIZE, spec).transforms[-1]

		assert getattr(last, attr) * expected_sign > 0

	def test_auto_and_in_produce_no_pan(self):
		planner = KenBurns2DPlanner()
		for direction in ("auto", "in", "out"):
			spec = CameraMotionSpec(direction=direction, pan_fraction=(0.0, 0.5), num_frames=5)
			for transform in planner.plan(SOURCE_SIZE, spec).transforms:
				assert transform.translate_x == pytest.approx(0.0)
				assert transform.translate_y == pytest.approx(0.0)

	def test_face_bbox_keeps_face_inside_crop_for_every_frame(self):
		planner = KenBurns2DPlanner()
		# Off-center face near the right edge, aggressive pan/zoom requested.
		face_bbox = (700.0, 300.0, 820.0, 460.0)
		spec = CameraMotionSpec(
			direction="left", zoom_range=(1.0, 2.5), pan_fraction=(0.0, 0.9), num_frames=12, face_bbox=face_bbox
		)

		plan = planner.plan(SOURCE_SIZE, spec)

		for transform in plan.transforms:
			x1, y1, x2, y2 = _crop_box(SOURCE_SIZE, transform)
			assert x1 <= face_bbox[0]
			assert y1 <= face_bbox[1]
			assert x2 >= face_bbox[2]
			assert y2 >= face_bbox[3]

	def test_invalid_source_size_raises(self):
		with pytest.raises(MotionPlanError):
			KenBurns2DPlanner().plan((0, 800), CameraMotionSpec())

	def test_invalid_num_frames_raises(self):
		with pytest.raises(MotionPlanError):
			KenBurns2DPlanner().plan(SOURCE_SIZE, CameraMotionSpec(num_frames=0))


class TestStaticMotionPlanner:
	def test_all_transforms_are_identity(self):
		plan = StaticMotionPlanner().plan(SOURCE_SIZE, CameraMotionSpec(num_frames=5))

		assert len(plan.transforms) == 5
		for transform in plan.transforms:
			assert transform.scale == pytest.approx(1.0)
			assert transform.translate_x == pytest.approx(0.0)
			assert transform.translate_y == pytest.approx(0.0)


class TestAffineFrameWarper:
	def test_output_size_matches_source(self):
		source = Image.new("RGB", SOURCE_SIZE, color=(10, 20, 30))
		transform = KenBurns2DPlanner().plan(SOURCE_SIZE, CameraMotionSpec(direction="right", num_frames=5)).transforms[-1]

		warped = AffineFrameWarper().warp(source, transform)

		assert warped.size == SOURCE_SIZE


class TestBuildMotionPlanner:
	@pytest.mark.parametrize("mode,expected_type", [("static", StaticMotionPlanner), ("ken_burns_2d", KenBurns2DPlanner)])
	def test_dispatches_known_modes(self, mode, expected_type):
		from app.core.config import CameraMotionConfig

		planner = build_motion_planner(CameraMotionConfig(mode=mode))

		assert isinstance(planner, expected_type)

	def test_unknown_mode_raises(self):
		from app.core.config import CameraMotionConfig

		with pytest.raises(ValueError):
			build_motion_planner(CameraMotionConfig(mode="depth_parallax"))
