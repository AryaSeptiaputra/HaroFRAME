from __future__ import annotations

from app.generation.exceptions import MotionPlanError
from app.generation.interfaces import CameraMotionSpec, FrameTransform, MotionPlan


class StaticMotionPlanner:
	"""Trivial planner for CameraMotionConfig.mode == "static": no pan, no zoom.

	Exists so "static" is a real, usable config value rather than a dead end --
	useful as a baseline for A/B'ing against the Ken Burns planners, or for
	requests that only want LoRA/prompt variation across otherwise-identical frames.
	"""

	def plan(self, source_size: tuple[int, int], spec: CameraMotionSpec) -> MotionPlan:
		width, height = source_size
		if width <= 0 or height <= 0:
			raise MotionPlanError(f"invalid source_size: {source_size!r}")
		if spec.num_frames < 1:
			raise MotionPlanError(f"num_frames must be >= 1, got {spec.num_frames}")

		transforms = [
			FrameTransform(scale=1.0, translate_x=0.0, translate_y=0.0, frame_index=index)
			for index in range(spec.num_frames)
		]
		return MotionPlan(transforms=transforms, num_frames=spec.num_frames, fps=spec.fps)
