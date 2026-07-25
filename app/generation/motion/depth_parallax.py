from __future__ import annotations

from app.generation.motion.ken_burns import KenBurns2DPlanner


class DepthParallaxPlanner(KenBurns2DPlanner):
	"""Camera trajectory planner for ``CameraMotionConfig.mode == "depth_parallax"``.

	Produces the exact same face-aware pan/zoom FrameTransform curve as
	KenBurns2DPlanner -- depth_parallax only changes how a frame is *warped*
	from that trajectory (see DepthParallaxWarper), not how the trajectory
	itself is planned, so there is nothing to override here.
	"""
