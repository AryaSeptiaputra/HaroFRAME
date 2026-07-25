from __future__ import annotations

from app.core.config import CameraMotionConfig
from app.generation.interfaces import CameraMotionPlanner, FrameWarper
from app.generation.motion.depth_parallax import DepthParallaxPlanner
from app.generation.motion.ken_burns import KenBurns2DPlanner
from app.generation.motion.parallax_warp import DepthParallaxWarper
from app.generation.motion.static import StaticMotionPlanner
from app.generation.motion.warp import AffineFrameWarper

_PLANNERS: dict[str, type] = {
	"static": StaticMotionPlanner,
	"ken_burns_2d": KenBurns2DPlanner,
	"depth_parallax": DepthParallaxPlanner,
}


def build_motion_planner(config: CameraMotionConfig) -> CameraMotionPlanner:
	try:
		planner_cls = _PLANNERS[config.mode]
	except KeyError as exc:
		raise ValueError(f"unknown motion mode: {config.mode!r}") from exc
	return planner_cls()


def build_frame_warper(config: CameraMotionConfig) -> FrameWarper:
	if config.mode == "depth_parallax":
		return DepthParallaxWarper()
	return AffineFrameWarper()
