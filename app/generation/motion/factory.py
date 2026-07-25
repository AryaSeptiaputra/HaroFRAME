from __future__ import annotations

from app.core.config import CameraMotionConfig
from app.generation.interfaces import CameraMotionPlanner
from app.generation.motion.ken_burns import KenBurns2DPlanner
from app.generation.motion.static import StaticMotionPlanner

_PLANNERS: dict[str, type] = {
	"static": StaticMotionPlanner,
	"ken_burns_2d": KenBurns2DPlanner,
}


def build_motion_planner(config: CameraMotionConfig) -> CameraMotionPlanner:
	try:
		planner_cls = _PLANNERS[config.mode]
	except KeyError as exc:
		raise ValueError(
			f"unknown or not-yet-implemented motion mode: {config.mode!r}"
		) from exc
	return planner_cls()
