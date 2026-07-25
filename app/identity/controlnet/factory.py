from __future__ import annotations

from app.core.config import ControlNetConfig
from app.identity.controlnet.interfaces import StructureConditioner
from app.identity.controlnet.provider.depth import DepthConditioner
from app.identity.controlnet.provider.pose_dwpose import DwPoseConditioner


def build_structure_conditioners(config: ControlNetConfig) -> list[StructureConditioner]:
	"""Build 0, 1, or 2 structure conditioners (pose and/or depth) per config."""
	conditioners: list[StructureConditioner] = []
	if config.pose_enabled:
		conditioners.append(DwPoseConditioner(config))
	if config.depth_enabled:
		conditioners.append(DepthConditioner(config))
	return conditioners
