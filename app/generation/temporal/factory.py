from __future__ import annotations

from app.core.config import TemporalConfig
from app.generation.interfaces import TemporalSmoother
from app.generation.temporal.ema_blend import EmaFrameSmoother
from app.generation.temporal.passthrough import NullTemporalSmoother


def build_temporal_smoother(config: TemporalConfig) -> TemporalSmoother:
	if config.method == "ema":
		return EmaFrameSmoother(smoothing_strength=config.smoothing_strength)
	return NullTemporalSmoother()
