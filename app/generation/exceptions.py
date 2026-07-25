from __future__ import annotations


class GenerationModuleError(Exception):
	"""Base class for all errors raised by the generation module."""


class NoRendererAvailableError(GenerationModuleError):
	"""Raised when a GenerationPipeline is built without an active face adapter."""


class MotionPlanError(GenerationModuleError):
	"""Raised when a CameraMotionPlanner cannot produce a valid MotionPlan."""


class FrameRenderError(GenerationModuleError):
	"""Raised when a FrameRenderer fails to render a frame."""


class VideoEncodeError(GenerationModuleError):
	"""Raised when a VideoEncoder fails to mux frames into an output file."""
