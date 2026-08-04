from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from PIL import Image

from app.identity.interfaces import IdentityReference


@dataclass(slots=True, frozen=True)
class FrameTransform:
	"""Resolved per-frame camera transform: how far to crop/pan/zoom into the source image."""

	scale: float
	translate_x: float
	translate_y: float
	frame_index: int


@dataclass(slots=True)
class MotionPlan:
	"""The resolved output of a CameraMotionPlanner: one FrameTransform per output frame."""

	transforms: list[FrameTransform]
	num_frames: int
	fps: int


@dataclass(slots=True, frozen=True)
class CameraMotionSpec:
	"""User/config-facing motion request, deliberately separate from the resolved MotionPlan.

	``face_bbox`` (if known) lets a planner keep the face inside frame across the
	whole clip; it is optional because a planner may run before face analysis.
	"""

	mode: str = "ken_burns_2d"
	direction: str = "auto"
	zoom_range: tuple[float, float] = (1.0, 1.15)
	pan_fraction: tuple[float, float] = (0.0, 0.08)
	easing: str = "ease_in_out"
	num_frames: int = 32
	fps: int = 8
	face_bbox: tuple[float, float, float, float] | None = None


@dataclass(slots=True)
class RenderedFrame:
	"""One rendered output frame plus enough metadata to debug identity/consistency issues."""

	image: Image.Image
	frame_index: int
	seed: int
	face_detected: bool = True


@dataclass(slots=True)
class GenerationRequest:
	"""Top-level request bracketing a GenerationPipeline.generate() call.

	``inpaint_prompt`` describes what the stage-1 SourceEditor should put in the
	masked garment/body region ("a red hoodie", "bare arms"); it is separate from
	``prompt``, which drives the stage-2 pose/style render. ``None`` falls back to
	InpaintConfig.prompt, and is ignored outright when no SourceEditor is wired.
	"""

	reference: IdentityReference
	prompt: str
	negative_prompt: str = ""
	motion: CameraMotionSpec = field(default_factory=CameraMotionSpec)
	seed: int | None = None
	inpaint_prompt: str | None = None


@dataclass(slots=True)
class GenerationResult:
	"""Top-level result of a GenerationPipeline.generate() call."""

	frames: list[RenderedFrame]
	fps: int
	output_path: Path | None = None


class CameraMotionPlanner(Protocol):
	"""Turns a CameraMotionSpec into a concrete, frame-indexed MotionPlan."""

	def plan(self, source_size: tuple[int, int], spec: CameraMotionSpec) -> MotionPlan:
		"""Return one FrameTransform per frame, given the source image's (width, height)."""
		...


class FrameWarper(Protocol):
	"""Applies a single FrameTransform to the source image, producing one warped frame."""

	def warp(self, source_image: Image.Image, transform: FrameTransform) -> Image.Image:
		...


class SourceEditor(Protocol):
	"""Stage-1 edit of the source photo, applied **once** before any motion
	planning or frame rendering happens.

	This is where garment swaps and newly-generated body regions live: an
	inpaint pass rewrites part of the photo, and everything downstream (warp +
	img2img/InstantID render, per frame) then treats the *edited* photo as its
	source. Running once rather than per frame is both far cheaper and the only
	way the edit stays consistent across a clip.

	Returns an image the same size as ``source_image`` -- callers rely on that
	to keep an already-computed face bbox and motion plan valid.
	"""

	def edit(
		self,
		source_image: Image.Image,
		*,
		prompt: str | None = None,
		negative_prompt: str = "",
		seed: int,
		strength: float | None = None,
	) -> Image.Image:
		...


class FrameRenderer(Protocol):
	"""Re-renders one warped frame through the identity-preserving SDXL stack.

	Implementations differ by which face adapter IdentityEngine is holding (see
	app/generation/renderer/) -- ``strength`` is accepted-and-ignored by
	implementations that can't do img2img (e.g. InstantID's vendor pipeline).
	"""

	def render(
		self,
		warped_frame: Image.Image,
		*,
		reference: IdentityReference,
		prompt: str,
		negative_prompt: str,
		seed: int,
		frame_index: int,
		strength: float | None = None,
	) -> RenderedFrame:
		...


class TemporalSmoother(Protocol):
	"""Post-processes a full frame sequence to reduce flicker between frames."""

	def smooth(self, frames: list[Image.Image]) -> list[Image.Image]:
		...


class VideoEncoder(Protocol):
	"""Muxes a frame sequence into a video file on disk."""

	def encode(self, frames: list[Image.Image], fps: int, output_path: Path) -> Path:
		...
