from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path
from typing import Callable

from app.identity.engine import IdentityEngine
from app.generation.exceptions import NoRendererAvailableError
from app.generation.interfaces import (
	CameraMotionPlanner,
	FrameRenderer,
	FrameWarper,
	GenerationRequest,
	GenerationResult,
	SourceEditor,
	TemporalSmoother,
	VideoEncoder,
)

_MAX_SEED = 2**31 - 1


class GenerationPipeline:
	"""Orchestrates one generate() call: optional stage-1 source edit (inpaint)
	-> motion planning -> per-frame warp+render -> optional temporal smoothing
	-> optional video encoding.

	Analogous to IdentityEngine: holds one instance of each stage and never
	branches on adapter type itself -- that dispatch lives entirely in
	app/generation/factory.py, at construction time.
	"""

	def __init__(
		self,
		identity_engine: IdentityEngine,
		motion_planner: CameraMotionPlanner,
		frame_warper: FrameWarper,
		frame_renderer: FrameRenderer,
		*,
		temporal_smoother: TemporalSmoother | None = None,
		video_encoder: VideoEncoder | None = None,
		source_editor: SourceEditor | None = None,
	) -> None:
		self._identity_engine = identity_engine
		self._motion_planner = motion_planner
		self._frame_warper = frame_warper
		self._frame_renderer = frame_renderer
		self._temporal_smoother = temporal_smoother
		self._video_encoder = video_encoder
		self._source_editor = source_editor

	def generate(
		self,
		request: GenerationRequest,
		*,
		output_path: Path | None = None,
		progress_callback: Callable[[int, int], None] | None = None,
	) -> GenerationResult:
		"""``progress_callback(frames_done, total_frames)`` is invoked after each
		frame renders -- e.g. for a queue/job-status UI (scripts/interactive_generate.py)
		to report per-frame progress on a long-running video generation."""
		if self._identity_engine.face_adapter is None:
			raise NoRendererAvailableError(
				"no face adapter configured; enable identity.ipadapter or identity.instantid"
			)

		reference = request.reference
		if reference.fused_embedding is None:
			self._identity_engine.prepare_reference(reference)

		seed = request.seed if request.seed is not None else random.randint(0, _MAX_SEED)

		# Stage 1, once for the whole clip: rewrite the garment/body region of the
		# source photo. Deliberately *after* prepare_reference() -- identity stays
		# locked to the original, unedited photo -- and reference.images is left
		# untouched so the face adapter keeps conditioning on it. The editor
		# preserves image size, and the mask never covers the face, so the fused
		# embedding's bbox stays valid against the edited photo below.
		source_image = reference.images[0]
		if self._source_editor is not None:
			source_image = self._source_editor.edit(
				source_image,
				prompt=request.inpaint_prompt,
				negative_prompt=request.negative_prompt,
				seed=seed,
			)
			# Hand stage 1's VRAM back before the renderer asks for its own -- the
			# two never overlap, and holding both exhausts a 24GB card.
			release = getattr(self._source_editor, "release", None)
			if callable(release):
				release()

		motion_spec = request.motion
		if motion_spec.face_bbox is None and reference.fused_embedding is not None:
			motion_spec = replace(motion_spec, face_bbox=reference.fused_embedding.bbox)

		plan = self._motion_planner.plan(source_image.size, motion_spec)

		rendered_frames = []
		total_frames = len(plan.transforms)
		for transform in plan.transforms:
			warped = self._frame_warper.warp(source_image, transform)
			rendered_frames.append(
				self._frame_renderer.render(
					warped,
					reference=reference,
					prompt=request.prompt,
					negative_prompt=request.negative_prompt,
					seed=seed,
					frame_index=transform.frame_index,
				)
			)
			if progress_callback is not None:
				progress_callback(len(rendered_frames), total_frames)

		if self._temporal_smoother is not None:
			smoothed = self._temporal_smoother.smooth([frame.image for frame in rendered_frames])
			for frame, image in zip(rendered_frames, smoothed):
				frame.image = image

		result = GenerationResult(frames=rendered_frames, fps=plan.fps)
		if output_path is not None and self._video_encoder is not None:
			result.output_path = self._video_encoder.encode(
				[frame.image for frame in rendered_frames], plan.fps, output_path
			)
		return result
