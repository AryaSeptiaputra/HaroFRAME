from __future__ import annotations

import dataclasses
from typing import Any

import torch
from PIL import Image

from app.core.config import IdentityConfig, OutputConfig, RenderConfig
from app.identity.engine import IdentityEngine
from app.identity.face.analyzer_insightface import InsightFaceAnalyzer
from app.identity.instantid.pipeline import build_instantid_pipeline
from app.identity.interfaces import IdentityReference, StructureHint
from app.generation.interfaces import RenderedFrame


class InstantIdFrameRenderer:
	"""FrameRenderer for the InstantID branch.

	The vendored InstantID pipeline has no img2img/`strength` entry point, so
	each frame is a fresh txt2img+kps-ControlNet render rather than a partial
	denoise of the warped frame. Temporal consistency instead comes from a
	fixed seed plus re-detecting facial landmarks directly on the warped frame
	each time -- not transforming the reference photo's landmarks through the
	warp matrix, which would only stay correct for pure-affine warps and drift
	silently otherwise. The identity embedding itself always stays locked to
	the original reference photo; only the landmark *layout* comes from the
	warped frame.
	"""

	def __init__(
		self,
		identity_engine: IdentityEngine,
		identity_config: IdentityConfig,
		render_config: RenderConfig,
		output_config: OutputConfig,
	) -> None:
		self._identity_engine = identity_engine
		self._identity_config = identity_config
		self._render_config = render_config
		self._output_config = output_config
		self._pipeline: Any = None
		self._face_analyzer = InsightFaceAnalyzer(identity_config.face)

	def _ensure_pipeline(self) -> Any:
		if self._pipeline is not None:
			return self._pipeline
		pipeline = build_instantid_pipeline(self._identity_config)
		self._identity_engine.load(pipeline)
		self._pipeline = pipeline
		return pipeline

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
		pipeline = self._ensure_pipeline()

		detected_faces = self._face_analyzer.analyze(warped_frame)
		face_detected = bool(detected_faces)
		base_embedding = reference.fused_embedding
		if face_detected:
			best = max(detected_faces, key=lambda face: face.det_score)
			frame_embedding = dataclasses.replace(
				base_embedding, landmarks_5pt=best.landmarks_5pt, bbox=best.bbox
			)
		else:
			# Graceful degradation: fall back to the reference photo's own
			# landmark layout rather than failing the whole frame.
			frame_embedding = base_embedding

		frame_reference = IdentityReference(
			images=reference.images,
			embeddings=reference.embeddings,
			fused_embedding=frame_embedding,
		)
		conditioning = self._identity_engine.build_conditioning(
			frame_reference,
			structure=StructureHint(pose_image=warped_frame, source="driving_frame"),
			scale=self._identity_config.instantid.controlnet_conditioning_scale,
		)

		generator = torch.Generator(device=self._identity_config.device).manual_seed(seed)
		result = pipeline(
			prompt=prompt,
			negative_prompt=negative_prompt or self._render_config.negative_prompt,
			num_inference_steps=self._render_config.num_inference_steps,
			guidance_scale=self._render_config.guidance_scale,
			height=self._output_config.height,
			width=self._output_config.width,
			generator=generator,
			**conditioning.adapter_kwargs,
		)
		return RenderedFrame(
			image=result.images[0], frame_index=frame_index, seed=seed, face_detected=face_detected
		)
