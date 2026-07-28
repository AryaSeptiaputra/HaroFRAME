from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from app.core.config import IdentityConfig, RenderConfig
from app.identity.engine import IdentityEngine
from app.identity.exceptions import ModelLoadError
from app.identity.interfaces import IdentityReference, StructureHint
from app.identity.sdxl_pipeline_loader import load_sdxl_pipeline
from app.generation.interfaces import RenderedFrame
from app.generation.lora.interfaces import LoraManager

_TORCH_DTYPES = {
	"fp16": torch.float16,
	"bf16": torch.bfloat16,
	"fp32": torch.float32,
}


class Img2ImgFrameRenderer:
	"""FrameRenderer for the IP-Adapter-family branch (FaceID-SDXL or plain CLIP).

	Builds a plain StableDiffusionXLImg2ImgPipeline once (lazily, on first
	render), attaches the identity engine's face adapter to it, then re-renders
	each already-warped frame with the warped frame as the img2img init image --
	fixing warp artifacts while keeping the seed fixed across frames for
	temporal continuity.
	"""

	def __init__(
		self,
		identity_engine: IdentityEngine,
		identity_config: IdentityConfig,
		render_config: RenderConfig,
		lora_manager: LoraManager | None = None,
	) -> None:
		self._identity_engine = identity_engine
		self._identity_config = identity_config
		self._render_config = render_config
		self._lora_manager = lora_manager
		self._pipeline: Any = None

	def _ensure_pipeline(self) -> Any:
		if self._pipeline is not None:
			return self._pipeline
		try:
			from diffusers import StableDiffusionXLImg2ImgPipeline
		except ImportError as exc:
			raise ModelLoadError(
				"diffusers is not installed; cannot build the img2img rendering pipeline"
			) from exc

		dtype = _TORCH_DTYPES[self._identity_config.dtype]
		hf_token = self._identity_config.hf_token.get_secret_value() if self._identity_config.hf_token else None
		pipeline = load_sdxl_pipeline(
			StableDiffusionXLImg2ImgPipeline,
			self._identity_config.base_sdxl_model,
			torch_dtype=dtype,
			cache_dir=self._identity_config.cache_dir,
			token=hf_token,
		)
		pipeline.to(self._identity_config.device, dtype)
		self._identity_engine.load(pipeline)
		if self._lora_manager is not None:
			self._lora_manager.load(pipeline)
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
		controlnet_cfg = self._identity_config.controlnet
		structure = None
		if controlnet_cfg.pose_enabled or controlnet_cfg.depth_enabled:
			structure = StructureHint(
				pose_image=warped_frame if controlnet_cfg.pose_enabled else None,
				depth_image=warped_frame if controlnet_cfg.depth_enabled else None,
				source="driving_frame",
			)
		conditioning = self._identity_engine.build_conditioning(reference, structure=structure)
		generator = torch.Generator(device=self._identity_config.device).manual_seed(seed)

		result = pipeline(
			prompt=prompt,
			negative_prompt=negative_prompt or self._render_config.negative_prompt,
			image=warped_frame,
			strength=self._render_config.strength if strength is None else strength,
			guidance_scale=self._render_config.guidance_scale,
			num_inference_steps=self._render_config.num_inference_steps,
			generator=generator,
			**conditioning.adapter_kwargs,
		)
		return RenderedFrame(image=result.images[0], frame_index=frame_index, seed=seed, face_detected=True)
