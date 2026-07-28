from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from app.core.config import ControlNetConfig, GarmentSwapConfig, IdentityConfig, RenderConfig
from app.identity.controlnet.provider.pose_dwpose import DwPoseConditioner
from app.identity.engine import IdentityEngine
from app.identity.exceptions import ModelLoadError
from app.identity.interfaces import IdentityReference
from app.identity.sdxl_pipeline_loader import load_sdxl_pipeline
from app.identity.segmentation.interfaces import GarmentMaskGenerator
from app.generation.interfaces import RenderedFrame
from app.generation.lora.interfaces import LoraManager

_TORCH_DTYPES = {
	"fp16": torch.float16,
	"bf16": torch.bfloat16,
	"fp32": torch.float32,
}


class GarmentSwapFrameRenderer:
	"""Single-image (no motion) renderer for garment-swap: a SAM-derived mask
	around the garment-bearing body region, inpainted via
	StableDiffusionXLInpaintPipeline (or the ControlNet+Inpaint combo when
	garment_config.use_pose_controlnet is set, for better anatomical alignment
	in newly-generated limb regions).

	IP-Adapter/FaceID-SDXL branch only -- InstantID rejection happens in
	app.generation.factory.build_garment_renderer(), not here (this class trusts
	it was only constructed for a compatible face adapter).

	The pose ControlNet used here is intentionally independent of
	IdentityConfig.controlnet.pose_enabled (Feature A's img2img structure-
	conditioning toggle) -- the two serve different purposes (ordinary img2img
	structure guidance vs. anatomical guidance for inpainted limb regions) and
	coupling them would force one to be silently enabled to get the other.
	"""

	def __init__(
		self,
		identity_engine: IdentityEngine,
		identity_config: IdentityConfig,
		garment_config: GarmentSwapConfig,
		render_config: RenderConfig,
		mask_generator: GarmentMaskGenerator,
		lora_manager: LoraManager | None = None,
	) -> None:
		self._identity_engine = identity_engine
		self._identity_config = identity_config
		self._garment_config = garment_config
		self._render_config = render_config
		self._mask_generator = mask_generator
		self._lora_manager = lora_manager
		self._pipeline: Any = None
		self._pose_conditioner = (
			DwPoseConditioner(
				ControlNetConfig(
					pose_repo_id=garment_config.pose_repo_id,
					pose_conditioning_scale=garment_config.pose_conditioning_scale,
				)
			)
			if garment_config.use_pose_controlnet
			else None
		)

	def _ensure_pipeline(self) -> Any:
		if self._pipeline is not None:
			return self._pipeline
		try:
			if self._pose_conditioner is not None:
				from diffusers import StableDiffusionXLControlNetInpaintPipeline as PipelineCls
			else:
				from diffusers import StableDiffusionXLInpaintPipeline as PipelineCls
		except ImportError as exc:
			raise ModelLoadError(
				"diffusers is not installed; cannot build the garment-swap inpaint pipeline"
			) from exc

		dtype = _TORCH_DTYPES[self._identity_config.dtype]
		hf_token = self._identity_config.hf_token.get_secret_value() if self._identity_config.hf_token else None
		extra_kwargs: dict[str, Any] = {}
		if self._pose_conditioner is not None:
			extra_kwargs["controlnet"] = self._pose_conditioner.ensure_controlnet()
		pipeline = load_sdxl_pipeline(
			PipelineCls,
			self._identity_config.base_sdxl_model,
			torch_dtype=dtype,
			cache_dir=self._identity_config.cache_dir,
			token=hf_token,
			**extra_kwargs,
		)
		pipeline.to(self._identity_config.device, dtype)
		self._identity_engine.load(pipeline)
		if self._lora_manager is not None:
			self._lora_manager.load(pipeline)
		self._pipeline = pipeline
		return pipeline

	def render_garment_swap(
		self,
		source_image: Image.Image,
		*,
		reference: IdentityReference,
		garment_prompt: str,
		negative_prompt: str,
		seed: int,
		strength: float | None = None,
	) -> RenderedFrame:
		pipeline = self._ensure_pipeline()
		garment_mask = self._mask_generator.generate_mask(source_image)
		conditioning = self._identity_engine.build_conditioning(reference)
		generator = torch.Generator(device=self._identity_config.device).manual_seed(seed)

		call_kwargs: dict[str, Any] = dict(
			prompt=garment_prompt,
			negative_prompt=negative_prompt or self._garment_config.negative_prompt,
			image=source_image,
			mask_image=garment_mask.mask,
			strength=self._garment_config.inpaint_strength if strength is None else strength,
			guidance_scale=self._garment_config.guidance_scale,
			num_inference_steps=self._garment_config.num_inference_steps,
			generator=generator,
			**conditioning.adapter_kwargs,
		)
		if self._pose_conditioner is not None:
			call_kwargs["control_image"] = self._pose_conditioner.preprocess(source_image)
			call_kwargs["controlnet_conditioning_scale"] = self._garment_config.pose_conditioning_scale

		result = pipeline(**call_kwargs)
		return RenderedFrame(image=result.images[0], frame_index=0, seed=seed, face_detected=True)
