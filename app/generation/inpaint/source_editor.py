from __future__ import annotations

import gc
from typing import Any

import torch
from PIL import Image

from app.core.config import ControlNetConfig, IdentityConfig, InpaintConfig
from app.identity.controlnet.provider.pose_dwpose import DwPoseConditioner
from app.identity.exceptions import ModelLoadError
from app.identity.sdxl_pipeline_loader import load_sdxl_pipeline
from app.identity.segmentation.interfaces import GarmentMaskGenerator
from app.generation.exceptions import SourceEditError
from app.generation.lora.interfaces import LoraManager
from app.generation.resolution import sdxl_working_size

_TORCH_DTYPES = {
	"fp16": torch.float16,
	"bf16": torch.bfloat16,
	"fp32": torch.float32,
}


class InpaintSourceEditor:
	"""SourceEditor that rewrites the garment/body region of the source photo
	with StableDiffusionXLInpaintPipeline (or the ControlNet+Inpaint combo when
	``config.use_pose_controlnet`` is set, for better anatomy in newly-generated
	limb regions), using a SAM-derived mask.

	**Runs once per generation, not per frame**, and hands its output back as the
	new source photo -- pose, motion and style are then produced entirely by the
	ordinary i2i/i2v stage downstream. That split is what makes this work with
	*both* face-adapter branches: nothing about this stage touches the face
	adapter, so InstantID (whose vendored pipeline has no inpaint entry point)
	is no longer excluded from garment/body editing.

	**No identity conditioning is applied here on purpose.** The mask covers
	clothing and limbs, never the face -- face pixels pass through untouched --
	so a face embedding would contribute nothing except pulling the newly
	generated garment back toward the reference photo's original outfit, which
	is the opposite of the intent. Identity preservation is stage 2's job.
	"""

	def __init__(
		self,
		identity_config: IdentityConfig,
		config: InpaintConfig,
		mask_generator: GarmentMaskGenerator,
		lora_manager: LoraManager | None = None,
	) -> None:
		self._identity_config = identity_config
		self._config = config
		self._mask_generator = mask_generator
		self._lora_manager = lora_manager
		self._pipeline: Any = None
		self._pose_conditioner = (
			DwPoseConditioner(
				ControlNetConfig(
					pose_repo_id=config.pose_repo_id,
					pose_conditioning_scale=config.pose_conditioning_scale,
				)
			)
			if config.use_pose_controlnet
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
				"diffusers is not installed; cannot build the inpaint source-editing pipeline"
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
		if self._lora_manager is not None:
			self._lora_manager.load(pipeline)
		self._pipeline = pipeline
		return pipeline

	def edit(
		self,
		source_image: Image.Image,
		*,
		prompt: str | None = None,
		negative_prompt: str = "",
		seed: int,
		strength: float | None = None,
	) -> Image.Image:
		effective_prompt = prompt if prompt else self._config.prompt
		if not effective_prompt:
			raise SourceEditError(
				"inpainting is enabled but no prompt describes what the masked garment/body "
				"region should become; set generation.inpaint.prompt or pass one per request"
			)

		# Mask first: an uninstalled segment-anything or a missing SAM checkpoint
		# should fail before paying for a multi-GB inpaint pipeline load.
		garment_mask = self._mask_generator.generate_mask(source_image)
		pipeline = self._ensure_pipeline()
		generator = torch.Generator(device=self._identity_config.device).manual_seed(seed)

		work_width, work_height = sdxl_working_size(source_image.size)
		call_kwargs: dict[str, Any] = dict(
			prompt=effective_prompt,
			negative_prompt=negative_prompt or self._config.negative_prompt,
			image=source_image,
			mask_image=garment_mask.mask,
			# Explicit, so the init image, the mask and the pose control image are
			# all resized to the same grid -- see sdxl_working_size().
			height=work_height,
			width=work_width,
			strength=self._config.strength if strength is None else strength,
			guidance_scale=self._config.guidance_scale,
			num_inference_steps=self._config.num_inference_steps,
			generator=generator,
		)
		if self._pose_conditioner is not None:
			call_kwargs["control_image"] = self._pose_conditioner.preprocess(source_image)
			call_kwargs["controlnet_conditioning_scale"] = self._config.pose_conditioning_scale

		edited = pipeline(**call_kwargs).images[0]
		# Stage 1 renders at the SDXL-sized grid above, so for a large photo this
		# scales back up. Restoring the exact source size is not cosmetic: the
		# caller's face bbox (pixel coords from the original) and motion plan both
		# assume it.
		if edited.size != source_image.size:
			edited = edited.resize(source_image.size, Image.LANCZOS)
		return edited

	def release(self) -> None:
		"""Drop everything stage 1 holds on the GPU.

		Stage 1 finishes entirely before stage 2 begins, so its inpaint pipeline
		(~7GB), the pose ControlNet (~2.5GB) and SAM stay resident for no reason
		while the renderer tries to allocate. On a 24GB card that is the
		difference between running and torch.OutOfMemoryError.

		Callers that run one generation then move on should call this; the cost is
		that a *batch* reloads these weights per item, which is why it is not done
		inside edit() itself.
		"""
		self._pipeline = None
		if self._pose_conditioner is not None:
			self._pose_conditioner.release()
		release = getattr(self._mask_generator, "release", None)
		if callable(release):
			release()
		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
