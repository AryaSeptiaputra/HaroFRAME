from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

FusionStrategyName = Literal["mean", "weighted_by_det_score", "best_quality"]


class FaceConfig(BaseModel):
	model_pack: str = "buffalo_l"
	det_size: tuple[int, int] = (640, 640)
	min_det_score: float = 0.5
	fusion_strategy: FusionStrategyName = "mean"


class IpAdapterConfig(BaseModel):
	enabled: bool = False
	variant: Literal["clip", "faceid_sdxl"] = "faceid_sdxl"
	repo_id: str = "h94/IP-Adapter-FaceID"
	subfolder: str | None = None
	weight_name: str = "ip-adapter-faceid_sdxl.bin"
	lora_weight_name: str | None = "ip-adapter-faceid_sdxl_lora.safetensors"
	# Cross-attention strength of the adapter itself.
	scale: float = 0.6
	# Weight of the FaceID *companion LoRA*, which is a separate lever from
	# ``scale`` and used to be pinned at diffusers' 1.0 default with no way to
	# lower it. At 1.0 on top of an already-merged photoreal checkpoint
	# (RealVisXL, Juggernaut) it tends to break skin into high-frequency coloured
	# speckle. Lower it before lowering ``scale`` if the texture looks wrong;
	# raise it toward 1.0 if identity is too weak.
	lora_scale: float = 0.6


class InstantIdConfig(BaseModel):
	enabled: bool = False
	controlnet_repo_id: str = "InstantX/InstantID"
	controlnet_subfolder: str = "ControlNetModel"
	ip_adapter_weight_name: str = "ip-adapter.bin"
	controlnet_conditioning_scale: float = 0.8
	ip_adapter_scale: float = 0.8


class ControlNetConfig(BaseModel):
	pose_enabled: bool = False
	pose_repo_id: str = "thibaud/controlnet-openpose-sdxl-1.0"
	pose_conditioning_scale: float = 0.5
	depth_enabled: bool = False
	depth_repo_id: str = "diffusers/controlnet-depth-sdxl-1.0"
	depth_conditioning_scale: float = 0.5
	depth_estimator: Literal["midas", "zoe"] = "midas"


class RestorationConfig(BaseModel):
	enabled: bool = False
	backend: Literal["gfpgan"] = "gfpgan"
	model_path: Path = Path(".cache/models/gfpgan/GFPGANv1.4.pth")
	upscale: int = 1


SAM_CHECKPOINT_FILENAMES = {
	"vit_b": "sam_vit_b_01ec64.pth",
	"vit_l": "sam_vit_l_0b3195.pth",
	"vit_h": "sam_vit_h_4b8939.pth",
}


class SamConfig(BaseModel):
	"""SAM model used to derive the stage-1 garment/body mask.

	Bigger backbones give better masks at more disk/VRAM: vit_b 375MB, vit_l
	1.25GB, vit_h 2.56GB. scripts/prefetch_models.py downloads whichever is
	configured during instance setup.
	"""

	model_type: Literal["vit_h", "vit_l", "vit_b"] = "vit_b"
	checkpoint_path: Path = Path(".cache/models/sam") / SAM_CHECKPOINT_FILENAMES["vit_b"]

	@model_validator(mode="after")
	def _default_checkpoint_path_tracks_model_type(self) -> "SamConfig":
		"""Keep the default filename in step with ``model_type``.

		Without this, setting only SAM__MODEL_TYPE=vit_l leaves the path pointing
		at the vit_b filename -- so a machine that already downloaded vit_b skips
		the download and then feeds vit_b weights to a vit_l architecture, which
		fails deep inside SAM with nothing pointing back at the config. An
		explicitly-set path always wins.
		"""
		if "checkpoint_path" not in self.model_fields_set:
			self.checkpoint_path = Path(".cache/models/sam") / SAM_CHECKPOINT_FILENAMES[self.model_type]
		return self


class InpaintConfig(BaseModel):
	"""Stage 1 of the unified pipeline: edit the source photo's garment/body
	region via SDXL inpainting, *before* the i2i/i2v stage renders pose+style
	from the edited photo (see app/generation/inpaint/).

	``enabled`` gates the whole stage and defaults to **on**: a generation is two
	prompts by default, one for what the masked region becomes and one for the
	stage-2 render. Turning it off (``--no-inpaint`` on the scripts, or
	``HAROFRAME_GENERATION__INPAINT__ENABLED=false``) drops back to the
	single-stage i2i/i2v behaviour. Because it is on by default, the ``garment``
	extra is part of the default install (Dockerfile/entrypoint.sh) -- but the SAM
	checkpoint itself is still a separate manual download, see VAST_GUIDE.md.

	``prompt`` describes what the masked region should become ("sleeveless summer
	top", "bare arms"). It has no default worth guessing, so an enabled stage with
	an empty prompt is a configuration error the entry points catch up front (see
	``inpaint_prompt_missing``) rather than a silent no-op; callers may also
	override it per request.

	``use_pose_controlnet``/``pose_conditioning_scale``/``pose_repo_id`` are
	deliberately independent of IdentityConfig.controlnet's pose settings: this
	one guides the anatomy of limb regions *being generated* here in stage 1,
	that one is structure conditioning for the stage-2 img2img render. Coupling
	them would force one to be silently enabled to get the other.
	"""

	enabled: bool = True
	prompt: str = ""
	sam: SamConfig = Field(default_factory=SamConfig)
	mask_dilation_px: int = 40
	mask_feather_px: int = 8
	mask_min_confidence: float = 0.3
	include_arms_in_mask: bool = True
	include_legs_in_mask: bool = False
	strength: float = 0.85
	guidance_scale: float = 6.0
	num_inference_steps: int = 35
	use_pose_controlnet: bool = True
	pose_conditioning_scale: float = 0.5
	pose_repo_id: str = "thibaud/controlnet-openpose-sdxl-1.0"
	negative_prompt: str = ""


class CameraMotionConfig(BaseModel):
	mode: Literal["static", "ken_burns_2d", "depth_parallax"] = "ken_burns_2d"
	direction: Literal["auto", "left", "right", "up", "down", "in", "out"] = "auto"
	zoom_range: tuple[float, float] = (1.0, 1.15)
	pan_fraction: tuple[float, float] = (0.0, 0.08)
	easing: Literal["linear", "ease_in_out"] = "ease_in_out"


class RenderConfig(BaseModel):
	strength: float = 0.35
	guidance_scale: float = 5.0
	num_inference_steps: int = 30
	negative_prompt: str = ""
	landmark_redetect_every_n_frames: int = 1


class LoraEntryConfig(BaseModel):
	enabled: bool = True
	adapter_name: str
	source: str
	weight_name: str | None = None
	subfolder: str | None = None
	scale: float = 0.6

	@field_validator("adapter_name")
	@classmethod
	def _adapter_name_not_reserved(cls, value: str) -> str:
		if value == "faceid":
			raise ValueError(
				"adapter_name 'faceid' is reserved for the IP-Adapter-FaceID companion LoRA"
			)
		return value


class LoraConfig(BaseModel):
	"""Container for the multi-LoRA mechanism (app/generation/lora/).

	Distinct from IpAdapterConfig.lora_weight_name, which is a single companion
	LoRA loaded by FaceIdSdxlProvider itself under the reserved "faceid" adapter
	name -- this config is for additional, user-chosen style/aesthetic LoRAs
	stacked on top via PEFT multi-adapter support.
	"""

	max_active_loras: int = 3
	civitai_api_key: SecretStr | None = None
	entries: list[LoraEntryConfig] = Field(default_factory=list)

	@model_validator(mode="after")
	def _enabled_count_within_limit(self) -> "LoraConfig":
		enabled_count = sum(1 for entry in self.entries if entry.enabled)
		if enabled_count > self.max_active_loras:
			raise ValueError(
				f"{enabled_count} LoRA entries are enabled, but max_active_loras={self.max_active_loras}; "
				"disable some entries or raise max_active_loras"
			)
		return self


class TemporalConfig(BaseModel):
	method: Literal["none", "ema"] = "none"
	smoothing_strength: float = 0.5


class OutputConfig(BaseModel):
	fps: int = 8
	duration_seconds: float = 4.0
	width: int = 1024
	height: int = 1024
	format: Literal["mp4"] = "mp4"
	codec: str = "libx264"
	output_dir: Path = Path("outputs")


class GenerationConfig(BaseModel):
	motion: CameraMotionConfig = Field(default_factory=CameraMotionConfig)
	render: RenderConfig = Field(default_factory=RenderConfig)
	lora: LoraConfig = Field(default_factory=LoraConfig)
	temporal: TemporalConfig = Field(default_factory=TemporalConfig)
	output: OutputConfig = Field(default_factory=OutputConfig)
	inpaint: InpaintConfig = Field(default_factory=InpaintConfig)
	seed: int | None = None


class IdentityConfig(BaseModel):
	device: Literal["cuda", "cpu", "mps"] = "cuda"
	dtype: Literal["fp16", "bf16", "fp32"] = "fp16"
	cache_dir: Path = Path(".cache/models")
	hf_token: SecretStr | None = None
	# RealVisXL V5.0 rather than stock stabilityai/stable-diffusion-xl-base-1.0:
	# this pipeline's whole job is people -- stage 1 generates limbs and torsos
	# from scratch, stage 2 has to hold a plausible pose -- and base SDXL is
	# markedly weaker at human anatomy than the photoreal community merges.
	# Ungated, openrail++, diffusers layout with an fp16 variant (~7GB).
	# RunDiffusion/Juggernaut-XL-v9 is the closest alternative; swap via
	# HAROFRAME_IDENTITY__BASE_SDXL_MODEL, no code change needed.
	base_sdxl_model: str = "SG161222/RealVisXL_V5.0"

	face: FaceConfig = Field(default_factory=FaceConfig)
	ipadapter: IpAdapterConfig = Field(default_factory=IpAdapterConfig)
	instantid: InstantIdConfig = Field(default_factory=InstantIdConfig)
	controlnet: ControlNetConfig = Field(default_factory=ControlNetConfig)
	restoration: RestorationConfig = Field(default_factory=RestorationConfig)


class Settings(BaseSettings):
	model_config = SettingsConfigDict(
		env_prefix="HAROFRAME_",
		env_nested_delimiter="__",
		env_file=".env",
		extra="ignore",
	)

	identity: IdentityConfig = Field(default_factory=IdentityConfig)
	generation: GenerationConfig = Field(default_factory=GenerationConfig)


@lru_cache
def get_settings() -> Settings:
	return Settings()


INPAINT_PROMPT_REQUIRED_MESSAGE = (
	"inpainting is on but no prompt says what the masked garment/body region should "
	"become. Pass --inpaint-prompt, set HAROFRAME_GENERATION__INPAINT__PROMPT, or turn "
	"the stage off with --no-inpaint."
)


def apply_inpaint_overrides(
	generation: GenerationConfig, *, prompt: str | None = None, disabled: bool = False
) -> GenerationConfig:
	"""Fold an entry point's ``--inpaint-prompt``/``--no-inpaint`` flags into a
	GenerationConfig, returning the original object untouched when neither is given.

	Shared by every script under scripts/ so the two stage-1 flags mean exactly the
	same thing wherever they appear.
	"""
	updates: dict = {}
	if disabled:
		updates["enabled"] = False
	if prompt:
		updates["prompt"] = prompt
	if not updates:
		return generation
	return generation.model_copy(update={"inpaint": generation.inpaint.model_copy(update=updates)})


def inpaint_prompt_missing(generation: GenerationConfig) -> bool:
	"""True when stage 1 would run but has nothing to inpaint toward. Entry points
	check this so the run fails immediately instead of after a multi-GB model load."""
	return generation.inpaint.enabled and not generation.inpaint.prompt
