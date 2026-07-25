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
	scale: float = 0.6


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
	seed: int | None = None


class IdentityConfig(BaseModel):
	device: Literal["cuda", "cpu", "mps"] = "cuda"
	dtype: Literal["fp16", "bf16", "fp32"] = "fp16"
	cache_dir: Path = Path(".cache/models")
	hf_token: SecretStr | None = None
	base_sdxl_model: str = "stabilityai/stable-diffusion-xl-base-1.0"

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
