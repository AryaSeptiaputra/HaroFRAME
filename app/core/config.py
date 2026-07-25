from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr
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


@lru_cache
def get_settings() -> Settings:
	return Settings()
