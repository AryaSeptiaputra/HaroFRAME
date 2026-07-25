from __future__ import annotations

from app.core.config import GenerationConfig, IdentityConfig
from app.identity.engine import IdentityEngine
from app.identity.instantid.provider import InstantIdProvider
from app.generation.encode.video_writer import ImageioVideoEncoder
from app.generation.exceptions import NoRendererAvailableError
from app.generation.interfaces import FrameRenderer
from app.generation.lora.interfaces import LoraManager
from app.generation.lora.manager import PeftLoraManager
from app.generation.motion.factory import build_motion_planner
from app.generation.motion.warp import AffineFrameWarper
from app.generation.pipeline import GenerationPipeline
from app.generation.renderer.img2img_renderer import Img2ImgFrameRenderer
from app.generation.renderer.instantid_renderer import InstantIdFrameRenderer


def _build_frame_renderer(
	identity_engine: IdentityEngine,
	identity_config: IdentityConfig,
	generation_config: GenerationConfig,
	lora_manager: LoraManager,
) -> FrameRenderer:
	if identity_engine.face_adapter is None:
		raise NoRendererAvailableError(
			"no face adapter configured; enable identity.ipadapter or identity.instantid"
		)
	if isinstance(identity_engine.face_adapter, InstantIdProvider):
		return InstantIdFrameRenderer(
			identity_engine,
			identity_config,
			generation_config.render,
			generation_config.output,
			lora_manager,
		)
	return Img2ImgFrameRenderer(identity_engine, identity_config, generation_config.render, lora_manager)


def build_generation_pipeline(
	generation_config: GenerationConfig,
	identity_config: IdentityConfig,
	identity_engine: IdentityEngine,
) -> GenerationPipeline:
	lora_manager = PeftLoraManager(generation_config.lora, identity_config.cache_dir)
	return GenerationPipeline(
		identity_engine,
		build_motion_planner(generation_config.motion),
		AffineFrameWarper(),
		_build_frame_renderer(identity_engine, identity_config, generation_config, lora_manager),
		video_encoder=ImageioVideoEncoder(codec=generation_config.output.codec),
	)
