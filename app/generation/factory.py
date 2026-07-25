from __future__ import annotations

from app.core.config import GenerationConfig, IdentityConfig
from app.identity.engine import IdentityEngine
from app.generation.encode.video_writer import ImageioVideoEncoder
from app.generation.exceptions import NoRendererAvailableError
from app.generation.interfaces import FrameRenderer
from app.generation.motion.factory import build_motion_planner
from app.generation.motion.warp import AffineFrameWarper
from app.generation.pipeline import GenerationPipeline
from app.generation.renderer.img2img_renderer import Img2ImgFrameRenderer


def _build_frame_renderer(
	identity_engine: IdentityEngine,
	identity_config: IdentityConfig,
	generation_config: GenerationConfig,
) -> FrameRenderer:
	if identity_engine.face_adapter is None:
		raise NoRendererAvailableError(
			"no face adapter configured; enable identity.ipadapter or identity.instantid"
		)
	# InstantID branch (app/generation/renderer/instantid_renderer.py) is dispatched
	# here too once it exists; for now every configured face adapter renders via img2img.
	return Img2ImgFrameRenderer(identity_engine, identity_config, generation_config.render)


def build_generation_pipeline(
	generation_config: GenerationConfig,
	identity_config: IdentityConfig,
	identity_engine: IdentityEngine,
) -> GenerationPipeline:
	return GenerationPipeline(
		identity_engine,
		build_motion_planner(generation_config.motion),
		AffineFrameWarper(),
		_build_frame_renderer(identity_engine, identity_config, generation_config),
		video_encoder=ImageioVideoEncoder(codec=generation_config.output.codec),
	)
