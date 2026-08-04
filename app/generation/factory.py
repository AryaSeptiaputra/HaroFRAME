from __future__ import annotations

from app.core.config import GenerationConfig, IdentityConfig
from app.identity.engine import IdentityEngine
from app.identity.instantid.provider import InstantIdProvider
from app.identity.segmentation.sam_provider import SamGarmentMaskGenerator
from app.generation.encode.video_writer import ImageioVideoEncoder
from app.generation.exceptions import NoRendererAvailableError
from app.generation.inpaint.source_editor import InpaintSourceEditor
from app.generation.interfaces import FrameRenderer, SourceEditor
from app.generation.lora.interfaces import LoraManager
from app.generation.lora.manager import PeftLoraManager
from app.generation.motion.factory import build_frame_warper, build_motion_planner
from app.generation.pipeline import GenerationPipeline
from app.generation.renderer.img2img_renderer import Img2ImgFrameRenderer
from app.generation.renderer.instantid_renderer import InstantIdFrameRenderer
from app.generation.temporal.factory import build_temporal_smoother


def build_frame_renderer(
	identity_engine: IdentityEngine,
	identity_config: IdentityConfig,
	generation_config: GenerationConfig,
) -> FrameRenderer:
	"""Build the adapter-appropriate FrameRenderer, LoRA manager included.

	Public (not just used by build_generation_pipeline) so single-image
	image2image entry points (scripts/generate_image.py) can render one frame
	directly without going through motion planning/video encoding at all.
	"""
	if identity_engine.face_adapter is None:
		raise NoRendererAvailableError(
			"no face adapter configured; enable identity.ipadapter or identity.instantid"
		)
	lora_manager: LoraManager = PeftLoraManager(
		generation_config.lora,
		identity_config.cache_dir,
		reserved_adapter_scales={"faceid": identity_config.ipadapter.lora_scale},
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


def build_source_editor(
	identity_config: IdentityConfig,
	generation_config: GenerationConfig,
) -> SourceEditor | None:
	"""Build the stage-1 inpaint source editor, or None when
	``generation_config.inpaint.enabled`` is off (generation is then the plain
	single-stage i2i/i2v it always was).

	Takes no IdentityEngine and imposes no face-adapter restriction: stage 1
	never attaches a face adapter (see InpaintSourceEditor's docstring), so
	unlike the frame renderers it needs no per-adapter dispatch and works under
	InstantID as well as the IP-Adapter family.

	Public for the same reason build_frame_renderer() is -- single-image entry
	points (scripts/generate_image.py) need the same stage-1 edit without going
	through motion planning or video encoding.
	"""
	if not generation_config.inpaint.enabled:
		return None
	lora_manager: LoraManager = PeftLoraManager(
		generation_config.lora,
		identity_config.cache_dir,
		reserved_adapter_scales={"faceid": identity_config.ipadapter.lora_scale},
	)
	mask_generator = SamGarmentMaskGenerator(generation_config.inpaint, device=identity_config.device)
	return InpaintSourceEditor(
		identity_config,
		generation_config.inpaint,
		mask_generator,
		lora_manager,
	)


def build_generation_pipeline(
	generation_config: GenerationConfig,
	identity_config: IdentityConfig,
	identity_engine: IdentityEngine,
) -> GenerationPipeline:
	return GenerationPipeline(
		identity_engine,
		build_motion_planner(generation_config.motion),
		build_frame_warper(generation_config.motion),
		build_frame_renderer(identity_engine, identity_config, generation_config),
		temporal_smoother=build_temporal_smoother(generation_config.temporal),
		video_encoder=ImageioVideoEncoder(codec=generation_config.output.codec),
		source_editor=build_source_editor(identity_config, generation_config),
	)
