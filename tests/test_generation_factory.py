from __future__ import annotations

import pytest

from app.core.config import GenerationConfig, IdentityConfig, InpaintConfig, InstantIdConfig
from app.generation.exceptions import NoRendererAvailableError
from app.generation.encode.video_writer import ImageioVideoEncoder
from app.generation.factory import build_frame_renderer, build_generation_pipeline, build_source_editor
from app.generation.inpaint.source_editor import InpaintSourceEditor
from app.generation.lora.manager import PeftLoraManager
from app.generation.pipeline import GenerationPipeline
from app.generation.renderer.img2img_renderer import Img2ImgFrameRenderer
from app.generation.renderer.instantid_renderer import InstantIdFrameRenderer
from app.generation.temporal.passthrough import NullTemporalSmoother
from app.identity.instantid.provider import InstantIdProvider
from app.identity.segmentation.sam_provider import SamGarmentMaskGenerator


class _FakeIdentityEngine:
	def __init__(self, face_adapter):
		self.face_adapter = face_adapter


def test_build_generation_pipeline_raises_without_face_adapter():
	identity_engine = _FakeIdentityEngine(face_adapter=None)

	with pytest.raises(NoRendererAvailableError):
		build_generation_pipeline(GenerationConfig(), IdentityConfig(), identity_engine)


def test_build_generation_pipeline_wires_img2img_renderer():
	identity_engine = _FakeIdentityEngine(face_adapter=object())

	pipeline = build_generation_pipeline(GenerationConfig(), IdentityConfig(), identity_engine)

	assert isinstance(pipeline, GenerationPipeline)
	assert isinstance(pipeline._frame_renderer, Img2ImgFrameRenderer)
	assert isinstance(pipeline._video_encoder, ImageioVideoEncoder)
	assert isinstance(pipeline._frame_renderer._lora_manager, PeftLoraManager)
	assert isinstance(pipeline._temporal_smoother, NullTemporalSmoother)


def test_build_generation_pipeline_wires_instantid_renderer():
	identity_engine = _FakeIdentityEngine(face_adapter=InstantIdProvider(InstantIdConfig()))

	pipeline = build_generation_pipeline(GenerationConfig(), IdentityConfig(), identity_engine)

	assert isinstance(pipeline._frame_renderer, InstantIdFrameRenderer)


def test_build_frame_renderer_raises_without_face_adapter():
	identity_engine = _FakeIdentityEngine(face_adapter=None)

	with pytest.raises(NoRendererAvailableError):
		build_frame_renderer(identity_engine, IdentityConfig(), GenerationConfig())


def test_build_frame_renderer_usable_standalone_for_image2image():
	# scripts/generate_image.py calls this directly (not via build_generation_pipeline)
	# to render a single image without motion planning/video encoding.
	identity_engine = _FakeIdentityEngine(face_adapter=object())

	renderer = build_frame_renderer(identity_engine, IdentityConfig(), GenerationConfig())

	assert isinstance(renderer, Img2ImgFrameRenderer)
	assert isinstance(renderer._lora_manager, PeftLoraManager)


def _inpaint_enabled_config() -> GenerationConfig:
	return GenerationConfig(inpaint=InpaintConfig(prompt="a red hoodie"))


def _inpaint_disabled_config() -> GenerationConfig:
	return GenerationConfig(inpaint=InpaintConfig(enabled=False))


def test_build_source_editor_returns_none_when_inpaint_disabled():
	assert build_source_editor(IdentityConfig(), _inpaint_disabled_config()) is None


def test_build_source_editor_is_wired_by_default():
	# InpaintConfig.enabled defaults to True -- a plain GenerationConfig is two-stage.
	assert isinstance(build_source_editor(IdentityConfig(), GenerationConfig()), InpaintSourceEditor)


def test_build_source_editor_wires_sam_mask_generator():
	editor = build_source_editor(IdentityConfig(), _inpaint_enabled_config())

	assert isinstance(editor, InpaintSourceEditor)
	assert isinstance(editor._mask_generator, SamGarmentMaskGenerator)
	assert isinstance(editor._lora_manager, PeftLoraManager)


def test_build_source_editor_needs_no_face_adapter_and_allows_instantid():
	# Stage 1 never attaches a face adapter, so unlike the frame renderers it takes
	# no IdentityEngine at all and imposes no adapter restriction -- InstantID users
	# get garment/body editing too.
	editor = build_source_editor(IdentityConfig(), _inpaint_enabled_config())

	assert isinstance(editor, InpaintSourceEditor)


def test_build_generation_pipeline_wires_source_editor_when_inpaint_enabled():
	identity_engine = _FakeIdentityEngine(face_adapter=InstantIdProvider(InstantIdConfig()))

	pipeline = build_generation_pipeline(_inpaint_enabled_config(), IdentityConfig(), identity_engine)

	assert isinstance(pipeline._source_editor, InpaintSourceEditor)


def test_build_generation_pipeline_leaves_source_editor_unset_when_inpaint_disabled():
	identity_engine = _FakeIdentityEngine(face_adapter=object())

	pipeline = build_generation_pipeline(_inpaint_disabled_config(), IdentityConfig(), identity_engine)

	assert pipeline._source_editor is None
