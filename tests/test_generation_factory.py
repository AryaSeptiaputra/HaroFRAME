from __future__ import annotations

import pytest

from app.core.config import GenerationConfig, IdentityConfig, InstantIdConfig
from app.generation.exceptions import NoRendererAvailableError
from app.generation.encode.video_writer import ImageioVideoEncoder
from app.generation.factory import build_generation_pipeline
from app.generation.pipeline import GenerationPipeline
from app.generation.renderer.img2img_renderer import Img2ImgFrameRenderer
from app.generation.renderer.instantid_renderer import InstantIdFrameRenderer
from app.identity.instantid.provider import InstantIdProvider


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


def test_build_generation_pipeline_wires_instantid_renderer():
	identity_engine = _FakeIdentityEngine(face_adapter=InstantIdProvider(InstantIdConfig()))

	pipeline = build_generation_pipeline(GenerationConfig(), IdentityConfig(), identity_engine)

	assert isinstance(pipeline._frame_renderer, InstantIdFrameRenderer)
