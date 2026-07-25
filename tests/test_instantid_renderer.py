from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app.core.config import IdentityConfig, OutputConfig, RenderConfig
from app.generation.renderer.instantid_renderer import InstantIdFrameRenderer
from app.identity.interfaces import FaceEmbedding, IdentityConditioning, IdentityReference


class _FakePipelineResult:
	def __init__(self, image):
		self.images = [image]


class _FakePipeline:
	def __init__(self):
		self.call_kwargs = None

	def __call__(self, **kwargs):
		self.call_kwargs = kwargs
		return _FakePipelineResult(Image.new("RGB", (8, 8)))


class _FakeIdentityEngine:
	def __init__(self):
		self.loaded_pipelines = []
		self.build_conditioning_calls = []

	def load(self, pipeline):
		self.loaded_pipelines.append(pipeline)

	def build_conditioning(self, reference, *, structure=None, scale=1.0):
		self.build_conditioning_calls.append((reference, structure, scale))
		return IdentityConditioning(
			adapter_kwargs={
				"image_embeds": reference.fused_embedding.vector,
				"image": structure.pose_image,
				"controlnet_conditioning_scale": scale,
			},
			applied_adapters=["instantid"],
			used_structure_conditioning=True,
		)


def _reference_with_embedding():
	embedding = FaceEmbedding(
		vector=np.ones(512, dtype=np.float32),
		det_score=0.95,
		bbox=(10.0, 10.0, 50.0, 50.0),
		landmarks_5pt=np.zeros((5, 2), dtype=np.float32),
	)
	image = Image.new("RGB", (100, 100))
	return IdentityReference(images=[image], embeddings=[embedding], fused_embedding=embedding)


def _patch_pipeline_builder(mocker, fake_pipeline):
	return mocker.patch(
		"app.generation.renderer.instantid_renderer.build_instantid_pipeline", return_value=fake_pipeline
	)


def _patch_analyzer(mocker, faces):
	return mocker.patch("app.identity.face.analyzer_insightface.InsightFaceAnalyzer.analyze", return_value=faces)


def test_render_redetects_landmarks_on_warped_frame_but_keeps_identity_vector(mocker):
	fake_pipeline = _FakePipeline()
	_patch_pipeline_builder(mocker, fake_pipeline)
	detected = FaceEmbedding(
		vector=np.zeros(512, dtype=np.float32),
		det_score=0.8,
		bbox=(20.0, 20.0, 60.0, 60.0),
		landmarks_5pt=np.ones((5, 2), dtype=np.float32),
	)
	_patch_analyzer(mocker, [detected])
	identity_engine = _FakeIdentityEngine()
	renderer = InstantIdFrameRenderer(identity_engine, IdentityConfig(device="cpu"), RenderConfig(), OutputConfig())
	reference = _reference_with_embedding()
	warped = Image.new("RGB", (64, 64))

	result = renderer.render(warped, reference=reference, prompt="p", negative_prompt="", seed=5, frame_index=2)

	assert result.face_detected is True
	assert result.frame_index == 2
	used_reference, structure, _ = identity_engine.build_conditioning_calls[0]
	np.testing.assert_array_equal(used_reference.fused_embedding.vector, reference.fused_embedding.vector)
	np.testing.assert_array_equal(used_reference.fused_embedding.landmarks_5pt, detected.landmarks_5pt)
	assert structure.pose_image is warped
	assert structure.source == "driving_frame"


def test_render_falls_back_to_reference_landmarks_when_detection_fails(mocker):
	fake_pipeline = _FakePipeline()
	_patch_pipeline_builder(mocker, fake_pipeline)
	_patch_analyzer(mocker, [])
	identity_engine = _FakeIdentityEngine()
	renderer = InstantIdFrameRenderer(identity_engine, IdentityConfig(device="cpu"), RenderConfig(), OutputConfig())
	reference = _reference_with_embedding()

	result = renderer.render(
		Image.new("RGB", (64, 64)), reference=reference, prompt="p", negative_prompt="", seed=5, frame_index=0
	)

	assert result.face_detected is False
	used_reference, _, _ = identity_engine.build_conditioning_calls[0]
	np.testing.assert_array_equal(used_reference.fused_embedding.landmarks_5pt, reference.fused_embedding.landmarks_5pt)


def test_render_uses_instantid_conditioning_scale_from_config(mocker):
	fake_pipeline = _FakePipeline()
	_patch_pipeline_builder(mocker, fake_pipeline)
	_patch_analyzer(mocker, [])
	identity_engine = _FakeIdentityEngine()
	identity_config = IdentityConfig(device="cpu")
	identity_config.instantid.controlnet_conditioning_scale = 0.65
	renderer = InstantIdFrameRenderer(identity_engine, identity_config, RenderConfig(), OutputConfig())
	reference = _reference_with_embedding()

	renderer.render(
		Image.new("RGB", (64, 64)), reference=reference, prompt="p", negative_prompt="", seed=1, frame_index=0
	)

	_, _, scale = identity_engine.build_conditioning_calls[0]
	assert scale == pytest.approx(0.65)


def test_render_builds_pipeline_once_and_reuses_it(mocker):
	fake_pipeline = _FakePipeline()
	build_pipeline = _patch_pipeline_builder(mocker, fake_pipeline)
	_patch_analyzer(mocker, [])
	identity_engine = _FakeIdentityEngine()
	renderer = InstantIdFrameRenderer(identity_engine, IdentityConfig(device="cpu"), RenderConfig(), OutputConfig())
	reference = _reference_with_embedding()
	warped = Image.new("RGB", (64, 64))

	renderer.render(warped, reference=reference, prompt="p", negative_prompt="", seed=1, frame_index=0)
	renderer.render(warped, reference=reference, prompt="p", negative_prompt="", seed=1, frame_index=1)

	build_pipeline.assert_called_once()
	assert identity_engine.loaded_pipelines == [fake_pipeline]


def test_render_passes_output_dimensions(mocker):
	fake_pipeline = _FakePipeline()
	_patch_pipeline_builder(mocker, fake_pipeline)
	_patch_analyzer(mocker, [])
	identity_engine = _FakeIdentityEngine()
	renderer = InstantIdFrameRenderer(
		identity_engine, IdentityConfig(device="cpu"), RenderConfig(), OutputConfig(width=768, height=512)
	)
	reference = _reference_with_embedding()

	renderer.render(
		Image.new("RGB", (64, 64)), reference=reference, prompt="p", negative_prompt="", seed=1, frame_index=0
	)

	assert fake_pipeline.call_kwargs["width"] == 768
	assert fake_pipeline.call_kwargs["height"] == 512
