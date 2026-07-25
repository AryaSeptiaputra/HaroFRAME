from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app.generation.exceptions import NoRendererAvailableError
from app.generation.interfaces import FrameTransform, GenerationRequest, MotionPlan, RenderedFrame
from app.generation.pipeline import GenerationPipeline
from app.identity.interfaces import FaceEmbedding, IdentityReference


class _FakeIdentityEngine:
	def __init__(self, face_adapter=object()):
		self.face_adapter = face_adapter
		self.prepare_reference_calls = 0

	def prepare_reference(self, reference):
		self.prepare_reference_calls += 1
		reference.fused_embedding = FaceEmbedding(
			vector=np.zeros(4, dtype=np.float32), det_score=0.9, bbox=(1.0, 2.0, 3.0, 4.0)
		)
		return reference


class _FakeMotionPlanner:
	def __init__(self, num_frames=3):
		self.num_frames = num_frames
		self.calls = []

	def plan(self, source_size, spec):
		self.calls.append((source_size, spec))
		transforms = [
			FrameTransform(scale=1.0, translate_x=0.0, translate_y=0.0, frame_index=i)
			for i in range(self.num_frames)
		]
		return MotionPlan(transforms=transforms, num_frames=self.num_frames, fps=8)


class _FakeFrameWarper:
	def warp(self, source_image, transform):
		return source_image


class _FakeFrameRenderer:
	def __init__(self):
		self.render_calls = []

	def render(self, warped_frame, *, reference, prompt, negative_prompt, seed, frame_index, strength=None):
		self.render_calls.append(frame_index)
		return RenderedFrame(image=warped_frame, frame_index=frame_index, seed=seed)


class _FakeTemporalSmoother:
	def __init__(self):
		self.smooth_calls = 0

	def smooth(self, frames):
		self.smooth_calls += 1
		return list(frames)


class _FakeVideoEncoder:
	def __init__(self):
		self.encode_calls = []

	def encode(self, frames, fps, output_path):
		self.encode_calls.append((len(frames), fps, output_path))
		return output_path


def _reference():
	return IdentityReference(images=[Image.new("RGB", (100, 80))])


def test_generate_raises_without_face_adapter():
	pipeline = GenerationPipeline(
		_FakeIdentityEngine(face_adapter=None), _FakeMotionPlanner(), _FakeFrameWarper(), _FakeFrameRenderer()
	)

	with pytest.raises(NoRendererAvailableError):
		pipeline.generate(GenerationRequest(reference=_reference(), prompt="x"))


def test_generate_prepares_reference_when_not_already_prepared():
	identity_engine = _FakeIdentityEngine()
	pipeline = GenerationPipeline(identity_engine, _FakeMotionPlanner(), _FakeFrameWarper(), _FakeFrameRenderer())

	pipeline.generate(GenerationRequest(reference=_reference(), prompt="x"))

	assert identity_engine.prepare_reference_calls == 1


def test_generate_skips_prepare_reference_when_already_prepared():
	identity_engine = _FakeIdentityEngine()
	reference = _reference()
	reference.fused_embedding = FaceEmbedding(vector=np.zeros(4, dtype=np.float32), det_score=0.9, bbox=(0, 0, 1, 1))
	pipeline = GenerationPipeline(identity_engine, _FakeMotionPlanner(), _FakeFrameWarper(), _FakeFrameRenderer())

	pipeline.generate(GenerationRequest(reference=reference, prompt="x"))

	assert identity_engine.prepare_reference_calls == 0


def test_generate_passes_face_bbox_into_motion_spec():
	motion_planner = _FakeMotionPlanner()
	pipeline = GenerationPipeline(_FakeIdentityEngine(), motion_planner, _FakeFrameWarper(), _FakeFrameRenderer())

	pipeline.generate(GenerationRequest(reference=_reference(), prompt="x"))

	_, spec = motion_planner.calls[0]
	assert spec.face_bbox == (1.0, 2.0, 3.0, 4.0)


def test_generate_renders_one_frame_per_planned_transform():
	renderer = _FakeFrameRenderer()
	pipeline = GenerationPipeline(_FakeIdentityEngine(), _FakeMotionPlanner(num_frames=5), _FakeFrameWarper(), renderer)

	result = pipeline.generate(GenerationRequest(reference=_reference(), prompt="x"))

	assert renderer.render_calls == [0, 1, 2, 3, 4]
	assert len(result.frames) == 5
	assert result.fps == 8


def test_generate_applies_temporal_smoother_when_present():
	smoother = _FakeTemporalSmoother()
	pipeline = GenerationPipeline(
		_FakeIdentityEngine(),
		_FakeMotionPlanner(num_frames=2),
		_FakeFrameWarper(),
		_FakeFrameRenderer(),
		temporal_smoother=smoother,
	)

	pipeline.generate(GenerationRequest(reference=_reference(), prompt="x"))

	assert smoother.smooth_calls == 1


def test_generate_encodes_video_when_output_path_and_encoder_given(tmp_path):
	encoder = _FakeVideoEncoder()
	pipeline = GenerationPipeline(
		_FakeIdentityEngine(),
		_FakeMotionPlanner(num_frames=3),
		_FakeFrameWarper(),
		_FakeFrameRenderer(),
		video_encoder=encoder,
	)
	output_path = tmp_path / "out.mp4"

	result = pipeline.generate(GenerationRequest(reference=_reference(), prompt="x"), output_path=output_path)

	assert encoder.encode_calls == [(3, 8, output_path)]
	assert result.output_path == output_path


def test_generate_skips_encoding_without_encoder(tmp_path):
	pipeline = GenerationPipeline(_FakeIdentityEngine(), _FakeMotionPlanner(), _FakeFrameWarper(), _FakeFrameRenderer())

	result = pipeline.generate(GenerationRequest(reference=_reference(), prompt="x"), output_path=tmp_path / "out.mp4")

	assert result.output_path is None
