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
		self.rendered_sources = []

	def render(self, warped_frame, *, reference, prompt, negative_prompt, seed, frame_index, strength=None):
		self.render_calls.append(frame_index)
		self.rendered_sources.append(warped_frame)
		return RenderedFrame(image=warped_frame, frame_index=frame_index, seed=seed)


class _FakeSourceEditor:
	def __init__(self, edited_image=None):
		self.calls = []
		self.edited_image = edited_image or Image.new("RGB", (100, 80), color="red")

	def edit(self, source_image, *, prompt=None, negative_prompt="", seed, strength=None):
		self.calls.append((source_image, prompt, negative_prompt, seed))
		return self.edited_image


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


def test_generate_calls_progress_callback_after_each_frame():
	pipeline = GenerationPipeline(_FakeIdentityEngine(), _FakeMotionPlanner(num_frames=3), _FakeFrameWarper(), _FakeFrameRenderer())
	progress_calls = []

	pipeline.generate(
		GenerationRequest(reference=_reference(), prompt="x"),
		progress_callback=lambda done, total: progress_calls.append((done, total)),
	)

	assert progress_calls == [(1, 3), (2, 3), (3, 3)]


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


def test_generate_runs_source_editor_once_for_the_whole_clip():
	editor = _FakeSourceEditor()
	renderer = _FakeFrameRenderer()
	pipeline = GenerationPipeline(
		_FakeIdentityEngine(),
		_FakeMotionPlanner(num_frames=4),
		_FakeFrameWarper(),
		renderer,
		source_editor=editor,
	)

	pipeline.generate(GenerationRequest(reference=_reference(), prompt="x", inpaint_prompt="a red hoodie"))

	assert len(editor.calls) == 1
	assert editor.calls[0][1] == "a red hoodie"
	# every frame is rendered from the edited photo, not the original
	assert renderer.rendered_sources == [editor.edited_image] * 4


def test_generate_keeps_identity_reference_on_the_unedited_photo():
	# The face adapter must keep conditioning on the original photo -- stage 1 only
	# replaces the image being transformed.
	identity_engine = _FakeIdentityEngine()
	reference = _reference()
	original_image = reference.images[0]
	pipeline = GenerationPipeline(
		identity_engine,
		_FakeMotionPlanner(),
		_FakeFrameWarper(),
		_FakeFrameRenderer(),
		source_editor=_FakeSourceEditor(),
	)

	pipeline.generate(GenerationRequest(reference=reference, prompt="x", inpaint_prompt="p"))

	assert identity_engine.prepare_reference_calls == 1
	assert reference.images == [original_image]


def test_generate_shares_one_seed_between_source_edit_and_frames():
	editor = _FakeSourceEditor()
	pipeline = GenerationPipeline(
		_FakeIdentityEngine(), _FakeMotionPlanner(num_frames=2), _FakeFrameWarper(), _FakeFrameRenderer(),
		source_editor=editor,
	)

	result = pipeline.generate(GenerationRequest(reference=_reference(), prompt="x", inpaint_prompt="p", seed=77))

	assert editor.calls[0][3] == 77
	assert [frame.seed for frame in result.frames] == [77, 77]


def test_generate_renders_from_original_photo_without_source_editor():
	reference = _reference()
	renderer = _FakeFrameRenderer()
	pipeline = GenerationPipeline(_FakeIdentityEngine(), _FakeMotionPlanner(num_frames=2), _FakeFrameWarper(), renderer)

	pipeline.generate(GenerationRequest(reference=reference, prompt="x", inpaint_prompt="ignored"))

	assert renderer.rendered_sources == [reference.images[0]] * 2


def test_generate_skips_encoding_without_encoder(tmp_path):
	pipeline = GenerationPipeline(_FakeIdentityEngine(), _FakeMotionPlanner(), _FakeFrameWarper(), _FakeFrameRenderer())

	result = pipeline.generate(GenerationRequest(reference=_reference(), prompt="x"), output_path=tmp_path / "out.mp4")

	assert result.output_path is None


def test_generate_releases_the_source_editor_before_rendering():
	# Stage 1's pipeline is dead weight during stage 2, and the two together do
	# not fit on a 24GB card.
	class _ReleasingEditor(_FakeSourceEditor):
		def __init__(self):
			super().__init__()
			self.released_after = None

		def release(self):
			self.released_after = len(self.calls)

	editor = _ReleasingEditor()
	renderer = _FakeFrameRenderer()
	pipeline = GenerationPipeline(
		_FakeIdentityEngine(), _FakeMotionPlanner(num_frames=2), _FakeFrameWarper(), renderer,
		source_editor=editor,
	)

	pipeline.generate(GenerationRequest(reference=_reference(), prompt="x", inpaint_prompt="p"))

	assert editor.released_after == 1
	assert len(renderer.render_calls) == 2


def test_generate_tolerates_a_source_editor_without_release():
	# release() is an optional part of the SourceEditor protocol.
	pipeline = GenerationPipeline(
		_FakeIdentityEngine(), _FakeMotionPlanner(), _FakeFrameWarper(), _FakeFrameRenderer(),
		source_editor=_FakeSourceEditor(),
	)

	pipeline.generate(GenerationRequest(reference=_reference(), prompt="x", inpaint_prompt="p"))
