from __future__ import annotations

from PIL import Image

from app.core.config import IdentityConfig, RenderConfig
from app.generation.renderer.img2img_renderer import Img2ImgFrameRenderer
from app.identity.interfaces import IdentityConditioning, IdentityReference


class _FakePipelineResult:
	def __init__(self, image):
		self.images = [image]


class _FakePipeline:
	def __init__(self):
		self.call_kwargs = None
		self.to_calls = []

	def to(self, device, dtype):
		self.to_calls.append((device, dtype))
		return self

	def __call__(self, **kwargs):
		self.call_kwargs = kwargs
		return _FakePipelineResult(Image.new("RGB", (8, 8)))


class _FakeIdentityEngine:
	def __init__(self):
		self.loaded_pipelines = []
		self.conditioning = IdentityConditioning(
			adapter_kwargs={"ip_adapter_image_embeds": ["x"]}, applied_adapters=["ip_adapter_faceid_sdxl"]
		)

	def load(self, pipeline):
		self.loaded_pipelines.append(pipeline)

	def build_conditioning(self, reference, *, structure=None, scale=1.0):
		return self.conditioning


def _reference():
	return IdentityReference(images=[Image.new("RGB", (8, 8))])


def test_render_builds_pipeline_once_and_reuses_it(mocker):
	fake_pipeline = _FakePipeline()
	from_pretrained = mocker.patch(
		"diffusers.StableDiffusionXLImg2ImgPipeline.from_pretrained", return_value=fake_pipeline
	)
	identity_engine = _FakeIdentityEngine()
	renderer = Img2ImgFrameRenderer(identity_engine, IdentityConfig(device="cpu"), RenderConfig())
	warped = Image.new("RGB", (8, 8))
	reference = _reference()

	renderer.render(warped, reference=reference, prompt="p", negative_prompt="", seed=42, frame_index=0)
	renderer.render(warped, reference=reference, prompt="p", negative_prompt="", seed=42, frame_index=1)

	from_pretrained.assert_called_once()
	assert identity_engine.loaded_pipelines == [fake_pipeline]


def test_render_passes_conditioning_kwargs_and_render_config(mocker):
	fake_pipeline = _FakePipeline()
	mocker.patch("diffusers.StableDiffusionXLImg2ImgPipeline.from_pretrained", return_value=fake_pipeline)
	identity_engine = _FakeIdentityEngine()
	render_config = RenderConfig(strength=0.4, guidance_scale=6.0, num_inference_steps=20, negative_prompt="blurry")
	renderer = Img2ImgFrameRenderer(identity_engine, IdentityConfig(device="cpu"), render_config)
	warped = Image.new("RGB", (8, 8))
	reference = _reference()

	result = renderer.render(warped, reference=reference, prompt="a person", negative_prompt="", seed=7, frame_index=3)

	kwargs = fake_pipeline.call_kwargs
	assert kwargs["prompt"] == "a person"
	assert kwargs["negative_prompt"] == "blurry"
	assert kwargs["strength"] == 0.4
	assert kwargs["guidance_scale"] == 6.0
	assert kwargs["num_inference_steps"] == 20
	assert kwargs["ip_adapter_image_embeds"] == ["x"]
	assert result.frame_index == 3
	assert result.seed == 7


def test_render_strength_override_takes_precedence(mocker):
	fake_pipeline = _FakePipeline()
	mocker.patch("diffusers.StableDiffusionXLImg2ImgPipeline.from_pretrained", return_value=fake_pipeline)
	identity_engine = _FakeIdentityEngine()
	renderer = Img2ImgFrameRenderer(identity_engine, IdentityConfig(device="cpu"), RenderConfig(strength=0.35))
	warped = Image.new("RGB", (8, 8))
	reference = _reference()

	renderer.render(warped, reference=reference, prompt="p", negative_prompt="", seed=1, frame_index=0, strength=0.9)

	assert fake_pipeline.call_kwargs["strength"] == 0.9


def test_render_loads_lora_manager_once_when_provided(mocker):
	fake_pipeline = _FakePipeline()
	mocker.patch("diffusers.StableDiffusionXLImg2ImgPipeline.from_pretrained", return_value=fake_pipeline)
	identity_engine = _FakeIdentityEngine()
	lora_manager = mocker.Mock()
	renderer = Img2ImgFrameRenderer(identity_engine, IdentityConfig(device="cpu"), RenderConfig(), lora_manager)
	warped = Image.new("RGB", (8, 8))
	reference = _reference()

	renderer.render(warped, reference=reference, prompt="p", negative_prompt="", seed=1, frame_index=0)
	renderer.render(warped, reference=reference, prompt="p", negative_prompt="", seed=1, frame_index=1)

	lora_manager.load.assert_called_once_with(fake_pipeline)
