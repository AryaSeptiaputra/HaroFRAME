from __future__ import annotations

from PIL import Image

from app.core.config import GarmentSwapConfig, IdentityConfig, RenderConfig
from app.generation.renderer.garment_swap_renderer import GarmentSwapFrameRenderer
from app.identity.controlnet.provider.pose_dwpose import DwPoseConditioner
from app.identity.interfaces import IdentityConditioning, IdentityReference
from app.identity.segmentation.interfaces import GarmentMask, SamPromptSet


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


class _FakeMaskGenerator:
	def __init__(self):
		self.received_images = []
		self.mask_image = Image.new("L", (8, 8), color=255)

	def generate_mask(self, image):
		self.received_images.append(image)
		return GarmentMask(
			mask=self.mask_image,
			prompt_set=SamPromptSet(points=None, labels=None, box=None),
		)


def _reference():
	return IdentityReference(images=[Image.new("RGB", (8, 8))])


def test_render_garment_swap_without_pose_controlnet_uses_plain_inpaint_pipeline(mocker):
	fake_pipeline = _FakePipeline()
	from_pretrained = mocker.patch(
		"diffusers.StableDiffusionXLInpaintPipeline.from_pretrained", return_value=fake_pipeline
	)
	identity_engine = _FakeIdentityEngine()
	garment_config = GarmentSwapConfig(use_pose_controlnet=False, inpaint_strength=0.7)
	mask_generator = _FakeMaskGenerator()
	renderer = GarmentSwapFrameRenderer(
		identity_engine, IdentityConfig(device="cpu"), garment_config, RenderConfig(), mask_generator
	)
	source = Image.new("RGB", (8, 8))

	result = renderer.render_garment_swap(
		source, reference=_reference(), garment_prompt="summer outfit", negative_prompt="", seed=1
	)

	from_pretrained.assert_called_once()
	kwargs = fake_pipeline.call_kwargs
	assert kwargs["prompt"] == "summer outfit"
	assert kwargs["image"] is source
	assert kwargs["mask_image"] is mask_generator.mask_image
	assert kwargs["strength"] == 0.7
	assert kwargs["ip_adapter_image_embeds"] == ["x"]
	assert "control_image" not in kwargs
	assert "controlnet_conditioning_scale" not in kwargs
	assert result.seed == 1
	assert mask_generator.received_images == [source]


def test_render_garment_swap_with_pose_controlnet_uses_controlnet_inpaint_pipeline(mocker):
	fake_pipeline = _FakePipeline()
	from_pretrained = mocker.patch(
		"diffusers.StableDiffusionXLControlNetInpaintPipeline.from_pretrained", return_value=fake_pipeline
	)
	fake_control_image = Image.new("RGB", (8, 8))
	mocker.patch.object(DwPoseConditioner, "ensure_controlnet", return_value=object())
	mocker.patch.object(DwPoseConditioner, "preprocess", return_value=fake_control_image)
	identity_engine = _FakeIdentityEngine()
	garment_config = GarmentSwapConfig(use_pose_controlnet=True, pose_conditioning_scale=0.4)
	mask_generator = _FakeMaskGenerator()
	renderer = GarmentSwapFrameRenderer(
		identity_engine, IdentityConfig(device="cpu"), garment_config, RenderConfig(), mask_generator
	)
	source = Image.new("RGB", (8, 8))

	renderer.render_garment_swap(source, reference=_reference(), garment_prompt="p", negative_prompt="", seed=1)

	from_pretrained.assert_called_once()
	kwargs = fake_pipeline.call_kwargs
	assert kwargs["control_image"] is fake_control_image
	assert kwargs["controlnet_conditioning_scale"] == 0.4


def test_render_garment_swap_strength_override_takes_precedence(mocker):
	fake_pipeline = _FakePipeline()
	mocker.patch("diffusers.StableDiffusionXLInpaintPipeline.from_pretrained", return_value=fake_pipeline)
	identity_engine = _FakeIdentityEngine()
	garment_config = GarmentSwapConfig(use_pose_controlnet=False, inpaint_strength=0.5)
	renderer = GarmentSwapFrameRenderer(
		identity_engine, IdentityConfig(device="cpu"), garment_config, RenderConfig(), _FakeMaskGenerator()
	)

	renderer.render_garment_swap(
		Image.new("RGB", (8, 8)),
		reference=_reference(),
		garment_prompt="p",
		negative_prompt="",
		seed=1,
		strength=0.95,
	)

	assert fake_pipeline.call_kwargs["strength"] == 0.95


def test_render_garment_swap_builds_pipeline_once_and_reuses_it(mocker):
	fake_pipeline = _FakePipeline()
	from_pretrained = mocker.patch(
		"diffusers.StableDiffusionXLInpaintPipeline.from_pretrained", return_value=fake_pipeline
	)
	identity_engine = _FakeIdentityEngine()
	garment_config = GarmentSwapConfig(use_pose_controlnet=False)
	renderer = GarmentSwapFrameRenderer(
		identity_engine, IdentityConfig(device="cpu"), garment_config, RenderConfig(), _FakeMaskGenerator()
	)

	renderer.render_garment_swap(
		Image.new("RGB", (8, 8)), reference=_reference(), garment_prompt="p", negative_prompt="", seed=1
	)
	renderer.render_garment_swap(
		Image.new("RGB", (8, 8)), reference=_reference(), garment_prompt="p", negative_prompt="", seed=2
	)

	from_pretrained.assert_called_once()
	assert identity_engine.loaded_pipelines == [fake_pipeline]
