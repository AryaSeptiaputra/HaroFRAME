from __future__ import annotations

import pytest
from PIL import Image

from app.core.config import IdentityConfig, InpaintConfig
from app.generation.exceptions import SourceEditError
from app.generation.inpaint.source_editor import InpaintSourceEditor
from app.generation.resolution import sdxl_working_size
from app.identity.controlnet.provider.pose_dwpose import DwPoseConditioner
from app.identity.exceptions import ModelLoadError
from app.identity.segmentation.interfaces import GarmentMask, SamPromptSet


class _FakePipelineResult:
	def __init__(self, image):
		self.images = [image]


class _FakePipeline:
	def __init__(self, output_image=None):
		self.call_kwargs = None
		self.to_calls = []
		self._output_image = output_image or Image.new("RGB", (8, 8))

	def to(self, device, dtype):
		self.to_calls.append((device, dtype))
		return self

	def __call__(self, **kwargs):
		self.call_kwargs = kwargs
		return _FakePipelineResult(self._output_image)


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


def _editor(mocker, config, *, mask_generator=None, output_image=None, lora_manager=None):
	fake_pipeline = _FakePipeline(output_image)
	target = (
		"diffusers.StableDiffusionXLControlNetInpaintPipeline.from_pretrained"
		if config.use_pose_controlnet
		else "diffusers.StableDiffusionXLInpaintPipeline.from_pretrained"
	)
	from_pretrained = mocker.patch(target, return_value=fake_pipeline)
	editor = InpaintSourceEditor(
		IdentityConfig(device="cpu"),
		config,
		mask_generator or _FakeMaskGenerator(),
		lora_manager,
	)
	return editor, fake_pipeline, from_pretrained


def test_edit_without_pose_controlnet_uses_plain_inpaint_pipeline(mocker):
	config = InpaintConfig(enabled=True, prompt="summer outfit", use_pose_controlnet=False, strength=0.7)
	mask_generator = _FakeMaskGenerator()
	editor, fake_pipeline, from_pretrained = _editor(mocker, config, mask_generator=mask_generator)
	source = Image.new("RGB", (8, 8))

	result = editor.edit(source, seed=1)

	from_pretrained.assert_called_once()
	kwargs = fake_pipeline.call_kwargs
	assert kwargs["prompt"] == "summer outfit"
	assert kwargs["image"] is source
	assert kwargs["mask_image"] is mask_generator.mask_image
	assert kwargs["strength"] == 0.7
	assert "control_image" not in kwargs
	assert mask_generator.received_images == [source]
	assert result.size == source.size


def test_edit_applies_no_identity_conditioning(mocker):
	# Stage 1 is a plain SDXL inpaint: the mask covers clothing/limbs, never the
	# face, so no face-adapter kwargs are passed -- that is what lets InstantID
	# use this stage at all.
	config = InpaintConfig(enabled=True, prompt="p", use_pose_controlnet=False)
	editor, fake_pipeline, _ = _editor(mocker, config)

	editor.edit(Image.new("RGB", (8, 8)), seed=1)

	kwargs = fake_pipeline.call_kwargs
	assert not any(key.startswith("ip_adapter") for key in kwargs)
	assert "image_embeds" not in kwargs


def test_edit_with_pose_controlnet_uses_controlnet_inpaint_pipeline(mocker):
	fake_control_image = Image.new("RGB", (8, 8))
	mocker.patch.object(DwPoseConditioner, "ensure_controlnet", return_value=object())
	mocker.patch.object(DwPoseConditioner, "preprocess", return_value=fake_control_image)
	config = InpaintConfig(enabled=True, prompt="p", use_pose_controlnet=True, pose_conditioning_scale=0.4)
	editor, fake_pipeline, from_pretrained = _editor(mocker, config)

	editor.edit(Image.new("RGB", (8, 8)), seed=1)

	from_pretrained.assert_called_once()
	kwargs = fake_pipeline.call_kwargs
	assert kwargs["control_image"] is fake_control_image
	assert kwargs["controlnet_conditioning_scale"] == 0.4


def test_edit_per_call_prompt_and_strength_take_precedence(mocker):
	config = InpaintConfig(enabled=True, prompt="from config", use_pose_controlnet=False, strength=0.5)
	editor, fake_pipeline, _ = _editor(mocker, config)

	editor.edit(Image.new("RGB", (8, 8)), prompt="per call", seed=1, strength=0.95)

	assert fake_pipeline.call_kwargs["prompt"] == "per call"
	assert fake_pipeline.call_kwargs["strength"] == 0.95


def test_edit_raises_when_no_prompt_anywhere(mocker):
	config = InpaintConfig(enabled=True, use_pose_controlnet=False)
	editor, _, from_pretrained = _editor(mocker, config)

	with pytest.raises(SourceEditError):
		editor.edit(Image.new("RGB", (8, 8)), seed=1)

	# fails before paying for any model load
	from_pretrained.assert_not_called()


def test_edit_restores_source_size_when_pipeline_rounds_resolution(mocker):
	# SDXL rounds to a multiple of 8; the caller's face bbox/motion plan assume
	# the source size, so the editor must hand back exactly that.
	config = InpaintConfig(enabled=True, prompt="p", use_pose_controlnet=False)
	editor, _, _ = _editor(mocker, config, output_image=Image.new("RGB", (64, 64)))
	source = Image.new("RGB", (70, 50))

	result = editor.edit(source, seed=1)

	assert result.size == (70, 50)


def test_edit_generates_mask_before_loading_the_pipeline(mocker):
	# A missing SAM checkpoint must surface before a multi-GB inpaint pipeline load.
	class _FailingMaskGenerator:
		def generate_mask(self, image):
			raise ModelLoadError("SAM checkpoint not found")

	config = InpaintConfig(enabled=True, prompt="p", use_pose_controlnet=False)
	editor, _, from_pretrained = _editor(mocker, config, mask_generator=_FailingMaskGenerator())

	with pytest.raises(ModelLoadError):
		editor.edit(Image.new("RGB", (8, 8)), seed=1)

	from_pretrained.assert_not_called()


def test_sdxl_working_size_leaves_small_images_alone():
	assert sdxl_working_size((768, 512)) == (768, 512)


def test_sdxl_working_size_scales_large_images_into_the_budget():
	# A 1824px phone photo is what produced "tensor a (228) vs tensor b (64)".
	width, height = sdxl_working_size((1824, 1216))

	assert width * height <= 1024 * 1024
	assert abs((width / height) - (1824 / 1216)) < 0.02


@pytest.mark.parametrize("size", [(1824, 1216), (4000, 3000), (999, 733), (100, 4000)])
def test_sdxl_working_size_always_lands_on_the_multiple_of_eight_grid(size):
	width, height = sdxl_working_size(size)

	assert width % 8 == 0 and height % 8 == 0
	assert width >= 8 and height >= 8


def test_edit_passes_an_explicit_matching_height_and_width(mocker):
	# Without these the pipeline preprocesses the init image and the pose control
	# image separately, and they disagree inside the ControlNet.
	config = InpaintConfig(enabled=True, prompt="p", use_pose_controlnet=False)
	editor, fake_pipeline, _ = _editor(mocker, config)

	editor.edit(Image.new("RGB", (1824, 1216)), seed=1)

	kwargs = fake_pipeline.call_kwargs
	assert (kwargs["width"], kwargs["height"]) == sdxl_working_size((1824, 1216))


def test_edit_builds_pipeline_once_and_reuses_it(mocker):
	config = InpaintConfig(enabled=True, prompt="p", use_pose_controlnet=False)
	editor, _, from_pretrained = _editor(mocker, config)

	editor.edit(Image.new("RGB", (8, 8)), seed=1)
	editor.edit(Image.new("RGB", (8, 8)), seed=2)

	from_pretrained.assert_called_once()


def test_release_drops_the_pipeline_so_it_rebuilds_on_next_use(mocker):
	# Stage 1 and stage 2 never overlap; holding both pipelines exhausts a 24GB card.
	config = InpaintConfig(enabled=True, prompt="p", use_pose_controlnet=False)
	editor, _, from_pretrained = _editor(mocker, config)

	editor.edit(Image.new("RGB", (8, 8)), seed=1)
	editor.release()
	editor.edit(Image.new("RGB", (8, 8)), seed=2)

	assert from_pretrained.call_count == 2


def test_release_cascades_to_the_pose_conditioner_and_mask_generator(mocker):
	class _ReleasableMaskGenerator(_FakeMaskGenerator):
		def __init__(self):
			super().__init__()
			self.released = False

		def release(self):
			self.released = True

	mocker.patch.object(DwPoseConditioner, "ensure_controlnet", return_value=object())
	mocker.patch.object(DwPoseConditioner, "preprocess", return_value=Image.new("RGB", (8, 8)))
	config = InpaintConfig(enabled=True, prompt="p", use_pose_controlnet=True)
	mask_generator = _ReleasableMaskGenerator()
	editor, _, _ = _editor(mocker, config, mask_generator=mask_generator)
	editor.edit(Image.new("RGB", (8, 8)), seed=1)
	editor._pose_conditioner._controlnet = object()
	editor._pose_conditioner._detector = object()

	editor.release()

	assert mask_generator.released
	assert editor._pose_conditioner._controlnet is None
	assert editor._pose_conditioner._detector is None


def test_release_is_safe_before_anything_was_built(mocker):
	config = InpaintConfig(enabled=True, prompt="p", use_pose_controlnet=False)
	editor, _, _ = _editor(mocker, config)

	editor.release()  # must not raise
