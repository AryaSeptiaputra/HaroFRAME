from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import scripts.quick_generate as quick_generate  # noqa: E402
from app.core.config import GenerationConfig, InpaintConfig  # noqa: E402
from app.identity.interfaces import FaceEmbedding  # noqa: E402
from app.generation.interfaces import RenderedFrame  # noqa: E402
from scripts.quick_generate import _unique_output_path  # noqa: E402


def test_unique_output_path_uses_plain_name_when_free(tmp_path):
	assert _unique_output_path(tmp_path, "alex") == tmp_path / "alex_quick.png"


def test_unique_output_path_bumps_counter_instead_of_overwriting(tmp_path):
	(tmp_path / "alex_quick.png").touch()
	(tmp_path / "alex_quick_2.png").touch()

	assert _unique_output_path(tmp_path, "alex") == tmp_path / "alex_quick_3.png"


class _FakeIdentityEngine:
	def __init__(self):
		self.face_adapter = object()

	def prepare_reference(self, reference):
		reference.fused_embedding = FaceEmbedding(
			vector=np.zeros(4, dtype=np.float32), det_score=0.9, bbox=(0.0, 0.0, 1.0, 1.0)
		)
		return reference


class _FakeRenderer:
	def __init__(self):
		self.render_calls = []

	def render(self, image, *, reference, prompt, negative_prompt, seed, frame_index, strength=None):
		self.render_calls.append((image, prompt, negative_prompt, seed, frame_index))
		return RenderedFrame(image=Image.new("RGB", (8, 8)), frame_index=frame_index, seed=seed)


def _run_main(mocker, tmp_path, argv, *, generation_config, source_editor=None):
	photo = tmp_path / "alex.png"
	Image.new("RGB", (8, 8)).save(photo)
	settings = mocker.Mock()
	settings.identity = mocker.Mock(base_sdxl_model="some/model")
	settings.generation = generation_config.model_copy(
		update={"output": generation_config.output.model_copy(update={"output_dir": tmp_path / "out"})}
	)
	renderer = _FakeRenderer()
	mocker.patch.object(quick_generate, "get_settings", return_value=settings)
	mocker.patch.object(quick_generate, "build_identity_engine", return_value=_FakeIdentityEngine())
	mocker.patch.object(quick_generate, "build_frame_renderer", return_value=renderer)
	mocker.patch.object(quick_generate, "build_source_editor", return_value=source_editor)
	mocker.patch.object(sys, "argv", ["quick_generate.py", str(photo), *argv])
	return quick_generate.main(), renderer, tmp_path / "out"


def test_main_renders_one_image_straight_from_the_photo_with_no_inpaint(mocker, tmp_path):
	exit_code, renderer, out_dir = _run_main(
		mocker, tmp_path, ["a portrait", "-n", "blurry", "--no-inpaint"], generation_config=GenerationConfig()
	)

	assert exit_code == 0
	assert len(renderer.render_calls) == 1
	_, prompt, negative_prompt, _, frame_index = renderer.render_calls[0]
	assert (prompt, negative_prompt, frame_index) == ("a portrait", "blurry", 0)
	assert (out_dir / "alex_quick.png").is_file()


def test_main_feeds_the_stage_one_edit_into_the_render(mocker, tmp_path):
	edited = Image.new("RGB", (8, 8), color="red")
	editor = mocker.Mock()
	editor.edit.return_value = edited

	exit_code, renderer, _ = _run_main(
		mocker,
		tmp_path,
		["a portrait", "-i", "a red hoodie"],
		generation_config=GenerationConfig(),
		source_editor=editor,
	)

	assert exit_code == 0
	editor.edit.assert_called_once()
	rendered_source, _, _, render_seed, _ = renderer.render_calls[0]
	assert rendered_source is edited
	# one seed covers both stages, same as GenerationPipeline
	assert editor.edit.call_args.kwargs["seed"] == render_seed


def test_main_fails_up_front_when_inpaint_is_on_without_a_stage_one_prompt(mocker, tmp_path):
	# The default config is exactly this: enabled=True, prompt="".
	photo = tmp_path / "alex.png"
	Image.new("RGB", (8, 8)).save(photo)
	mocker.patch.object(quick_generate, "build_identity_engine", return_value=_FakeIdentityEngine())
	build_frame_renderer = mocker.patch.object(quick_generate, "build_frame_renderer")
	build_source_editor = mocker.patch.object(quick_generate, "build_source_editor")
	mocker.patch.object(sys, "argv", ["quick_generate.py", str(photo), "a portrait"])

	assert quick_generate.main() == 1
	# ...and before anything expensive: neither stage was ever built
	build_frame_renderer.assert_not_called()
	build_source_editor.assert_not_called()


def test_main_takes_the_stage_one_prompt_from_config_when_flag_omitted(mocker, tmp_path):
	editor = mocker.Mock()
	editor.edit.return_value = Image.new("RGB", (8, 8))

	exit_code, _, _ = _run_main(
		mocker,
		tmp_path,
		["a portrait"],
		generation_config=GenerationConfig(inpaint=InpaintConfig(prompt="from config")),
		source_editor=editor,
	)

	assert exit_code == 0
	editor.edit.assert_called_once()


def test_main_fails_cleanly_without_a_face_adapter(mocker, tmp_path):
	engine = _FakeIdentityEngine()
	engine.face_adapter = None
	photo = tmp_path / "alex.png"
	Image.new("RGB", (8, 8)).save(photo)
	mocker.patch.object(quick_generate, "build_identity_engine", return_value=engine)
	mocker.patch.object(sys, "argv", ["quick_generate.py", str(photo), "a portrait"])

	assert quick_generate.main() == 1


def test_main_fails_cleanly_on_a_missing_photo(mocker):
	mocker.patch.object(sys, "argv", ["quick_generate.py", "nope.jpg", "a portrait"])

	assert quick_generate.main() == 1
