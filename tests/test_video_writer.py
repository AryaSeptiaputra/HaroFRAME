from __future__ import annotations

import pytest
from PIL import Image

from app.generation.encode.video_writer import ImageioVideoEncoder
from app.generation.exceptions import VideoEncodeError


def test_encode_raises_on_empty_frames(tmp_path):
	encoder = ImageioVideoEncoder()

	with pytest.raises(VideoEncodeError):
		encoder.encode([], 8, tmp_path / "out.mp4")


def test_encode_calls_imageio_mimwrite_with_expected_args(mocker, tmp_path):
	mimwrite = mocker.patch("imageio.mimwrite")
	encoder = ImageioVideoEncoder(codec="libx264")
	frames = [Image.new("RGB", (4, 4)) for _ in range(3)]
	output_path = tmp_path / "nested" / "out.mp4"

	result = encoder.encode(frames, 12, output_path)

	assert result == output_path
	assert output_path.parent.exists()
	mimwrite.assert_called_once()
	args, kwargs = mimwrite.call_args
	assert args[0] == str(output_path)
	assert len(args[1]) == 3
	assert kwargs["fps"] == 12
	assert kwargs["codec"] == "libx264"


def test_encode_wraps_imageio_errors(mocker, tmp_path):
	mocker.patch("imageio.mimwrite", side_effect=RuntimeError("boom"))
	encoder = ImageioVideoEncoder()
	frames = [Image.new("RGB", (4, 4))]

	with pytest.raises(VideoEncodeError):
		encoder.encode(frames, 8, tmp_path / "out.mp4")
