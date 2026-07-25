from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.core.config import LoraEntryConfig
from app.generation.lora.source_resolver import LoraSourceError, resolve_lora_source


def _mock_response(read_bytes=b"", headers=None):
	response = MagicMock()
	response.read.return_value = read_bytes
	response.headers = headers or {}
	response.__enter__.return_value = response
	response.__exit__.return_value = False
	return response


def test_resolve_returns_local_path_when_exists(tmp_path):
	local_file = tmp_path / "lora.safetensors"
	local_file.write_bytes(b"x")
	entry = LoraEntryConfig(adapter_name="a", source=str(local_file))

	result = resolve_lora_source(entry, tmp_path, None)

	assert result == str(local_file)


def test_resolve_returns_repo_id_unchanged(tmp_path):
	entry = LoraEntryConfig(adapter_name="a", source="some-org/some-lora-repo")

	result = resolve_lora_source(entry, tmp_path, None)

	assert result == "some-org/some-lora-repo"


def test_resolve_direct_url_downloads_to_cache_dir(tmp_path, mocker):
	response = _mock_response(
		read_bytes=b"weights", headers={"Content-Disposition": 'attachment; filename="my_lora.safetensors"'}
	)
	mocker.patch("urllib.request.urlopen", return_value=response)
	entry = LoraEntryConfig(adapter_name="a", source="https://example.com/download/lora")

	result = resolve_lora_source(entry, tmp_path, None)

	expected = tmp_path / "loras" / "my_lora.safetensors"
	assert result == str(expected)
	assert expected.read_bytes() == b"weights"


def test_resolve_reuses_already_downloaded_file(tmp_path, mocker):
	target = tmp_path / "loras" / "cached.safetensors"
	target.parent.mkdir(parents=True)
	target.write_bytes(b"cached")
	response = _mock_response(
		read_bytes=b"should not be used", headers={"Content-Disposition": 'filename="cached.safetensors"'}
	)
	mocker.patch("urllib.request.urlopen", return_value=response)
	entry = LoraEntryConfig(adapter_name="a", source="https://example.com/download/lora")

	result = resolve_lora_source(entry, tmp_path, None)

	assert result == str(target)
	assert target.read_bytes() == b"cached"


def test_resolve_civitai_model_url_calls_api_then_downloads(tmp_path, mocker):
	api_payload = json.dumps(
		{
			"modelVersions": [
				{
					"files": [
						{
							"primary": True,
							"downloadUrl": "https://civitai.com/api/download/models/999",
							"name": "cool_style.safetensors",
						}
					]
				}
			]
		}
	).encode()
	api_response = _mock_response(read_bytes=api_payload)
	download_response = _mock_response(read_bytes=b"weights")
	urlopen = mocker.patch("urllib.request.urlopen", side_effect=[api_response, download_response])
	entry = LoraEntryConfig(adapter_name="a", source="https://civitai.com/models/12345")

	result = resolve_lora_source(entry, tmp_path, "secret-key")

	assert result == str(tmp_path / "loras" / "cool_style.safetensors")
	assert urlopen.call_count == 2
	first_request = urlopen.call_args_list[0].args[0]
	assert first_request.full_url == "https://civitai.com/api/v1/models/12345"
	assert first_request.get_header("Authorization") == "Bearer secret-key"


def test_resolve_civitai_download_url_skips_api_call(tmp_path, mocker):
	response = _mock_response(read_bytes=b"weights", headers={"Content-Disposition": 'filename="direct.safetensors"'})
	urlopen = mocker.patch("urllib.request.urlopen", return_value=response)
	entry = LoraEntryConfig(adapter_name="a", source="https://civitai.com/api/download/models/999")

	result = resolve_lora_source(entry, tmp_path, None)

	assert urlopen.call_count == 1
	assert result == str(tmp_path / "loras" / "direct.safetensors")


def test_resolve_raises_lora_source_error_when_civitai_has_no_files(tmp_path, mocker):
	api_payload = json.dumps({"modelVersions": [{"files": []}]}).encode()
	api_response = _mock_response(read_bytes=api_payload)
	mocker.patch("urllib.request.urlopen", return_value=api_response)
	entry = LoraEntryConfig(adapter_name="a", source="https://civitai.com/models/12345")

	with pytest.raises(LoraSourceError):
		resolve_lora_source(entry, tmp_path, None)
