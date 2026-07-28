from __future__ import annotations

import json
import urllib.error
import urllib.request
from unittest.mock import MagicMock

import pytest

from app.generation.source_resolver import (
	ModelSourceError,
	_DropAuthOnCrossHostRedirect,
	resolve_model_source,
)


def _mock_response(read_bytes=b"", headers=None):
	response = MagicMock()
	response.read.return_value = read_bytes
	response.headers = headers or {}
	response.__enter__.return_value = response
	response.__exit__.return_value = False
	return response


def test_resolve_returns_local_path_when_exists(tmp_path):
	local_file = tmp_path / "model.safetensors"
	local_file.write_bytes(b"x")

	result = resolve_model_source(str(local_file), tmp_path, None)

	assert result == str(local_file)


def test_resolve_returns_repo_id_unchanged(tmp_path):
	result = resolve_model_source("some-org/some-repo", tmp_path, None)

	assert result == "some-org/some-repo"


def test_resolve_direct_url_downloads_to_cache_dir(tmp_path, mocker):
	response = _mock_response(
		read_bytes=b"weights", headers={"Content-Disposition": 'attachment; filename="my_lora.safetensors"'}
	)
	mocker.patch("app.generation.source_resolver._OPENER.open", return_value=response)

	result = resolve_model_source("https://example.com/download/lora", tmp_path, None, subdir="loras")

	expected = tmp_path / "loras" / "my_lora.safetensors"
	assert result == str(expected)
	assert expected.read_bytes() == b"weights"


def test_resolve_uses_given_subdir_for_checkpoints(tmp_path, mocker):
	response = _mock_response(
		read_bytes=b"weights", headers={"Content-Disposition": 'attachment; filename="model.safetensors"'}
	)
	mocker.patch("app.generation.source_resolver._OPENER.open", return_value=response)

	result = resolve_model_source("https://example.com/download/model", tmp_path, None, subdir="checkpoints")

	expected = tmp_path / "checkpoints" / "model.safetensors"
	assert result == str(expected)


def test_resolve_reuses_already_downloaded_file(tmp_path, mocker):
	target = tmp_path / "loras" / "cached.safetensors"
	target.parent.mkdir(parents=True)
	target.write_bytes(b"cached")
	response = _mock_response(
		read_bytes=b"should not be used", headers={"Content-Disposition": 'filename="cached.safetensors"'}
	)
	mocker.patch("app.generation.source_resolver._OPENER.open", return_value=response)

	result = resolve_model_source("https://example.com/download/lora", tmp_path, None, subdir="loras")

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
	urlopen = mocker.patch(
		"app.generation.source_resolver._OPENER.open", side_effect=[api_response, download_response]
	)

	result = resolve_model_source("https://civitai.com/models/12345", tmp_path, "secret-key", subdir="loras")

	assert result == str(tmp_path / "loras" / "cool_style.safetensors")
	assert urlopen.call_count == 2
	first_request = urlopen.call_args_list[0].args[0]
	assert first_request.full_url == "https://civitai.com/api/v1/models/12345"
	# Tried without the key first (works for public models, avoids sending a
	# possibly-stale key) -- it succeeded, so the key was never attached.
	assert first_request.get_header("Authorization") is None


def test_resolve_civitai_retries_with_api_key_when_unauthenticated_call_fails(tmp_path, mocker):
	api_payload = json.dumps(
		{
			"modelVersions": [
				{
					"files": [
						{
							"primary": True,
							"downloadUrl": "https://civitai.com/api/download/models/999",
							"name": "gated.safetensors",
						}
					]
				}
			]
		}
	).encode()
	api_response = _mock_response(read_bytes=api_payload)
	download_response = _mock_response(read_bytes=b"weights")
	unauthorized = urllib.error.HTTPError("https://civitai.com/api/v1/models/12345", 401, "Unauthorized", {}, None)
	urlopen = mocker.patch(
		"app.generation.source_resolver._OPENER.open",
		side_effect=[unauthorized, api_response, download_response],
	)

	result = resolve_model_source("https://civitai.com/models/12345", tmp_path, "secret-key", subdir="loras")

	assert result == str(tmp_path / "loras" / "gated.safetensors")
	assert urlopen.call_count == 3
	no_key_attempt = urlopen.call_args_list[0].args[0]
	retry_attempt = urlopen.call_args_list[1].args[0]
	assert no_key_attempt.get_header("Authorization") is None
	assert retry_attempt.get_header("Authorization") == "Bearer secret-key"


def test_resolve_civitai_download_reraises_when_retry_with_key_also_fails(tmp_path, mocker):
	unauthorized = urllib.error.HTTPError("https://civitai.com/api/v1/models/12345", 401, "Unauthorized", {}, None)
	still_forbidden = urllib.error.HTTPError("https://civitai.com/api/v1/models/12345", 403, "Forbidden", {}, None)
	mocker.patch(
		"app.generation.source_resolver._OPENER.open", side_effect=[unauthorized, still_forbidden]
	)

	with pytest.raises(ModelSourceError):
		resolve_model_source("https://civitai.com/models/12345", tmp_path, "secret-key", subdir="loras")


def test_resolve_civitai_download_url_skips_api_call(tmp_path, mocker):
	response = _mock_response(read_bytes=b"weights", headers={"Content-Disposition": 'filename="direct.safetensors"'})
	urlopen = mocker.patch("app.generation.source_resolver._OPENER.open", return_value=response)

	result = resolve_model_source("https://civitai.com/api/download/models/999", tmp_path, None, subdir="loras")

	assert urlopen.call_count == 1
	assert result == str(tmp_path / "loras" / "direct.safetensors")


def test_resolve_raises_model_source_error_when_civitai_has_no_files(tmp_path, mocker):
	api_payload = json.dumps({"modelVersions": [{"files": []}]}).encode()
	api_response = _mock_response(read_bytes=api_payload)
	mocker.patch("app.generation.source_resolver._OPENER.open", return_value=api_response)

	with pytest.raises(ModelSourceError):
		resolve_model_source("https://civitai.com/models/12345", tmp_path, None, subdir="loras")


def test_resolve_civitai_direct_download_retries_with_key_on_failure(tmp_path, mocker):
	unauthorized = urllib.error.HTTPError(
		"https://civitai.com/api/download/models/999", 401, "Unauthorized", {}, None
	)
	response = _mock_response(read_bytes=b"weights", headers={"Content-Disposition": 'filename="gated.safetensors"'})
	urlopen = mocker.patch(
		"app.generation.source_resolver._OPENER.open", side_effect=[unauthorized, response]
	)

	result = resolve_model_source("https://civitai.com/api/download/models/999", tmp_path, "secret-key", subdir="loras")

	assert result == str(tmp_path / "loras" / "gated.safetensors")
	assert urlopen.call_count == 2
	assert urlopen.call_args_list[0].args[0].get_header("Authorization") is None
	assert urlopen.call_args_list[1].args[0].get_header("Authorization") == "Bearer secret-key"


def test_resolve_civitai_direct_download_no_retry_without_key(tmp_path, mocker):
	unauthorized = urllib.error.HTTPError(
		"https://civitai.com/api/download/models/999", 401, "Unauthorized", {}, None
	)
	urlopen = mocker.patch("app.generation.source_resolver._OPENER.open", side_effect=[unauthorized])

	with pytest.raises(ModelSourceError):
		resolve_model_source("https://civitai.com/api/download/models/999", tmp_path, None, subdir="loras")

	assert urlopen.call_count == 1


def test_drop_auth_on_cross_host_redirect_strips_header_for_different_host():
	handler = _DropAuthOnCrossHostRedirect()
	req = urllib.request.Request("https://civitai.com/api/download/models/999")
	req.add_header("Authorization", "Bearer secret-key")

	new_req = handler.redirect_request(
		req, None, 307, "Temporary Redirect", {}, "https://r2.example.com/model.safetensors?X-Amz-Signature=abc"
	)

	assert new_req is not None
	assert new_req.get_header("Authorization") is None


def test_drop_auth_on_cross_host_redirect_keeps_header_for_same_host():
	handler = _DropAuthOnCrossHostRedirect()
	req = urllib.request.Request("https://civitai.com/api/v1/models/12345")
	req.add_header("Authorization", "Bearer secret-key")

	new_req = handler.redirect_request(
		req, None, 307, "Temporary Redirect", {}, "https://civitai.com/api/v1/models/12345/redirected"
	)

	assert new_req is not None
	assert new_req.get_header("Authorization") == "Bearer secret-key"
