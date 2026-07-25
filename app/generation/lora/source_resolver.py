from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from app.core.config import LoraEntryConfig
from app.generation.exceptions import GenerationModuleError

_CIVITAI_MODEL_URL_RE = re.compile(r"civitai\.com/models/(\d+)", re.IGNORECASE)
_CIVITAI_VERSION_QUERY_RE = re.compile(r"modelVersionId=(\d+)", re.IGNORECASE)
_CIVITAI_DOWNLOAD_URL_RE = re.compile(r"civitai\.com/api/download/models/(\d+)", re.IGNORECASE)
_CONTENT_DISPOSITION_FILENAME_RE = re.compile(r'filename="?([^";]+)"?')


class LoraSourceError(GenerationModuleError):
	"""Raised when a LoRA source (URL, repo_id, or path) can't be resolved/downloaded."""


def resolve_lora_source(entry: LoraEntryConfig, cache_dir: Path, civitai_api_key: str | None) -> str:
	"""Resolve ``entry.source`` into something ``pipeline.load_lora_weights()`` can
	consume directly.

	Three cases: an existing local path is returned as-is; a Civitai model-page
	URL is resolved via Civitai's API to the real file and downloaded; any other
	http(s) URL is downloaded directly. Anything else (no scheme, not a local
	path) is assumed to already be a Hugging Face repo_id and is returned
	unchanged -- diffusers/huggingface_hub resolve those natively.
	"""
	source = entry.source
	local_path = Path(source)
	if local_path.exists():
		return str(local_path)

	parsed = urlparse(source)
	if parsed.scheme in ("http", "https"):
		download_url, filename = _resolve_download_url(source, civitai_api_key)
		return str(_download(download_url, filename, cache_dir, civitai_api_key))

	return source


def _resolve_download_url(source: str, civitai_api_key: str | None) -> tuple[str, str]:
	if _CIVITAI_DOWNLOAD_URL_RE.search(source):
		# Already a direct download endpoint -- filename is figured out from
		# the response headers when we actually download it.
		return source, ""

	model_match = _CIVITAI_MODEL_URL_RE.search(source)
	if model_match:
		version_match = _CIVITAI_VERSION_QUERY_RE.search(source)
		if version_match:
			api_url = f"https://civitai.com/api/v1/model-versions/{version_match.group(1)}"
		else:
			api_url = f"https://civitai.com/api/v1/models/{model_match.group(1)}"
		payload = _fetch_json(api_url, civitai_api_key)
		version = payload if "files" in payload else payload["modelVersions"][0]
		files = version.get("files", [])
		if not files:
			raise LoraSourceError(f"Civitai model at {source!r} has no downloadable files")
		primary = next((f for f in files if f.get("primary")), files[0])
		return primary["downloadUrl"], primary.get("name", "")

	# Generic direct-download URL (any other platform).
	return source, ""


def _fetch_json(url: str, civitai_api_key: str | None) -> dict:
	request = urllib.request.Request(url)
	if civitai_api_key:
		request.add_header("Authorization", f"Bearer {civitai_api_key}")
	try:
		with urllib.request.urlopen(request, timeout=30) as response:
			return json.loads(response.read())
	except LoraSourceError:
		raise
	except Exception as exc:
		raise LoraSourceError(f"failed to query {url!r}: {exc}") from exc


def _download(url: str, filename: str, cache_dir: Path, civitai_api_key: str | None) -> Path:
	request = urllib.request.Request(url)
	if civitai_api_key and "civitai.com" in url:
		request.add_header("Authorization", f"Bearer {civitai_api_key}")
	try:
		with urllib.request.urlopen(request, timeout=300) as response:
			resolved_filename = filename or _filename_from_response(response, url)
			target_dir = cache_dir / "loras"
			target_dir.mkdir(parents=True, exist_ok=True)
			target_path = target_dir / resolved_filename
			if not target_path.exists():
				with open(target_path, "wb") as file_obj:
					file_obj.write(response.read())
			return target_path
	except LoraSourceError:
		raise
	except Exception as exc:
		raise LoraSourceError(f"failed to download LoRA from {url!r}: {exc}") from exc


def _filename_from_response(response, url: str) -> str:
	content_disposition = response.headers.get("Content-Disposition", "")
	match = _CONTENT_DISPOSITION_FILENAME_RE.search(content_disposition)
	if match:
		return match.group(1)
	basename = Path(urlparse(url).path).name
	if basename:
		return basename
	return hashlib.sha1(url.encode()).hexdigest() + ".safetensors"
