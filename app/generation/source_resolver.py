from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from app.generation.exceptions import GenerationModuleError

_CIVITAI_MODEL_URL_RE = re.compile(r"civitai\.com/models/(\d+)", re.IGNORECASE)
_CIVITAI_VERSION_QUERY_RE = re.compile(r"modelVersionId=(\d+)", re.IGNORECASE)
_CIVITAI_DOWNLOAD_URL_RE = re.compile(r"civitai\.com/api/download/models/(\d+)", re.IGNORECASE)
_CONTENT_DISPOSITION_FILENAME_RE = re.compile(r'filename="?([^";]+)"?')


class _DropAuthOnCrossHostRedirect(urllib.request.HTTPRedirectHandler):
	"""Civitai's download endpoint 307-redirects to a presigned Cloudflare R2 URL
	(auth embedded in the query string) -- urllib's default redirect handling
	carries the original request's Authorization header along to that new host,
	and R2/S3 rejects requests that present both a presigned-URL signature and an
	Authorization header (query-string auth + header auth at once), returning 403.
	Dropping the header whenever the redirect target's host differs from the
	original request's host avoids that, and is a no-op for any other redirect."""

	def redirect_request(self, req, fp, code, msg, headers, newurl):
		new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
		if new_req is not None and urlparse(newurl).netloc != urlparse(req.full_url).netloc:
			new_req.headers.pop("Authorization", None)
			new_req.unredirected_hdrs.pop("Authorization", None)
		return new_req


_OPENER = urllib.request.build_opener(_DropAuthOnCrossHostRedirect)


class ModelSourceError(GenerationModuleError):
	"""Raised when a model source (URL, repo_id, or path) can't be resolved/downloaded.

	Shared by LoRA installation (app/generation/lora/manager.py) and SDXL
	checkpoint installation (scripts/interactive_generate.py) -- both a LoRA
	weight file and a single-file SDXL checkpoint are downloadable from the
	exact same kinds of platforms (Civitai, direct URL, local path, HF repo_id).
	"""


def resolve_model_source(
	source: str, cache_dir: Path, civitai_api_key: str | None, *, subdir: str = "downloads"
) -> str:
	"""Resolve ``source`` into something directly consumable (a local file path
	string, or an unchanged Hugging Face repo_id).

	Three cases: an existing local path is returned as-is; a Civitai model-page
	URL is resolved via Civitai's API to the real file and downloaded; any other
	http(s) URL is downloaded directly. Anything else (no scheme, not a local
	path) is assumed to already be a Hugging Face repo_id and is returned
	unchanged -- diffusers/huggingface_hub resolve those natively.

	``subdir`` controls which cache_dir subfolder downloaded files land in
	(e.g. "loras" vs "checkpoints") so the two don't collide.
	"""
	local_path = Path(source)
	if local_path.exists():
		return str(local_path)

	parsed = urlparse(source)
	if parsed.scheme in ("http", "https"):
		download_url, filename = _resolve_download_url(source, civitai_api_key)
		return str(_download(download_url, filename, cache_dir, civitai_api_key, subdir))

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
			raise ModelSourceError(f"Civitai model at {source!r} has no downloadable files")
		primary = next((f for f in files if f.get("primary")), files[0])
		return primary["downloadUrl"], primary.get("name", "")

	# Generic direct-download URL (any other platform).
	return source, ""


def _fetch_json(url: str, civitai_api_key: str | None) -> dict:
	"""Try without a key first (works for public models, and avoids sending a
	possibly-stale/invalid key), then only retry with the Authorization header
	if that fails and a key is actually available -- covers gated models that
	need auth without penalizing the common public-model case."""
	try:
		return _fetch_json_once(url, None)
	except ModelSourceError:
		if not civitai_api_key:
			raise
		return _fetch_json_once(url, civitai_api_key)


def _fetch_json_once(url: str, civitai_api_key: str | None) -> dict:
	request = urllib.request.Request(url)
	if civitai_api_key:
		request.add_header("Authorization", f"Bearer {civitai_api_key}")
	try:
		with _OPENER.open(request, timeout=30) as response:
			return json.loads(response.read())
	except Exception as exc:
		raise ModelSourceError(f"failed to query {url!r}: {exc}") from exc


def _download(url: str, filename: str, cache_dir: Path, civitai_api_key: str | None, subdir: str) -> Path:
	"""Same no-key-first, retry-with-key-on-failure strategy as _fetch_json --
	only meaningful for Civitai URLs, since a key is never attached for any
	other host in the first place."""
	try:
		return _download_once(url, filename, cache_dir, None, subdir)
	except ModelSourceError:
		if not (civitai_api_key and "civitai.com" in url):
			raise
		return _download_once(url, filename, cache_dir, civitai_api_key, subdir)


def _download_once(url: str, filename: str, cache_dir: Path, civitai_api_key: str | None, subdir: str) -> Path:
	request = urllib.request.Request(url)
	if civitai_api_key:
		request.add_header("Authorization", f"Bearer {civitai_api_key}")
	try:
		with _OPENER.open(request, timeout=300) as response:
			resolved_filename = filename or _filename_from_response(response, url)
			target_dir = cache_dir / subdir
			target_dir.mkdir(parents=True, exist_ok=True)
			target_path = target_dir / resolved_filename
			if not target_path.exists():
				with open(target_path, "wb") as file_obj:
					file_obj.write(response.read())
			return target_path
	except Exception as exc:
		raise ModelSourceError(f"failed to download model from {url!r}: {exc}") from exc


def _filename_from_response(response, url: str) -> str:
	content_disposition = response.headers.get("Content-Disposition", "")
	match = _CONTENT_DISPOSITION_FILENAME_RE.search(content_disposition)
	if match:
		return match.group(1)
	basename = Path(urlparse(url).path).name
	if basename:
		return basename
	return hashlib.sha1(url.encode()).hexdigest() + ".safetensors"
