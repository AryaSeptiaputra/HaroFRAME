"""Interactive CLI for installing SDXL checkpoints/LoRAs, then writing a prompt
and picking which of the installed models to actually use, before *submitting*
either a video (image2video) or a single transformed image (image2image) job
to a background queue -- a tanya-jawab wrapper around the same pipelines
generate_video.py/generate_image.py use. Needs real model weights and a GPU --
run this on vast.ai (see VAST_GUIDE.md), not on the local dev machine.

Flow: (0) enter platform API keys (Hugging Face token, Civitai API key) if the
sources you're about to install need auth -- input is hidden via getpass.
Then (1) install one or more SDXL checkpoints and (2) install one or more
LoRAs -- these two steps just build up a pool of what's available, they don't
select what's used yet. Both accept the same four kinds of source: a Civitai
link (model page or direct download), a generic direct-download URL, a local
path, or a Hugging Face repo_id (checkpoints use snapshot_download for
repo_ids; LoRAs are left for diffusers to resolve at load time). From there
you land on a main menu that loops: (a) submit a new job -- pick reference
photo, write prompt, pick generation mode (image2video/image2image),
mode-specific overrides, select which installed checkpoint/LoRA(s) to use,
confirm -- the job goes on a queue and control returns to the menu
immediately (generation itself runs on a background worker thread, one job at
a time); (b) view queue status (per-job state -- queued/running/done/failed,
plus per-frame progress for image2video jobs); (c) exit (warns if jobs are
still queued/running, since exiting kills the worker thread).

All choices made here are for this run only -- nothing is written back to
.env. Motion mode and which face adapter (IP-Adapter/InstantID) is active are
still controlled via env vars, not this script.

Usage:
    python scripts/interactive_generate.py
"""

from __future__ import annotations

import dataclasses
import getpass
import queue
import random
import sys
import threading
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image
from pydantic import ValidationError

from app.core.config import GenerationConfig, IdentityConfig, LoraConfig, LoraEntryConfig, get_settings
from app.identity.factory import build_identity_engine
from app.identity.instantid.provider import InstantIdProvider
from app.identity.interfaces import IdentityReference
from app.generation.factory import build_frame_renderer, build_garment_renderer, build_generation_pipeline
from app.generation.interfaces import CameraMotionSpec, GenerationRequest
from app.generation.source_resolver import ModelSourceError, resolve_model_source

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
_TEST_IMAGES_DIR = Path("test_images")
_MAX_SEED = 2**31 - 1


def _prompt_text(label: str, *, required: bool = False, default: str = "") -> str:
	suffix = f" [{default}]" if default else ""
	while True:
		value = input(f"{label}{suffix}: ").strip()
		if not value:
			if required:
				print("  (wajib diisi, jangan kosong)")
				continue
			return default
		return value


def _prompt_float(label: str, default: float) -> float:
	while True:
		raw = input(f"{label} [{default}]: ").strip()
		if not raw:
			return default
		try:
			return float(raw)
		except ValueError:
			print("  (masukkan angka, contoh: 0.6)")


def _prompt_optional_int(label: str) -> int | None:
	raw = input(f"{label} [Enter = default dari config]: ").strip()
	if not raw:
		return None
	try:
		return int(raw)
	except ValueError:
		print("  (bukan angka, dilewati)")
		return None


def _prompt_int(label: str, default: int) -> int:
	while True:
		raw = input(f"{label} [{default}]: ").strip()
		if not raw:
			return default
		try:
			return int(raw)
		except ValueError:
			print("  (masukkan bilangan bulat, contoh: 30)")


def _prompt_yes_no(label: str, default: bool) -> bool:
	marker = "Y/n" if default else "y/N"
	raw = input(f"{label} [{marker}]: ").strip().lower()
	if not raw:
		return default
	return raw == "y"


def _choose_reference_image() -> Path:
	folder_input = input(f"\nFolder foto referensi [{_TEST_IMAGES_DIR}]: ").strip()
	folder = Path(folder_input) if folder_input else _TEST_IMAGES_DIR

	candidates = []
	if folder.is_dir():
		candidates = sorted(p for p in folder.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS)

	if candidates:
		print(f"Foto ditemukan di {folder}/:")
		for idx, path in enumerate(candidates, start=1):
			print(f"  [{idx}] {path.name}")
		print("  [0] Ketik path file lain")
		while True:
			choice = input("Pilih nomor: ").strip()
			if choice == "0":
				break
			if choice.isdigit() and 1 <= int(choice) <= len(candidates):
				return candidates[int(choice) - 1]
			print("  (nomor tidak valid)")
	else:
		print(f"  (folder {folder} tidak ditemukan atau tidak berisi foto)")

	while True:
		typed = input("Path foto referensi: ").strip()
		path = Path(typed)
		if path.is_file():
			return path
		print(f"  (file tidak ditemukan: {path})")


def _print_loras(entries: list[LoraEntryConfig]) -> None:
	if not entries:
		print("  (belum ada LoRA)")
		return
	for idx, entry in enumerate(entries, start=1):
		print(f"  [{idx}] {entry.adapter_name} <- {entry.source} (scale={entry.scale})")


# ---------------------------------------------------------------------------
# Step 0: configure platform API keys (HuggingFace, Civitai) for this session
# ---------------------------------------------------------------------------


def _prompt_secret_optional(label: str, current: str | None) -> str | None:
	status = "sudah diset dari .env" if current else "belum diset"
	prompt = f"{label} [{status}, Enter = pakai ini]: "
	# getpass hides typed characters in a real terminal -- appropriate for
	# tokens/API keys. On POSIX, getpass() itself falls back to a plain (echoed)
	# read when stdin isn't a tty, but on Windows it always reads via the raw
	# console (msvcrt) regardless of stdin, which hangs when stdin is piped/
	# redirected instead of falling back. Checking isatty() ourselves keeps
	# this correct (and testable) on both platforms.
	raw = (getpass.getpass(prompt) if sys.stdin.isatty() else input(prompt)).strip()
	if not raw:
		return current
	return raw


def _configure_api_keys_interactive(default_hf_token: str | None, default_civitai_api_key: str | None) -> tuple[str | None, str | None]:
	print("\n=== 0. API Key Platform ===")
	print("(dipakai untuk instalasi checkpoint/LoRA dari platform yang butuh auth; hanya untuk sesi ini, tidak ditulis ke .env)")
	hf_token = _prompt_secret_optional("Hugging Face token", default_hf_token)
	civitai_api_key = _prompt_secret_optional("Civitai API key", default_civitai_api_key)
	return hf_token, civitai_api_key


# ---------------------------------------------------------------------------
# Step 1: install SDXL checkpoints (pool -- not selection)
# ---------------------------------------------------------------------------


def _install_checkpoint_source(
	source: str, cache_dir: Path, hf_token: str | None, civitai_api_key: str | None
) -> str | None:
	"""Install one checkpoint source and return whatever base_sdxl_model should
	be set to for it: a local single-file path (Civitai/direct URL/existing
	local file) or the repo_id unchanged (Hugging Face, downloaded via
	snapshot_download so it's warmed in cache_dir ahead of time)."""
	if urlparse(source).scheme in ("http", "https") or Path(source).exists():
		print(f"  Mengunduh/memverifikasi {source} ...")
		try:
			resolved_path = resolve_model_source(source, cache_dir, civitai_api_key, subdir="checkpoints")
		except ModelSourceError as exc:
			print(f"  Gagal mengunduh: {exc}")
			return None
		print(f"  Berhasil (checkpoint single-file): {resolved_path}")
		return resolved_path

	print(f"  Mengunduh repo {source} ...")
	try:
		from huggingface_hub import snapshot_download

		snapshot_download(repo_id=source, cache_dir=cache_dir, token=hf_token)
	except Exception as exc:
		print(f"  Gagal mengunduh: {exc}")
		return None
	print(f"  Berhasil: {source}")
	return source


def _install_checkpoints_interactive(
	cache_dir: Path, hf_token: str | None, civitai_api_key: str | None, default_checkpoint: str
) -> list[str]:
	print("\n=== 1. Instalasi SDXL Checkpoint ===")
	print(f"  [terpasang] {default_checkpoint} (default dari config)")
	checkpoints = [default_checkpoint]
	while True:
		print("[1] Install checkpoint baru  [2] Selesai")
		choice = input("Pilih: ").strip()
		if choice == "1":
			source = _prompt_text(
				"Link/path checkpoint (link Civitai, link download langsung, path lokal, atau repo_id HuggingFace)",
				required=True,
			)
			if source in checkpoints:
				print("  (checkpoint ini sudah terpasang)")
				continue
			resolved = _install_checkpoint_source(source, cache_dir, hf_token, civitai_api_key)
			if resolved is not None:
				checkpoints.append(resolved)
		elif choice == "2":
			return checkpoints
		else:
			print("  (pilihan tidak dikenal)")


# ---------------------------------------------------------------------------
# Step 2: install LoRAs (pool -- not selection/activation)
# ---------------------------------------------------------------------------


def _install_lora_interactive(cache_dir: Path, civitai_api_key: str | None) -> LoraEntryConfig | None:
	print("\n-- Install LoRA --")
	adapter_name = _prompt_text("Nama adapter (bebas, unik, bukan 'faceid')", required=True)
	source = _prompt_text(
		"Link/path LoRA (link Civitai, link download langsung, path lokal, atau repo_id HuggingFace)",
		required=True,
	)
	scale = _prompt_float(
		"Bobot/scale LoRA (0-1, disarankan 0.5-0.8; makin tinggi = gaya LoRA makin dominan)", default=0.6
	)
	weight_name = _prompt_text("Nama file weight (Enter jika tidak perlu)") or None
	subfolder = _prompt_text("Subfolder repo (Enter jika tidak perlu)") or None

	try:
		entry = LoraEntryConfig(
			adapter_name=adapter_name, source=source, scale=scale, weight_name=weight_name, subfolder=subfolder
		)
	except ValidationError as exc:
		print(f"  LoRA ditolak: {exc}")
		return None

	if urlparse(source).scheme in ("http", "https"):
		print("  Mengunduh/memverifikasi dari link...")
		try:
			resolved_path = resolve_model_source(source, cache_dir, civitai_api_key, subdir="loras")
		except ModelSourceError as exc:
			print(f"  Gagal mengunduh: {exc}")
			return None
		print(f"  Berhasil: {resolved_path}")
	else:
		print("  (path lokal / repo_id HuggingFace -- diverifikasi otomatis saat generate)")

	return entry


def _install_loras_interactive(
	cache_dir: Path, civitai_api_key: str | None, existing_entries: list[LoraEntryConfig]
) -> list[LoraEntryConfig]:
	print("\n=== 2. Instalasi LoRA ===")
	entries = list(existing_entries)
	while True:
		_print_loras(entries)
		print("[1] Install LoRA baru  [2] Selesai")
		choice = input("Pilih: ").strip()
		if choice == "1":
			new_entry = _install_lora_interactive(cache_dir, civitai_api_key)
			if new_entry is not None:
				entries.append(new_entry)
		elif choice == "2":
			return entries
		else:
			print("  (pilihan tidak dikenal)")


# ---------------------------------------------------------------------------
# Step 5.5: select which installed checkpoint/LoRA(s) to actually use
# ---------------------------------------------------------------------------


def _select_checkpoint(installed_checkpoints: list[str]) -> str:
	print("\n=== Pilih SDXL Checkpoint ===")
	if len(installed_checkpoints) == 1:
		print(f"  (cuma ada satu terinstall, otomatis dipakai: {installed_checkpoints[0]})")
		return installed_checkpoints[0]
	for idx, repo_id in enumerate(installed_checkpoints, start=1):
		print(f"  [{idx}] {repo_id}")
	while True:
		choice = input("Pilih nomor: ").strip()
		if choice.isdigit() and 1 <= int(choice) <= len(installed_checkpoints):
			return installed_checkpoints[int(choice) - 1]
		print("  (nomor tidak valid)")


def _select_loras(installed_loras: list[LoraEntryConfig], default_max: int) -> tuple[list[LoraEntryConfig], int]:
	print("\n=== Pilih LoRA Aktif ===")
	if not installed_loras:
		print("  (belum ada LoRA terinstall, dilewati)")
		return [], default_max

	_print_loras(installed_loras)
	while True:
		raw = input(f"Jumlah maksimal LoRA aktif [{default_max}]: ").strip()
		if not raw:
			max_active = default_max
			break
		if raw.isdigit():
			max_active = int(raw)
			break
		print("  (masukkan angka bulat)")

	while True:
		raw = input("Pilih nomor LoRA yang mau diaktifkan (pisah koma, kosong = tidak pakai): ").strip()
		if not raw:
			return [], max_active
		try:
			indices = [int(part.strip()) for part in raw.split(",") if part.strip()]
		except ValueError:
			print("  (format tidak valid, pisahkan nomor dengan koma, mis. 1,2)")
			continue
		if any(i < 1 or i > len(installed_loras) for i in indices):
			print("  (ada nomor di luar jangkauan)")
			continue
		if len(indices) > max_active:
			print(f"  (melebihi batas maksimal {max_active}, pilih lebih sedikit)")
			continue
		return [installed_loras[i - 1] for i in indices], max_active


# ---------------------------------------------------------------------------
# Step 4.5: pick generation mode (image2video vs image2image)
# ---------------------------------------------------------------------------


def _select_generation_mode() -> str:
	print("\n=== Mode Generate ===")
	print("[1] Image2Video (animasi, motion dari config)")
	print("[2] Image2Image (satu gambar hasil, tanpa motion)")
	print("[3] Garment-Swap (ganti pakaian, SAM-based inpaint)")
	while True:
		choice = input("Pilih: ").strip()
		if choice == "1":
			return "i2v"
		if choice == "2":
			return "i2i"
		if choice == "3":
			return "garment"
		print("  (pilihan tidak dikenal)")


# ---------------------------------------------------------------------------
# Generation queue: submit returns immediately, jobs run one at a time on a
# background worker thread -- lets you keep submitting/checking status
# without waiting for each generate() call to finish. In-process only (not
# persisted); exiting the script drops any queued/running job.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class GenerationJob:
	job_id: int
	mode: str  # "i2v", "i2i", or "garment"
	reference_image: Path
	prompt: str
	negative_prompt: str
	seed: int | None
	frames_override: int | None
	strength_override: float | None
	output_path: Path
	identity_config: IdentityConfig
	generation_config: GenerationConfig
	garment_prompt: str | None = None
	garment_strength_override: float | None = None
	status: str = "queued"  # queued, running, done, failed
	current_frame: int = 0
	total_frames: int = 0
	result_path: Path | None = None
	error: str | None = None

	def status_line(self) -> str:
		if self.status == "running" and self.mode == "i2v" and self.total_frames:
			detail = f"frame {self.current_frame}/{self.total_frames}"
		elif self.status == "done":
			detail = f"selesai -> {self.result_path}"
		elif self.status == "failed":
			detail = f"error: {self.error}"
		else:
			detail = self.status
		return f"[{self.job_id}] {self.mode} {self.reference_image.name} -- {detail}"


class GenerationQueueManager:
	"""Runs GenerationJobs one at a time on a single background thread.

	Each job builds its own IdentityEngine/renderer (from that job's own
	identity_config/generation_config) rather than sharing one across jobs --
	FaceIdSdxlProvider.load() (and similar) tracks "already loaded" as instance
	state on the adapter, keyed by nothing but a bool, so reusing one
	IdentityEngine across jobs whose selected checkpoint/LoRA differ would
	silently skip re-attaching the adapter to each job's distinct pipeline
	object. Rebuilding is cheap (no model weights load at construction time).
	"""

	def __init__(self) -> None:
		self._pending: queue.Queue[GenerationJob] = queue.Queue()
		self._jobs: list[GenerationJob] = []
		self._lock = threading.Lock()
		self._worker = threading.Thread(target=self._run, daemon=True)
		self._worker.start()

	def submit(self, job: GenerationJob) -> None:
		with self._lock:
			self._jobs.append(job)
		self._pending.put(job)

	def pending_count(self) -> int:
		with self._lock:
			return sum(1 for job in self._jobs if job.status in ("queued", "running"))

	def status_lines(self) -> list[str]:
		with self._lock:
			jobs = list(self._jobs)
		return [job.status_line() for job in jobs]

	def _run(self) -> None:
		while True:
			job = self._pending.get()
			job.status = "running"
			try:
				identity_engine = build_identity_engine(job.identity_config)
				if job.mode == "i2v":
					self._run_i2v(job, identity_engine)
				elif job.mode == "i2i":
					self._run_i2i(job, identity_engine)
				else:
					self._run_garment(job, identity_engine)
				job.status = "done"
			except Exception as exc:  # noqa: BLE001 -- report any failure on the job, don't crash the worker
				job.status = "failed"
				job.error = str(exc)

	def _run_i2v(self, job: GenerationJob, identity_engine) -> None:
		generation_pipeline = build_generation_pipeline(job.generation_config, job.identity_config, identity_engine)
		reference = IdentityReference(images=[Image.open(job.reference_image).convert("RGB")])
		output_cfg = job.generation_config.output
		motion_cfg = job.generation_config.motion
		num_frames = (
			job.frames_override if job.frames_override is not None else int(output_cfg.fps * output_cfg.duration_seconds)
		)
		job.total_frames = num_frames
		spec = CameraMotionSpec(
			mode=motion_cfg.mode,
			direction=motion_cfg.direction,
			zoom_range=motion_cfg.zoom_range,
			pan_fraction=motion_cfg.pan_fraction,
			easing=motion_cfg.easing,
			num_frames=num_frames,
			fps=output_cfg.fps,
		)
		request = GenerationRequest(
			reference=reference, prompt=job.prompt, negative_prompt=job.negative_prompt, motion=spec, seed=job.seed
		)

		def on_progress(done: int, total: int) -> None:
			job.current_frame = done
			job.total_frames = total

		result = generation_pipeline.generate(request, output_path=job.output_path, progress_callback=on_progress)
		job.result_path = result.output_path

	def _run_i2i(self, job: GenerationJob, identity_engine) -> None:
		renderer = build_frame_renderer(identity_engine, job.identity_config, job.generation_config)
		source_image = Image.open(job.reference_image).convert("RGB")
		reference = IdentityReference(images=[source_image])
		seed = job.seed if job.seed is not None else random.randint(0, _MAX_SEED)

		identity_engine.prepare_reference(reference)
		rendered = renderer.render(
			source_image,
			reference=reference,
			prompt=job.prompt,
			negative_prompt=job.negative_prompt,
			seed=seed,
			frame_index=0,
			strength=job.strength_override,
		)
		job.output_path.parent.mkdir(parents=True, exist_ok=True)
		rendered.image.save(job.output_path)
		job.result_path = job.output_path

	def _run_garment(self, job: GenerationJob, identity_engine) -> None:
		renderer = build_garment_renderer(identity_engine, job.identity_config, job.generation_config)
		source_image = Image.open(job.reference_image).convert("RGB")
		reference = IdentityReference(images=[source_image])
		seed = job.seed if job.seed is not None else random.randint(0, _MAX_SEED)

		identity_engine.prepare_reference(reference)
		rendered = renderer.render_garment_swap(
			source_image,
			reference=reference,
			garment_prompt=job.garment_prompt,
			negative_prompt=job.negative_prompt,
			seed=seed,
			strength=job.garment_strength_override,
		)
		job.output_path.parent.mkdir(parents=True, exist_ok=True)
		rendered.image.save(job.output_path)
		job.result_path = job.output_path


def _configure_job_interactive(
	job_id: int,
	settings,
	installed_checkpoints: list[str],
	installed_loras: list[LoraEntryConfig],
	is_ipadapter_branch: bool,
) -> GenerationJob | None:
	"""Steps 3-6: photo, prompt, mode, mode-specific overrides, checkpoint/LoRA
	selection, summary + confirm. Returns a GenerationJob ready to enqueue, or
	None if the user cancels at the confirmation prompt."""
	reference_image = _choose_reference_image()
	prompt = _prompt_text("\nPrompt", required=True)
	negative_prompt = _prompt_text("Negative prompt (opsional)")

	mode = _select_generation_mode()
	if mode == "garment" and not is_ipadapter_branch:
		print(
			"Garment-Swap butuh IP-Adapter/FaceID-SDXL aktif -- InstantID tidak didukung "
			"(vendored pipeline tidak punya entry point inpaint/img2img)."
		)
		return None

	seed_override = _prompt_optional_int("Seed")
	garment_prompt = None
	garment_strength_override = None
	if mode == "i2v":
		frames_override = _prompt_optional_int("Jumlah frame")
		strength_override = None
		default_output_name = f"{reference_image.stem}.{settings.generation.output.format}"
	elif mode == "i2i":
		frames_override = None
		strength_override = _prompt_float(
			"Strength img2img (0-1, disarankan 0.2-0.5; makin tinggi = makin jauh dari foto asli; "
			"khusus IP-Adapter, diabaikan InstantID)",
			default=settings.generation.render.strength,
		)
		default_output_name = f"{reference_image.stem}_img2img.png"
	else:  # garment
		frames_override = None
		strength_override = None
		garment_prompt = _prompt_text(
			"Deskripsi pakaian baru (target outfit, mis. 'sleeveless summer top, light shorts')",
			required=True,
		)
		default_output_name = f"{reference_image.stem}_garment.png"

	print("\n--- Parameter Render (opsional, Enter = default dari config) ---")
	guidance_scale = _prompt_float(
		"Guidance scale/CFG (disarankan 4-8; makin tinggi = makin taat ke prompt, terlalu tinggi bisa oversaturate)",
		default=settings.generation.render.guidance_scale,
	)
	num_inference_steps = _prompt_int(
		"Jumlah inference steps (disarankan 20-50; makin tinggi = detail lebih baik tapi lebih lambat)",
		default=settings.generation.render.num_inference_steps,
	)

	pose_enabled = settings.identity.controlnet.pose_enabled
	pose_scale = settings.identity.controlnet.pose_conditioning_scale
	depth_enabled = settings.identity.controlnet.depth_enabled
	depth_scale = settings.identity.controlnet.depth_conditioning_scale
	if is_ipadapter_branch and mode != "garment":
		print("\n--- Structure ControlNet (opsional, khusus IP-Adapter) ---")
		pose_enabled = _prompt_yes_no("Aktifkan pose ControlNet (DWPose/OpenPose)?", default=pose_enabled)
		if pose_enabled:
			pose_scale = _prompt_float(
				"  Pose conditioning scale (0-1, disarankan 0.4-0.8; makin tinggi = pose makin ketat "
				"mengikuti foto asli)",
				default=pose_scale,
			)
		depth_enabled = _prompt_yes_no("Aktifkan depth ControlNet?", default=depth_enabled)
		if depth_enabled:
			depth_scale = _prompt_float(
				"  Depth conditioning scale (0-1, disarankan 0.4-0.8; makin tinggi = struktur 3D makin "
				"ketat mengikuti foto asli)",
				default=depth_scale,
			)

	garment_inpaint_strength = settings.generation.garment.inpaint_strength
	garment_mask_dilation_px = settings.generation.garment.mask_dilation_px
	garment_include_legs = settings.generation.garment.include_legs_in_mask
	if mode == "garment":
		print("\n--- Parameter Garment-Swap (opsional, Enter = default dari config) ---")
		garment_inpaint_strength = _prompt_float(
			"Inpaint strength (0-1, disarankan 0.7-0.95; rendah = baju baru kurang menyatu, tinggi = "
			"perubahan lebih total tapi risiko tepi mask meleset)",
			default=garment_inpaint_strength,
		)
		garment_mask_dilation_px = _prompt_int(
			"Mask dilation px (disarankan 20-60; naikkan kalau kulit yang baru terbuka masih kelihatan "
			"sisa baju lama)",
			default=garment_mask_dilation_px,
		)
		garment_include_legs = _prompt_yes_no("Termasuk area kaki dalam mask?", default=garment_include_legs)
		garment_strength_override = garment_inpaint_strength

	output_name = _prompt_text("Nama file output", default=default_output_name)

	base_model = _select_checkpoint(installed_checkpoints)
	lora_entries, max_active_loras = _select_loras(installed_loras, settings.generation.lora.max_active_loras)
	try:
		LoraConfig(max_active_loras=max_active_loras, entries=lora_entries)
	except ValidationError as exc:
		print(f"Konfigurasi LoRA tidak valid: {exc}")
		return None

	identity_updates: dict = {}
	if base_model != settings.identity.base_sdxl_model:
		identity_updates["base_sdxl_model"] = base_model
	if is_ipadapter_branch:
		new_controlnet = settings.identity.controlnet.model_copy(
			update={
				"pose_enabled": pose_enabled,
				"pose_conditioning_scale": pose_scale,
				"depth_enabled": depth_enabled,
				"depth_conditioning_scale": depth_scale,
			}
		)
		if new_controlnet != settings.identity.controlnet:
			identity_updates["controlnet"] = new_controlnet
	identity_config = settings.identity if not identity_updates else settings.identity.model_copy(update=identity_updates)

	new_render_config = settings.generation.render.model_copy(
		update={"guidance_scale": guidance_scale, "num_inference_steps": num_inference_steps}
	)
	generation_config = settings.generation.model_copy(
		update={
			"lora": LoraConfig(max_active_loras=max_active_loras, entries=lora_entries),
			"render": new_render_config,
		}
	)
	if mode == "garment":
		new_garment_config = settings.generation.garment.model_copy(
			update={
				"inpaint_strength": garment_inpaint_strength,
				"mask_dilation_px": garment_mask_dilation_px,
				"include_legs_in_mask": garment_include_legs,
			}
		)
		generation_config = generation_config.model_copy(update={"garment": new_garment_config})

	mode_label = {"i2v": "Image2Video", "i2i": "Image2Image", "garment": "Garment-Swap"}[mode]
	print("\n=== Ringkasan Job ===")
	print(f"Mode      : {mode_label}")
	print(f"Base model: {base_model}")
	print(f"Foto      : {reference_image}")
	print(f"Prompt    : {prompt}")
	print(f"Negative  : {negative_prompt or '(kosong)'}")
	print(f"LoRA aktif: {len(lora_entries)} ({', '.join(e.adapter_name for e in lora_entries) or '-'})")
	if mode == "i2v":
		print(f"Frame     : {frames_override if frames_override is not None else 'default dari config'}")
	elif mode == "i2i":
		print(f"Strength  : {strength_override}")
	else:
		print(f"Outfit    : {garment_prompt}")
		print(f"Inpaint strength: {garment_inpaint_strength}")
		print(f"Mask dilation px: {garment_mask_dilation_px}")
		print(f"Termasuk kaki   : {'ya' if garment_include_legs else 'tidak'}")
	print(f"Guidance  : {guidance_scale}")
	print(f"Steps     : {num_inference_steps}")
	if is_ipadapter_branch and mode != "garment":
		print(f"Pose CN   : {'on scale=' + str(pose_scale) if pose_enabled else 'off'}")
		print(f"Depth CN  : {'on scale=' + str(depth_scale) if depth_enabled else 'off'}")
	print(f"Output    : outputs/{output_name}")
	confirm = input("Tambahkan ke antrian? [y/N]: ").strip().lower()
	if confirm != "y":
		print("Dibatalkan.")
		return None

	output_dir = generation_config.output.output_dir
	output_dir.mkdir(parents=True, exist_ok=True)

	return GenerationJob(
		job_id=job_id,
		mode=mode,
		reference_image=reference_image,
		prompt=prompt,
		negative_prompt=negative_prompt,
		seed=seed_override,
		frames_override=frames_override,
		strength_override=strength_override,
		output_path=output_dir / output_name,
		identity_config=identity_config,
		generation_config=generation_config,
		garment_prompt=garment_prompt,
		garment_strength_override=garment_strength_override,
	)


def main() -> int:
	settings = get_settings()
	print(f"device={settings.identity.device} dtype={settings.identity.dtype}")

	identity_engine = build_identity_engine(settings.identity)
	if identity_engine.face_adapter is None:
		print("FAIL: no face adapter enabled (identity.ipadapter.enabled / identity.instantid.enabled)")
		return 1
	print(f"face adapter aktif : {type(identity_engine.face_adapter).__name__}")
	is_ipadapter_branch = not isinstance(identity_engine.face_adapter, InstantIdProvider)

	default_hf_token = settings.identity.hf_token.get_secret_value() if settings.identity.hf_token else None
	default_civitai_api_key = (
		settings.generation.lora.civitai_api_key.get_secret_value()
		if settings.generation.lora.civitai_api_key
		else None
	)
	hf_token, civitai_api_key = _configure_api_keys_interactive(default_hf_token, default_civitai_api_key)

	installed_checkpoints = _install_checkpoints_interactive(
		settings.identity.cache_dir, hf_token, civitai_api_key, settings.identity.base_sdxl_model
	)
	installed_loras = _install_loras_interactive(
		settings.identity.cache_dir, civitai_api_key, list(settings.generation.lora.entries)
	)

	manager = GenerationQueueManager()
	next_job_id = 1

	while True:
		print("\n=== Menu Utama ===")
		print("[1] Submit job generate baru")
		print("[2] Lihat status antrian")
		print("[3] Keluar")
		choice = input("Pilih: ").strip()
		if choice == "1":
			job = _configure_job_interactive(
				next_job_id, settings, installed_checkpoints, installed_loras, is_ipadapter_branch
			)
			if job is not None:
				manager.submit(job)
				print(f"Job #{job.job_id} ditambahkan ke antrian.")
				next_job_id += 1
		elif choice == "2":
			lines = manager.status_lines()
			if not lines:
				print("(belum ada job)")
			for line in lines:
				print(line)
		elif choice == "3":
			pending = manager.pending_count()
			if pending:
				confirm = input(
					f"Masih ada {pending} job berjalan/menunggu -- job yang belum selesai akan dibatalkan "
					"kalau keluar sekarang. Tetap keluar? [y/N]: "
				).strip().lower()
				if confirm != "y":
					continue
			print("Keluar.")
			return 0
		else:
			print("  (pilihan tidak dikenal)")


if __name__ == "__main__":
	sys.exit(main())
