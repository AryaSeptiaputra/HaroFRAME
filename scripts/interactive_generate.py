"""Interactive CLI for writing a prompt and configuring LoRAs before generating
a video -- a tanya-jawab wrapper around the same pipeline generate_video.py
uses, for when you want to try several LoRA combinations without editing .env
each time. Needs real model weights and a GPU -- run this on vast.ai (see
VASTAI.md), not on the local dev machine.

LoRA choices made here are for this run only -- they are not written back to
.env. Motion mode and which face adapter (IP-Adapter/InstantID) is active are
still controlled via env vars, not this script.

Usage:
    python scripts/interactive_generate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image
from pydantic import ValidationError

from app.core.config import LoraConfig, LoraEntryConfig, get_settings
from app.identity.exceptions import IdentityModuleError
from app.identity.factory import build_identity_engine
from app.identity.interfaces import IdentityReference
from app.generation.exceptions import GenerationModuleError
from app.generation.factory import build_generation_pipeline
from app.generation.interfaces import CameraMotionSpec, GenerationRequest

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
_TEST_IMAGES_DIR = Path("test_images")


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


def _choose_reference_image() -> Path:
	candidates = []
	if _TEST_IMAGES_DIR.is_dir():
		candidates = sorted(p for p in _TEST_IMAGES_DIR.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS)

	if candidates:
		print(f"\nFoto ditemukan di {_TEST_IMAGES_DIR}/:")
		for idx, path in enumerate(candidates, start=1):
			print(f"  [{idx}] {path.name}")
		print("  [0] Ketik path lain")
		while True:
			choice = input("Pilih nomor: ").strip()
			if choice == "0":
				break
			if choice.isdigit() and 1 <= int(choice) <= len(candidates):
				return candidates[int(choice) - 1]
			print("  (nomor tidak valid)")

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
		status = "aktif" if entry.enabled else "nonaktif"
		print(f"  [{idx}] {entry.adapter_name} <- {entry.source} (scale={entry.scale}, {status})")


def _add_lora_interactive() -> LoraEntryConfig | None:
	print("\n-- Tambah LoRA --")
	adapter_name = _prompt_text("Nama adapter (bebas, unik, bukan 'faceid')", required=True)
	source = _prompt_text(
		"Sumber (path lokal / link download / link Civitai / repo_id HuggingFace)", required=True
	)
	scale = _prompt_float("Bobot/scale", default=0.6)
	weight_name = _prompt_text("Nama file weight (Enter jika tidak perlu)") or None
	subfolder = _prompt_text("Subfolder repo (Enter jika tidak perlu)") or None
	try:
		return LoraEntryConfig(
			adapter_name=adapter_name, source=source, scale=scale, weight_name=weight_name, subfolder=subfolder
		)
	except ValidationError as exc:
		print(f"  LoRA ditolak: {exc}")
		return None


def _manage_loras(existing_entries: list[LoraEntryConfig], max_active_loras: int) -> list[LoraEntryConfig]:
	entries = list(existing_entries)
	print("\n=== Konfigurasi LoRA ===")
	print(f"(maksimal {max_active_loras} LoRA aktif sekaligus)")
	while True:
		_print_loras(entries)
		print("[1] Tambah LoRA  [2] Hapus LoRA  [3] Lihat daftar  [4] Selesai, lanjut")
		choice = input("Pilih: ").strip()
		if choice == "1":
			new_entry = _add_lora_interactive()
			if new_entry is not None:
				entries.append(new_entry)
		elif choice == "2":
			if not entries:
				print("  (belum ada LoRA untuk dihapus)")
				continue
			idx_raw = input("Nomor yang mau dihapus: ").strip()
			if idx_raw.isdigit() and 1 <= int(idx_raw) <= len(entries):
				removed = entries.pop(int(idx_raw) - 1)
				print(f"  Dihapus: {removed.adapter_name}")
			else:
				print("  (nomor tidak valid)")
		elif choice == "3":
			continue
		elif choice == "4":
			try:
				LoraConfig(max_active_loras=max_active_loras, entries=entries)
			except ValidationError as exc:
				print(f"  Tidak bisa lanjut: {exc}")
				continue
			return entries
		else:
			print("  (pilihan tidak dikenal)")


def main() -> int:
	settings = get_settings()
	print(f"device={settings.identity.device} dtype={settings.identity.dtype}")

	identity_engine = build_identity_engine(settings.identity)
	if identity_engine.face_adapter is None:
		print("FAIL: no face adapter enabled (identity.ipadapter.enabled / identity.instantid.enabled)")
		return 1
	print(f"face adapter aktif: {type(identity_engine.face_adapter).__name__}")

	reference_image = _choose_reference_image()
	prompt = _prompt_text("\nPrompt", required=True)
	negative_prompt = _prompt_text("Negative prompt (opsional)")

	lora_entries = _manage_loras(list(settings.generation.lora.entries), settings.generation.lora.max_active_loras)

	frames_override = _prompt_optional_int("Jumlah frame")
	seed_override = _prompt_optional_int("Seed")
	output_name = _prompt_text(
		"Nama file output", default=f"{reference_image.stem}.{settings.generation.output.format}"
	)

	generation_config = settings.generation.model_copy(
		update={"lora": LoraConfig(max_active_loras=settings.generation.lora.max_active_loras, entries=lora_entries)}
	)

	print("\n=== Ringkasan ===")
	print(f"Foto      : {reference_image}")
	print(f"Prompt    : {prompt}")
	print(f"Negative  : {negative_prompt or '(kosong)'}")
	print(f"LoRA aktif: {sum(1 for e in lora_entries if e.enabled)}/{len(lora_entries)}")
	print(f"Output    : outputs/{output_name}")
	confirm = input("Lanjut generate? [y/N]: ").strip().lower()
	if confirm != "y":
		print("Dibatalkan.")
		return 0

	generation_pipeline = build_generation_pipeline(generation_config, settings.identity, identity_engine)
	reference = IdentityReference(images=[Image.open(reference_image).convert("RGB")])
	output_cfg = generation_config.output
	motion_cfg = generation_config.motion
	spec = CameraMotionSpec(
		mode=motion_cfg.mode,
		direction=motion_cfg.direction,
		zoom_range=motion_cfg.zoom_range,
		pan_fraction=motion_cfg.pan_fraction,
		easing=motion_cfg.easing,
		num_frames=frames_override if frames_override is not None else int(output_cfg.fps * output_cfg.duration_seconds),
		fps=output_cfg.fps,
	)
	request = GenerationRequest(
		reference=reference, prompt=prompt, negative_prompt=negative_prompt, motion=spec, seed=seed_override
	)
	output_cfg.output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_cfg.output_dir / output_name

	try:
		result = generation_pipeline.generate(request, output_path=output_path)
	except (IdentityModuleError, GenerationModuleError) as exc:
		print(f"FAIL: {exc}")
		return 1

	if result.output_path is None:
		print("FAIL: video encoding did not run")
		return 1

	print(f"\nrendered {len(result.frames)} frames (fps={result.fps}) -> {result.output_path}")
	print("OK")
	return 0


if __name__ == "__main__":
	sys.exit(main())
