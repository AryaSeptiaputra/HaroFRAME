# HaroFRAME -- GPU dev/runtime image untuk vast.ai.
#
# Build & push:
#   docker build -t <registry-user>/haroframe:latest .
#   docker push <registry-user>/haroframe:latest
#
# Panduan lengkap step-by-step: lihat VAST_GUIDE.md di root repo.
#
# Di vast.ai: buat instance baru, isi field "Image Path/Tag" dengan
# <registry-user>/haroframe:latest, pilih instance dengan GPU + driver yang
# support CUDA 12.1+. Container ini idle (sleep infinity) -- connect lewat
# fitur "Instance Portal" / exec bawaan vast.ai, lalu jalankan skrip secara
# manual, contoh:
#   python scripts/smoke_test_identity.py ref1.jpg ref2.jpg
#   python scripts/smoke_test_generation.py ref.jpg "a person smiling" --frames 16
#   python scripts/generate_video.py ref.jpg "a person smiling" --out outputs/clip.mp4
#   python scripts/test_real_images.py --prompt "a person, natural lighting"
#
# test_images/ (foto referensi asli) dan outputs/ (hasil video) sengaja tidak
# di-COPY isinya dari build context yang Anda commit (lihat .gitignore) --
# pakai bind mount saat docker run/compose supaya foto & hasil video hidup di
# host, bukan di dalam image:
#   docker run --gpus all -it \
#     -v $(pwd)/test_images:/workspace/test_images \
#     -v $(pwd)/outputs:/workspace/outputs \
#     <registry-user>/haroframe:latest
# (atau pakai docker-compose.yml yang sudah menyiapkan mount ini -- lihat
# VAST_GUIDE.md untuk cara membawa mount yang sama ke instance vast.ai.)
#
# Env var yang perlu di-set di UI vast.ai saat create instance (lihat
# app/core/config.py -- Settings pakai prefix HAROFRAME_, nested delimiter __):
#   HAROFRAME_IDENTITY__HF_TOKEN=<hugging face token>
#   HAROFRAME_IDENTITY__IPADAPTER__ENABLED=true   (atau INSTANTID__ENABLED=true)
#   dst. sesuai kebutuhan -- lihat IdentityConfig/GenerationConfig di app/core/config.py.
#
# Bobot model TIDAK di-bake ke image ini (ukurannya puluhan GB dan sering
# diganti). Jalankan sekali di dalam container untuk mengunduh base checkpoint,
# LoRA, dan checkpoint SAM sesuai config:
#   python3 scripts/prefetch_models.py
# Jalur on-start-script (entrypoint.sh) melakukan ini otomatis di step 7.
#
# Restoration extra (gfpgan/basicsr/facexlib) SENGAJA tidak di-bake di image ini.
# Kalau dibutuhkan, install manual di dalam container:
#   pip install -e ".[restoration]"
#
# Extra "garment" (segment-anything) SEKARANG ikut di-bake, tidak seperti
# "restoration": tahap inpaint sudah default aktif
# (GenerationConfig.inpaint.enabled), jadi tanpa paket ini image-nya rusak
# begitu dipakai. Bobot checkpoint SAM-nya sendiri (375MB-2.6GB, pilihannya
# vit_b/vit_l/vit_h) tidak di-bake tapi diunduh oleh scripts/prefetch_models.py
# bersama base checkpoint dan LoRA (lihat catatan "Bobot model" di atas).
# Kalau tidak butuh inpaint sama sekali: jalankan skrip dengan --no-inpaint,
# atau set HAROFRAME_GENERATION__INPAINT__ENABLED=false.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
	libgl1 \
	libglib2.0-0 \
	build-essential \
	&& rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /workspace

COPY . /workspace

RUN pip install --no-cache-dir -e ".[gpu,garment]"

CMD ["sleep", "infinity"]
