# HaroFRAME -- GPU dev/runtime image untuk vast.ai.
#
# Build & push:
#   docker build -t <registry-user>/haroframe:latest .
#   docker push <registry-user>/haroframe:latest
#
# Di vast.ai: buat instance baru, isi field "Image Path/Tag" dengan
# <registry-user>/haroframe:latest, pilih instance dengan GPU + driver yang
# support CUDA 12.1+. Container ini idle (sleep infinity) -- connect lewat
# fitur "Instance Portal" / exec bawaan vast.ai, lalu jalankan skrip secara
# manual, contoh:
#   python scripts/smoke_test_identity.py ref1.jpg ref2.jpg
#
# Env var yang perlu di-set di UI vast.ai saat create instance (lihat
# app/core/config.py -- Settings pakai prefix HAROFRAME_, nested delimiter __):
#   HAROFRAME_IDENTITY__HF_TOKEN=<hugging face token>
#   HAROFRAME_IDENTITY__IPADAPTER__ENABLED=true   (atau INSTANTID__ENABLED=true)
#   dst. sesuai kebutuhan -- lihat IdentityConfig di app/core/config.py.
#
# Restoration extra (gfpgan/basicsr/facexlib) SENGAJA tidak di-bake di image ini.
# Kalau dibutuhkan, install manual di dalam container:
#   pip install -e ".[restoration]"

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
	libgl1 \
	libglib2.0-0 \
	build-essential \
	&& rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /workspace

COPY . /workspace

RUN pip install --no-cache-dir -e ".[gpu]"

CMD ["sleep", "infinity"]
