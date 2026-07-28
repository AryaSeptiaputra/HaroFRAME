# Panduan Deploy & Run HaroFRAME di vast.ai

Panduan step-by-step untuk membangun image Docker, menjalankannya sebagai instance
GPU di [vast.ai](https://vast.ai), lalu menghasilkan video sungguhan dari modul
`app/generation/`. Semua run yang butuh model asli/GPU **harus** lewat vast.ai --
mesin dev lokal tidak punya GPU yang memadai (lihat `CLAUDE.md`).

## Prasyarat

- Docker terpasang di mesin manapun yang akan Anda pakai untuk `docker build`
  (tidak harus mesin yang sama dengan yang menjalankan Claude Code).
- Akun registry (Docker Hub, GHCR, dll.) untuk push image.
- Akun vast.ai yang sudah diisi saldo.
- (Opsional tapi disarankan) token Hugging Face jika model yang dipakai
  gated/rate-limited.

## 1. Build & push image

```bash
cp .env.example .env      # isi HAROFRAME_IDENTITY__HF_TOKEN dll. (lihat langkah 4)
docker build -t <registry-user>/haroframe:latest .
docker push <registry-user>/haroframe:latest
```

Base image `python:3.11-slim` + `pip install -e ".[gpu]"` (lihat komentar header
`Dockerfile` untuk detail lengkap). Ekstra `restoration` (GFPGAN) sengaja tidak
di-bake -- install manual di dalam container kalau dibutuhkan.

## 2. Buat instance di vast.ai

1. Buka [Create Instance](https://cloud.vast.ai/create/) di dashboard vast.ai.
2. Filter GPU dengan driver yang support **CUDA 12.1+** (torch dari PyPI sudah
   bawa runtime CUDA sendiri, jadi yang penting driver host-nya kompatibel).
3. Sisakan disk space yang cukup (**disarankan minimal 40-60GB**) -- SDXL base
   model + InstantID/IP-Adapter/LoRA weights bisa mencapai puluhan GB total,
   dan itu semua di-cache di `.cache/models` di dalam container.
4. Di field **Image Path/Tag**, isi `<registry-user>/haroframe:latest`.
5. Di bagian **Environment Variables** (atau **Docker Options** tergantung UI
   vast.ai saat ini), tempel isi `.env` Anda satu per satu -- lihat langkah 4
   di bawah untuk daftar variabelnya. Jangan pernah menempelkan token asli ke
   field publik/log manapun.
6. Klik **Rent**. Container akan idle (`CMD sleep infinity`) begitu instance
   siap -- ini disengaja, lihat langkah 3.

## 3. Connect ke instance

Container ini **tidak** membundel SSH server sendiri. Gunakan salah satu:

- **Instance Portal** / tombol exec bawaan vast.ai di dashboard (paling mudah).
- CLI resmi vast.ai (`vastai ssh <instance-id>` atau serupa) kalau sudah
  terpasang di mesin Anda.

Setelah masuk, working directory default adalah `/workspace` (isi repo ini).

## 4. Environment variables

Semua field ada di `app/core/config.py` (`Settings`, prefix `HAROFRAME_`, nested
delimiter `__`). Yang paling sering dipakai:

| Variabel | Keterangan |
|---|---|
| `HAROFRAME_IDENTITY__HF_TOKEN` | Token Hugging Face untuk model gated/rate-limited |
| `HAROFRAME_IDENTITY__IPADAPTER__ENABLED` | `true`/`false` -- jalur IP-Adapter/FaceID-SDXL |
| `HAROFRAME_IDENTITY__INSTANTID__ENABLED` | `true`/`false` -- jalur InstantID (hanya boleh satu dari dua ini `true`) |
| `HAROFRAME_IDENTITY__DEVICE` | `cuda` di vast.ai |
| `HAROFRAME_GENERATION__MOTION__MODE` | `ken_burns_2d` (default), `static`, atau `depth_parallax` |
| `HAROFRAME_GENERATION__OUTPUT__FPS`, `HAROFRAME_GENERATION__OUTPUT__DURATION_SECONDS` | Panjang & fps video output |
| `HAROFRAME_GENERATION__LORA__CIVITAI_API_KEY` | Opsional, hanya kalau LoRA/checkpoint dari Civitai butuh auth |

Lihat `.env.example` untuk template siap-copy. Kalau pakai
`scripts/interactive_generate.py`, token HF & Civitai API key ini juga bisa
diisi/ditimpa langsung di awal sesi CLI-nya (langkah 0) tanpa perlu edit
`.env` -- lihat langkah 9 di bawah.

## 5. Verifikasi environment

```bash
python -c "import torch; print(torch.cuda.is_available())"   # harus True
python -m pytest tests/ -q                                    # semua test harus lolos
```

## 6. Kirim foto referensi ke instance

Foto referensi (personal, jangan pernah masuk git -- lihat `test_images/README.md`)
perlu dipindahkan manual ke instance. Beberapa cara:

- **`docker cp`** kalau Anda punya akses langsung ke host Docker vast.ai (jarang).
- **`scp`/`rsync`** ke instance lewat SSH kalau vast.ai memberi akses SSH port.
- **Upload lewat Instance Portal** (fitur file upload di web UI vast.ai, kalau tersedia).
- Taruh di `test_images/<nama>.jpg` relatif ke `/workspace`.

## 7. Jalankan smoke test

```bash
# Identity module saja (face analysis + fusion + build_conditioning, tanpa video):
python scripts/smoke_test_identity.py test_images/alex.jpg

# Generation, per-frame PNG saja (belum mux video, buat cek visual cepat):
python scripts/smoke_test_generation.py test_images/alex.jpg "a person smiling" --frames 16 --out outputs/smoke
```

## 8. Batch-test dengan foto asli

Setelah beberapa foto ada di `test_images/` (opsional: tambah file `.txt`
sejudul untuk prompt per-foto, lihat `test_images/README.md`):

```bash
python scripts/test_real_images.py --prompt "a person, natural lighting, subtle motion" --out outputs/test_real_images
```

Script ini jalan per-foto, satu foto gagal tidak menghentikan sisanya, dan
mencetak ringkasan OK/FAIL di akhir.

## 9. Generate video atau gambar tunggal

```bash
# Image2video -- animasi (motion dari HAROFRAME_GENERATION__MOTION__MODE):
python scripts/generate_video.py test_images/alex.jpg "a person smiling, gentle breeze" --out outputs/alex.mp4

# Image2image -- satu gambar hasil transformasi dari foto yang sama, tanpa motion:
python scripts/generate_image.py test_images/alex.jpg "anime style portrait" --out outputs/alex_img2img.png
```

Atau pakai CLI interaktif (`scripts/interactive_generate.py`) kalau mau
menulis prompt, memilih model, dan submit beberapa job sekaligus tanpa edit
`.env`:

```bash
python scripts/interactive_generate.py
```

Alurnya: (0) masukkan Hugging Face token / Civitai API key untuk sesi ini
(opsional, input disamarkan, Enter = pakai dari `.env`) → (1)/(2) install
satu atau beberapa **checkpoint SDXL** dan **LoRA** ke pool (bisa dari link
Civitai, link download langsung, path lokal, atau repo_id HuggingFace --
diunduh saat itu juga) → lalu menu utama yang bisa dipakai berulang kali:
**submit job baru** (pilih foto, tulis prompt, pilih mode **Image2Video**
atau **Image2Image**, pilih checkpoint & LoRA mana dari yang sudah
terinstall, konfirmasi), **lihat status antrian** (job jalan satu per satu di
background -- submit tidak menunggu selesai, bisa langsung submit job lain
atau cek status: `queued`/`running` dengan progress frame/`done`/`failed`),
atau **keluar** (akan memperingatkan kalau masih ada job yang belum selesai --
job itu hilang kalau tetap keluar).

## 10. Ambil hasil video dari instance

Sama seperti langkah 6 tapi arah sebaliknya (`scp`/download lewat Instance
Portal) dari `outputs/` di dalam container ke mesin lokal Anda.

## Persist cache model antar restart instance

Model weights (SDXL base, InstantID, IP-Adapter, LoRA) di-cache di
`.cache/models` (relatif ke `/workspace`, lihat `IdentityConfig.cache_dir`).
Kalau instance vast.ai Anda stop/restart tanpa disk persisten, cache ini hilang
dan semua model ke-download ulang (bisa puluhan GB, lama). Pastikan opsi disk
persisten diaktifkan saat membuat instance kalau Anda berencana stop/start
berulang, bukan sekali pakai.

## Testing lokal dengan Docker Desktop (opsional)

Kalau Anda punya mesin lain dengan GPU + Docker Desktop, `docker-compose.yml`
di root repo sudah menyiapkan volume mount untuk `test_images/`, `outputs/`,
dan `.cache/`:

```bash
cp .env.example .env   # isi dulu
docker compose up -d
docker compose exec haroframe python scripts/test_real_images.py --prompt "..."
docker compose down
```

Ini **bukan** cara vast.ai sendiri di-deploy (vast.ai pakai image langsung
lewat UI-nya, bukan compose file) -- ini cuma kemudahan untuk testing lokal
sebelum push ke vast.ai.

## Troubleshooting

- **`torch.cuda.is_available()` mengembalikan `False`** -- instance vast.ai
  yang Anda sewa mungkin tidak benar-benar meng-attach GPU, atau driver host
  tidak kompatibel dengan CUDA 12.1+. Cek ulang filter GPU saat create instance.
- **`ConflictingAdapterConfigError`** -- `HAROFRAME_IDENTITY__IPADAPTER__ENABLED`
  dan `HAROFRAME_IDENTITY__INSTANTID__ENABLED` sama-sama `true`. Pilih satu.
- **Download model lambat/timeout berulang** -- biasanya bandwidth instance
  vast.ai yang Anda sewa, atau `HAROFRAME_IDENTITY__HF_TOKEN` belum di-set
  untuk model yang rate-limited tanpa auth.
- **`ModelLoadError: gfpgan (and its basicsr/facexlib dependencies) are not
  installed`** -- disengaja, extra `restoration` tidak di-bake di image
  default. Install manual: `pip install -e ".[restoration]"`.
- **`ModelLoadError: segment-anything is not installed`** (mode Garment-Swap
  di `interactive_generate.py`) -- disengaja, extra `garment` tidak di-bake di
  image default. Install manual: `pip install -e ".[garment]"`, lalu unduh
  sebuah checkpoint SAM (mis. `sam_vit_l_0b3195.pth` dari
  `https://dl.fbaipublicfiles.com/segment_anything/`) dan set
  `HAROFRAME_GENERATION__GARMENT__SAM_CHECKPOINT_PATH` ke path-nya.
