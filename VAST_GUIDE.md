# Panduan Deploy & Run HaroFRAME di vast.ai

Panduan step-by-step untuk menjalankan instance GPU di [vast.ai](https://vast.ai)
dari template **PyTorch** resmi (lewat `entrypoint.sh` sebagai "On-start Script"),
lalu menghasilkan video sungguhan dari modul `app/generation/`. Semua run yang
butuh model asli/GPU **harus** lewat vast.ai -- mesin dev lokal tidak punya GPU
yang memadai (lihat `CLAUDE.md`).

Ada juga jalur alternatif berbasis image Docker custom (`Dockerfile` di root
repo) untuk yang lebih suka membangun/push image sendiri -- lihat bagian
["Alternatif: image Docker custom"](#alternatif-image-docker-custom) di bawah.
Panduan utama di bawah ini pakai template PyTorch + `entrypoint.sh`.

## Prasyarat

- Akun vast.ai yang sudah diisi saldo.
- (Opsional tapi disarankan) token Hugging Face jika model yang dipakai
  gated/rate-limited.

## 1. Buat instance dari template PyTorch

1. Buka [Create Instance](https://cloud.vast.ai/create/) di dashboard vast.ai.
2. Pilih template **PyTorch** resmi vast.ai (bukan template lain) -- template
   ini sudah membawa PyTorch pre-installed & pre-built melawan CUDA/driver
   instance-nya sendiri di `/venv/main`, yang dipakai ulang (bukan ditimpa)
   oleh `entrypoint.sh` (lihat langkah 2).
3. Filter GPU dengan driver yang support **CUDA 12.1+**.
4. Sisakan disk space yang cukup (**disarankan minimal 40-60GB**) -- SDXL base
   model + InstantID/IP-Adapter/LoRA weights bisa mencapai puluhan GB total,
   dan itu semua di-cache di `.cache/models` di dalam instance.
5. Di field **On-start Script**, tempel seluruh isi `entrypoint.sh` dari repo
   ini. Script ini idempotent -- aman dijalankan tiap kali instance
   (re)start, dan otomatis: clone/update repo ke `$HOME/HaroFRAME`, install
   system deps + `pip install -e ".[gpu]"`, siapkan folder cache/output, lalu
   cetak sanity check (status CUDA, token HF, adapter identitas mana yang
   aktif).
6. Di bagian **Environment Variables** (atau **Docker Options** tergantung UI
   vast.ai saat ini), tempel variabel Anda satu per satu -- lihat langkah 3
   di bawah untuk daftar variabelnya. Jangan pernah menempelkan token asli ke
   field publik/log manapun.
7. Klik **Rent**. On-start script otomatis jalan saat instance siap.

## 2. Connect ke instance

Gunakan salah satu:

- **Instance Portal** / tombol exec bawaan vast.ai di dashboard (paling mudah).
- CLI resmi vast.ai (`vastai ssh <instance-id>` atau serupa) kalau sudah
  terpasang di mesin Anda.

Setelah masuk, `cd $HOME/HaroFRAME` (biasanya `/root/HaroFRAME`) -- ini working
directory repo yang di-clone `entrypoint.sh`, bukan `/workspace`. Kalau
template PyTorch venv-nya aktif otomatis di shell interaktif Anda, `python3`
sudah menunjuk ke `/venv/main`; kalau tidak, jalankan
`source /venv/main/bin/activate` dulu (sama seperti yang dilakukan
`entrypoint.sh`).

## 3. Environment variables

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
env var instance -- lihat
["Panduan CLI Interaktif"](#panduan-cli-interaktif-scriptsinteractive_generatepy).

## 4. Verifikasi environment

```bash
cd $HOME/HaroFRAME
python3 -c "import torch; print(torch.cuda.is_available())"   # harus True
python3 -m pytest tests/ -q                                    # semua test harus lolos
```

`entrypoint.sh` sudah mencetak sanity check yang sama (status CUDA, token HF,
status adapter identitas) di akhir log-nya setiap kali instance start --
langkah ini untuk verifikasi manual/ulang kalau perlu.

## 5. Kirim foto referensi ke instance

Foto referensi (personal, jangan pernah masuk git -- lihat `test_images/README.md`)
perlu dipindahkan manual ke instance. Beberapa cara:

- **`scp`/`rsync`** ke instance lewat SSH kalau vast.ai memberi akses SSH port.
- **Upload lewat Instance Portal** (fitur file upload di web UI vast.ai, kalau tersedia).
- Taruh di `$HOME/HaroFRAME/test_images/<nama>.jpg`.

## 6. Jalankan smoke test

```bash
# Identity module saja (face analysis + fusion + build_conditioning, tanpa video):
python3 scripts/smoke_test_identity.py test_images/alex.jpg

# Generation, per-frame PNG saja (belum mux video, buat cek visual cepat):
python3 scripts/smoke_test_generation.py test_images/alex.jpg "a person smiling" --frames 16 --out outputs/smoke
```

## 7. Batch-test dengan foto asli

Setelah beberapa foto ada di `test_images/` (opsional: tambah file `.txt`
sejudul untuk prompt per-foto, lihat `test_images/README.md`):

```bash
python3 scripts/test_real_images.py --prompt "a person, natural lighting, subtle motion" --out outputs/test_real_images
```

Script ini jalan per-foto, satu foto gagal tidak menghentikan sisanya, dan
mencetak ringkasan OK/FAIL di akhir.

## 8. Generate video atau gambar tunggal

```bash
# Image2video -- animasi (motion dari HAROFRAME_GENERATION__MOTION__MODE):
python3 scripts/generate_video.py test_images/alex.jpg "a person smiling, gentle breeze" --out outputs/alex.mp4

# Image2image -- satu gambar hasil transformasi dari foto yang sama, tanpa motion:
python3 scripts/generate_image.py test_images/alex.jpg "anime style portrait" --out outputs/alex_img2img.png
```

Atau pakai CLI interaktif (`scripts/interactive_generate.py`) kalau mau
menulis prompt, memilih model, dan submit beberapa job sekaligus tanpa edit
env var instance -- lihat panduan lengkapnya di
["Panduan CLI Interaktif"](#panduan-cli-interaktif-scriptsinteractive_generatepy)
di bawah.

## 9. Ambil hasil video dari instance

Sama seperti langkah 5 tapi arah sebaliknya (`scp`/download lewat Instance
Portal) dari `outputs/` di dalam instance ke mesin lokal Anda.

## Panduan CLI Interaktif (`scripts/interactive_generate.py`)

CLI ini cocok kalau Anda mau coba-coba prompt/model tanpa edit env var tiap
kali, atau submit beberapa job berbeda (checkpoint/LoRA/mode berbeda-beda)
dalam satu sesi -- untuk sekali jalan cepat dengan satu kombinasi
model/prompt tetap, `generate_video.py`/`generate_image.py` (langkah 8 di
atas) lebih langsung. Jalankan dari `$HOME/HaroFRAME`:

```bash
python3 scripts/interactive_generate.py
```

### Langkah 0 -- API keys (opsional)

Diminta token Hugging Face dan Civitai API key untuk sesi ini (input
disamarkan lewat `getpass`). Tekan Enter di kedua prompt untuk pakai nilai
dari env var yang sudah di-set (`HAROFRAME_IDENTITY__HF_TOKEN`,
`HAROFRAME_GENERATION__LORA__CIVITAI_API_KEY`) -- tidak wajib diisi ulang
kalau env var instance sudah benar.

### Langkah 1-2 -- Install checkpoint & LoRA ke pool

Dua langkah ini **hanya mengunduh & mendaftarkan** model ke pool, belum
memilih mana yang dipakai job (itu terjadi nanti saat submit job). Untuk
masing-masing, sumber yang diterima sama: link model-page Civitai, link
download langsung Civitai (`civitai.com/api/download/models/...`), link
download langsung platform lain, path lokal, atau repo_id Hugging Face. Bisa
install lebih dari satu checkpoint/LoRA di sini, semuanya akan muncul sebagai
pilihan nanti. Saat install LoRA, Anda juga diminta bobot/scale-nya (ada guide
rentang yang disarankan di teks prompt-nya).

### Menu utama

Setelah checkpoint/LoRA pertama terinstall, masuk ke menu yang bisa dipakai
berulang kali:

```
[1] Submit job generate baru
[2] Lihat status antrian
[3] Keluar
```

### [1] Submit job generate baru

Urutan prompt-nya:

1. **Folder foto referensi** -- Enter untuk pakai `test_images/`, atau ketik
   path folder lain (mis. folder tempat Anda `scp` foto tadi). Foto yang
   ditemukan di folder itu ditampilkan bernomor untuk dipilih; kalau tidak
   ada foto di folder tersebut (atau Anda ingin foto di luar folder itu),
   ada fallback untuk ketik path file langsung.
2. **Prompt** -- teks generate (wajib diisi). Khusus mode Garment-Swap,
   prompt ini nanti diganti pertanyaan "deskripsi pakaian baru" (lihat poin 4).
3. **Negative prompt** -- opsional, Enter untuk kosong.
4. **Mode** -- pilih salah satu:
   - **[1] Image2Video** -- animasi dengan motion (pan/zoom) dari config,
     ditanya jumlah frame (Enter = dari `fps x duration_seconds` config).
   - **[2] Image2Image** -- satu gambar hasil transformasi dari foto yang
     sama, tanpa motion; ditanya strength img2img (ada guide rentang
     disarankan di prompt-nya). Hanya berefek di jalur IP-Adapter --
     diabaikan kalau adapter aktifnya InstantID.
   - **[3] Garment-Swap** -- ganti pakaian orang di foto lewat inpainting
     bermask SAM, latar/pose/wajah dipertahankan. **Hanya jalan di jalur
     IP-Adapter/FaceID-SDXL** -- kalau adapter aktif Anda InstantID, CLI
     menolak submit di titik ini dengan pesan jelas (InstantID tidak punya
     entry point inpaint/img2img). Ditanya deskripsi outfit baru (mis.
     `"sleeveless summer top, light shorts"`), lalu parameter khusus (lihat
     poin 6).
5. **Seed** -- opsional, Enter = acak.
6. **Parameter Render** -- guidance scale (CFG) & jumlah inference steps,
   keduanya punya guide rentang yang disarankan di teks prompt-nya, Enter =
   default dari config. Kalau adapter aktif IP-Adapter (bukan InstantID) dan
   mode bukan Garment-Swap, ditanya lagi apakah mau aktifkan **pose/depth
   ControlNet** beserta conditioning scale-nya -- ini membantu menjaga
   pose/struktur tubuh tetap dekat ke foto asli.
   Khusus mode Garment-Swap, ada blok parameter tambahan: inpaint strength,
   mask dilation px (perbesar area inpaint supaya kulit yang baru terbuka
   ikut tercakup), dan apakah area kaki ikut dimasukkan ke mask -- semuanya
   juga punya guide rentang disarankan.
7. **Nama file output** -- default otomatis dari nama foto + mode (mis.
   `alex_garment.png`), bisa diganti.
8. **Pilih checkpoint** -- dari checkpoint yang sudah diinstall di langkah 1.
9. **Pilih LoRA** -- dari LoRA yang sudah diinstall di langkah 2 (boleh nol
   atau lebih, dibatasi `max_active_loras`).
10. **Ringkasan & konfirmasi** -- semua pilihan di atas dicetak ulang, lalu
    `y/N` untuk benar-benar menambahkan ke antrian. Menjawab selain `y`
    membatalkan submit tanpa efek samping.

Submit **tidak menunggu** job selesai -- kembali ke menu utama seketika,
job jalan di background thread satu per satu.

### [2] Lihat status antrian

Mencetak satu baris per job yang pernah disubmit di sesi ini, format:

```
[<job_id>] <mode> <nama_foto> -- <status>
```

`<status>` salah satu dari: `queued`, `running` (untuk Image2Video juga
menampilkan `frame X/Y`), `selesai -> <path_output>`, atau
`error: <pesan_error>`. Bisa dipanggil kapan saja, termasuk sambil job lain
masih jalan di background.

### [3] Keluar

Kalau masih ada job berstatus `queued`/`running`, CLI memperingatkan dulu dan
minta konfirmasi -- job yang belum selesai **hilang** kalau tetap keluar
(background thread ikut mati bersama proses). Tunggu sampai status `selesai`/
`error` dulu di menu [2] kalau tidak mau kehilangan job yang sedang jalan.

## Persist cache model antar restart instance

Model weights (SDXL base, InstantID, IP-Adapter, LoRA) di-cache di
`.cache/models` (relatif ke `$HOME/HaroFRAME`, lihat `IdentityConfig.cache_dir`).
Kalau instance vast.ai Anda stop/restart tanpa disk persisten, cache ini hilang
dan semua model ke-download ulang (bisa puluhan GB, lama). Pastikan opsi disk
persisten diaktifkan saat membuat instance kalau Anda berencana stop/start
berulang, bukan sekali pakai.

## Alternatif: image Docker custom

Kalau lebih suka membangun & push image sendiri (bukan template PyTorch +
`entrypoint.sh` seperti panduan utama di atas):

```bash
cp .env.example .env      # isi HAROFRAME_IDENTITY__HF_TOKEN dll.
docker build -t <registry-user>/haroframe:latest .
docker push <registry-user>/haroframe:latest
```

Base image `python:3.11-slim` + `pip install -e ".[gpu]"` (lihat komentar
header `Dockerfile` untuk detail lengkap). Ekstra `restoration` dan `garment`
sengaja tidak di-bake -- install manual di dalam container kalau dibutuhkan.

Di vast.ai: buat instance baru, isi field **Image Path/Tag** dengan
`<registry-user>/haroframe:latest` (bukan memilih template PyTorch), isi
**Environment Variables** seperti langkah 3 di atas, lalu **Rent**. Container
ini idle (`CMD sleep infinity`, tanpa on-start script) dan **tidak** membundel
SSH server sendiri -- connect lewat Instance Portal/exec seperti biasa.
Working directory di dalam container ini adalah `/workspace` (bukan
`$HOME/HaroFRAME`), jadi sesuaikan path pada langkah 5-9 di atas. Perlu
Docker terpasang di mesin manapun yang menjalankan `docker build`, dan akun
registry (Docker Hub, GHCR, dll.) untuk push image -- keduanya tidak
dibutuhkan untuk jalur template PyTorch di atas.

Testing lokal dengan Docker Desktop (opsional, bukan cara vast.ai sendiri
di-deploy -- cuma kemudahan untuk testing sebelum push):

```bash
cp .env.example .env   # isi dulu
docker compose up -d
docker compose exec haroframe python scripts/test_real_images.py --prompt "..."
docker compose down
```

`docker-compose.yml` di root repo sudah menyiapkan volume mount untuk
`test_images/`, `outputs/`, dan `.cache/`.

## Troubleshooting

- **`torch.cuda.is_available()` mengembalikan `False`** -- instance vast.ai
  yang Anda sewa mungkin tidak benar-benar meng-attach GPU, atau driver host
  tidak kompatibel dengan CUDA 12.1+. Cek ulang filter GPU saat create instance.
- **`ConflictingAdapterConfigError`** -- `HAROFRAME_IDENTITY__IPADAPTER__ENABLED`
  dan `HAROFRAME_IDENTITY__INSTANTID__ENABLED` sama-sama `true`. Pilih satu.
  `entrypoint.sh` juga sudah memperingatkan ini di log-nya kalau terjadi.
- **Download model lambat/timeout berulang** -- biasanya bandwidth instance
  vast.ai yang Anda sewa, atau `HAROFRAME_IDENTITY__HF_TOKEN` belum di-set
  untuk model yang rate-limited tanpa auth.
- **`ModelLoadError: gfpgan (and its basicsr/facexlib dependencies) are not
  installed`** -- disengaja, extra `restoration` tidak diinstal otomatis.
  Install manual: `pip install -e ".[restoration]"`.
- **`ModelLoadError: segment-anything is not installed`** (mode Garment-Swap
  di `interactive_generate.py`) -- disengaja, extra `garment` tidak diinstal
  otomatis. Install manual: `pip install -e ".[garment]"`, lalu unduh
  sebuah checkpoint SAM (mis. `sam_vit_l_0b3195.pth` dari
  `https://dl.fbaipublicfiles.com/segment_anything/`) dan set
  `HAROFRAME_GENERATION__GARMENT__SAM_CHECKPOINT_PATH` ke path-nya.
- **`python3 -m pip install -e ".[gpu]"` di `entrypoint.sh` sepertinya
  menginstal ulang/menimpa torch** -- cek apakah `/venv/main/bin/activate`
  benar-benar berhasil di-source (baris pertama log `entrypoint.sh` seharusnya
  mencetak "activating vast.ai PyTorch template venv"); kalau instance Anda
  bukan dari template PyTorch resmi, path venv-nya bisa berbeda/tidak ada,
  dan `pip` akan jatuh ke Python sistem lalu menginstal torch dari PyPI
  sendiri (tetap berfungsi, hanya lebih lambat di run pertama).
