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
| `HAROFRAME_IDENTITY__BASE_SDXL_MODEL` | Checkpoint SDXL dasar (default `SG161222/RealVisXL_V5.0`, lihat langkah 3b) |
| `HAROFRAME_GENERATION__LORA__ENTRIES` | Daftar LoRA sebagai JSON, diinstal otomatis saat setup (lihat langkah 3b) |
| `HAROFRAME_PREFETCH` | `true` (default) -- unduh bobot model saat setup, bukan saat generate pertama |

Lihat `.env.example` untuk template siap-copy. Kalau pakai
`scripts/interactive_generate.py`, token HF & Civitai API key ini juga bisa
diisi/ditimpa langsung di awal sesi CLI-nya (langkah 0) tanpa perlu edit
env var instance -- lihat
["Panduan CLI Interaktif"](#panduan-cli-interaktif-scriptsinteractive_generatepy).

## 3b. Model base & LoRA (terinstal otomatis)

`entrypoint.sh` menjalankan `scripts/prefetch_models.py` di akhir setup, jadi
bobot model sudah siap sebelum Anda generate pertama kali -- bukan diunduh
sambil menunggu di tengah run. Yang diunduh:

1. **Checkpoint SDXL dasar** -- default **`SG161222/RealVisXL_V5.0`**.
   Dipilih karena seluruh kerja pipeline ini adalah manusia: tahap 1
   menggenerate lengan/torso dari nol, tahap 2 harus menjaga pose tetap masuk
   akal, dan SDXL bawaan (`stabilityai/stable-diffusion-xl-base-1.0`) jelas
   lebih lemah di anatomi manusia dibanding merge photoreal komunitas.
   Ungated, lisensi openrail++, ada varian fp16 (~7GB, bukan ~14GB fp32).
   Alternatif terdekat: `RunDiffusion/Juggernaut-XL-v9`. Ganti cukup lewat
   `HAROFRAME_IDENTITY__BASE_SDXL_MODEL`, tanpa ubah kode.
2. **LoRA** -- dari `HAROFRAME_GENERATION__LORA__ENTRIES`, ditulis sebagai JSON:

   ```
   HAROFRAME_GENERATION__LORA__ENTRIES=[{"adapter_name":"detail","source":"https://civitai.com/models/122359","scale":0.6}]
   ```

   `source` menerima empat bentuk yang sama seperti checkpoint: link Civitai,
   link download langsung, path lokal, atau repo_id Hugging Face. `adapter_name`
   harus unik dan tidak boleh `faceid` (dipakai internal). **Tidak ada LoRA yang
   aktif secara default** -- isi sendiri sesuai kebutuhan.
3. **Checkpoint SAM** untuk tahap inpaint -- `vit_b` (375MB) secara default;
   `vit_l` (1.25GB) / `vit_h` (2.56GB) memberi mask lebih baik. Ganti lewat
   `HAROFRAME_GENERATION__INPAINT__SAM__MODEL_TYPE`; path default ikut
   menyesuaikan sendiri.

Kalau salah satu gagal diunduh, setup tetap lanjut dan instance tetap
terpakai -- yang gagal akan diunduh ulang secara lazy saat generate. Ringkasan
OK/FAIL per item dicetak di log `entrypoint.sh`.

Set `HAROFRAME_PREFETCH=false` untuk melewati langkah ini (koneksi terbatas,
atau `.cache/models` sudah berupa volume persisten yang terisi). Bisa juga
dijalankan manual kapan saja setelah ganti model:

```bash
python3 scripts/prefetch_models.py              # semua
python3 scripts/prefetch_models.py --skip-sam   # base + LoRA saja
```

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

# Generation, per-frame PNG saja (belum mux video, buat cek visual cepat).
# --no-inpaint karena script ini memang untuk memeriksa render per-frame saja
# (soal dua tahap & dua prompt, lihat langkah 8):
python3 scripts/smoke_test_generation.py test_images/alex.jpg "a person smiling" --frames 16 --no-inpaint --out outputs/smoke
```

## 7. Batch-test dengan foto asli

Setelah beberapa foto ada di `test_images/` (opsional: tambah file `.txt`
sejudul untuk prompt per-foto, lihat `test_images/README.md`):

```bash
python3 scripts/test_real_images.py --prompt "a person, natural lighting, subtle motion" \
  --no-inpaint --out outputs/test_real_images
```

Script ini jalan per-foto, satu foto gagal tidak menghentikan sisanya, dan
mencetak ringkasan OK/FAIL di akhir. `--prompt`/file `.txt` adalah prompt
tahap 2 dan berbeda per foto; kalau mau pakai tahap inpaint, `--inpaint-prompt`
berlaku untuk seluruh batch (lihat langkah 8).

## 8. Dua tahap, dua prompt

Generate sekarang **dua tahap secara default**:

```
foto ──► [tahap 1: inpaint] ──► foto hasil edit ──► [tahap 2: render] ──► hasil
          ganti pakaian /                            pose & gaya
          generate anggota badan
```

Jadi setiap generate butuh **dua prompt**, dan semua script punya pasangan
flag yang sama:

| Flag | Artinya |
|---|---|
| `--inpaint-prompt "..."` (`-i` di `quick_generate.py`) | prompt tahap 1: area yang di-mask mau jadi apa |
| prompt posisional | prompt tahap 2: hasil akhirnya mau seperti apa |
| `--no-inpaint` | lewati tahap 1, kembali ke satu tahap seperti dulu |

Salah satu dari `--inpaint-prompt` atau `--no-inpaint` **wajib** ada. Kalau
tidak, script berhenti langsung (sebelum memuat model apa pun) dengan pesan
yang menyebutkan ketiga cara memperbaikinya. Prompt tahap 1 juga bisa di-set
sekali lewat `HAROFRAME_GENERATION__INPAINT__PROMPT` supaya tidak perlu
mengetik `--inpaint-prompt` tiap kali.

**Tahap 1 butuh checkpoint SAM**, tapi itu sudah diunduh otomatis saat setup
(langkah 3b) -- tidak ada yang perlu Anda lakukan. Kalau memang tidak butuh
inpaint sama sekali, pakai `--no-inpaint` atau set
`HAROFRAME_GENERATION__INPAINT__ENABLED=false` sekali di env var instance;
checkpoint SAM-nya pun tidak akan diunduh.

## 8a. Cara tercepat

Kalau yang Anda mau cuma "masukkan foto, tulis prompt, lihat hasilnya":

```bash
python3 scripts/quick_generate.py test_images/alex.jpg "anime style portrait, studio lighting" \
  -i "a red hoodie" -n "blurry, extra fingers"

# atau satu tahap saja:
python3 scripts/quick_generate.py test_images/alex.jpg "anime style portrait" --no-inpaint
```

`-n`/`--negative` opsional. Sisanya -- mode, strength, guidance, steps, seed,
checkpoint, LoRA, ControlNet, nama file output -- diambil dari config/env var
yang sudah Anda set di langkah 3. Semua nilai yang dipakai dicetak dulu
sebelum render, jadi bukan kotak hitam.

Yang sudah ditentukan untuk Anda:

- **Selalu satu gambar (image2image)**, bukan video -- satu render jauh lebih
  cepat daripada puluhan frame, jadi enak untuk coba-coba prompt. Untuk video
  pakai `generate_video.py` di bawah.
- **Seed acak tiap run**, dicetak di akhir bersama perintah siap-tempel untuk
  mengulang hasil yang Anda suka lewat `generate_image.py --seed`.
- **Output tidak pernah menimpa**: `outputs/alex_quick.png`, lalu
  `alex_quick_2.png`, dan seterusnya.

## 8b. Generate dengan kontrol penuh

```bash
# Image2video -- animasi (motion dari HAROFRAME_GENERATION__MOTION__MODE):
python3 scripts/generate_video.py test_images/alex.jpg "a person smiling, gentle breeze" \
  --inpaint-prompt "a red hoodie" --out outputs/alex.mp4

# Image2image -- satu gambar hasil transformasi dari foto yang sama, tanpa motion:
python3 scripts/generate_image.py test_images/alex.jpg "anime style portrait" \
  --inpaint-prompt "sleeveless summer top" --out outputs/alex_img2img.png

# Satu tahap saja (tanpa ganti pakaian / generate anggota badan):
python3 scripts/generate_video.py test_images/alex.jpg "a person smiling, gentle breeze" --no-inpaint
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
dalam satu sesi. Untuk sekadar "foto + prompt + negative prompt", pakai
`quick_generate.py` (langkah 8); untuk sekali jalan dengan satu kombinasi
model/prompt tetap, `generate_video.py`/`generate_image.py` (langkah 8b)
lebih langsung. Jalankan dari `$HOME/HaroFRAME`:

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
2. **Prompt** -- teks generate (wajib diisi). Ini prompt **tahap 2** (pose &
   gaya). Prompt tahap 1 (area yang di-inpaint) ditanya terpisah di poin 5.
3. **Negative prompt** -- opsional, Enter untuk kosong.
4. **Mode** -- pilih salah satu:
   - **[1] Image2Video** -- animasi dengan motion (pan/zoom) dari config,
     ditanya jumlah frame (Enter = dari `fps x duration_seconds` config).
   - **[2] Image2Image** -- satu gambar hasil transformasi dari foto yang
     sama, tanpa motion; ditanya strength img2img (ada guide rentang
     disarankan di prompt-nya). Hanya berefek di jalur IP-Adapter --
     diabaikan kalau adapter aktifnya InstantID.
5. **Tahap 1: Inpaint** -- **default `y`** (Enter = aktif). Ganti pakaian atau
   generate bagian tubuh pada foto sumber **sebelum** tahap render jalan; hasil
   inpaint jadi foto sumber untuk tahap 2. Ini bukan mode tersendiri -- bisa
   dikombinasikan dengan Image2Video maupun Image2Image, dan **jalan di kedua
   jalur adapter** (IP-Adapter maupun InstantID), karena tahap ini tidak
   memasang face adapter sama sekali (mask hanya menutupi pakaian/anggota
   badan, wajah tidak tersentuh). Kalau `y`, ditanya: deskripsi target
   area (mis. `"sleeveless summer top, light shorts"` -- ini prompt tahap 1),
   inpaint strength, mask dilation px (perbesar area inpaint supaya kulit yang
   baru terbuka ikut tercakup), apakah lengan/kaki ikut masuk mask, dan apakah
   pakai pose ControlNet untuk anatomi area baru -- semuanya punya guide rentang
   disarankan. Butuh checkpoint SAM (lihat langkah 8). Jawab `n` untuk render
   satu tahap langsung dari foto asli.
6. **Seed** -- opsional, Enter = acak. Seed yang sama dipakai untuk tahap
   inpaint dan semua frame render.
7. **Parameter Render** -- guidance scale (CFG) & jumlah inference steps,
   keduanya punya guide rentang yang disarankan di teks prompt-nya, Enter =
   default dari config. Kalau adapter aktif IP-Adapter (bukan InstantID),
   ditanya lagi apakah mau aktifkan **pose/depth ControlNet** beserta
   conditioning scale-nya -- ini membantu menjaga pose/struktur tubuh tetap
   dekat ke foto asli. Perhatikan ini beda dari pose ControlNet di poin 5:
   yang di sini mengarahkan struktur untuk render img2img, yang di poin 5
   mengarahkan anatomi area yang baru digenerate saat inpaint.
8. **Nama file output** -- default otomatis dari nama foto + mode (mis.
   `alex_img2img.png`), bisa diganti.
9. **Pilih checkpoint** -- dari checkpoint yang sudah diinstall di langkah 1.
10. **Pilih LoRA** -- dari LoRA yang sudah diinstall di langkah 2 (boleh nol
    atau lebih, dibatasi `max_active_loras`).
11. **Ringkasan & konfirmasi** -- semua pilihan di atas dicetak ulang, lalu
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

Base image `python:3.11-slim` + `pip install -e ".[gpu,garment]"` (lihat
komentar header `Dockerfile` untuk detail lengkap). Ekstra `garment` ikut
di-bake karena tahap inpaint aktif secara default; bobot checkpoint SAM-nya
tetap unduhan manual (lihat langkah 8). Ekstra `restoration` sengaja tidak
di-bake -- install manual di dalam container kalau dibutuhkan.

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
- **`ModelLoadError: segment-anything is not installed`** -- seharusnya tidak
  terjadi lagi (extra `garment` sudah ikut di `entrypoint.sh`/`Dockerfile`);
  kalau muncul, `entrypoint.sh` kemungkinan gagal/terlewat. Install manual:
  `pip install -e ".[garment]"`.
- **`ModelLoadError: SAM checkpoint not found at ...`** -- prefetch di langkah
  3b gagal atau dilewati (`HAROFRAME_PREFETCH=false`). Jalankan ulang manual:
  `python3 scripts/prefetch_models.py --skip-base --skip-loras`. Atau set
  `HAROFRAME_GENERATION__INPAINT__SAM__CHECKPOINT_PATH` ke checkpoint yang
  sudah Anda punya. Tidak butuh inpaint? Pakai `--no-inpaint`.
- **Generate pertama tetap mengunduh belasan GB padahal prefetch sudah OK** --
  biasanya `HAROFRAME_IDENTITY__BASE_SDXL_MODEL` atau `__DTYPE` diubah setelah
  prefetch jalan, jadi yang di-cache bukan file yang diminta loader (varian
  fp16 vs bobot default adalah file berbeda). Jalankan ulang
  `python3 scripts/prefetch_models.py` setelah mengganti keduanya.
- **`FAIL: inpainting is on but no prompt ...`** -- tahap inpaint aktif
  (default) tapi tidak ada deskripsi target areanya. Beri `--inpaint-prompt`,
  set `HAROFRAME_GENERATION__INPAINT__PROMPT`, atau matikan tahapnya dengan
  `--no-inpaint`. Script berhenti di sini sebelum memuat model apa pun, jadi
  tidak ada waktu/bandwidth yang terbuang.
- **`python3 -m pip install -e ".[gpu,garment]"` di `entrypoint.sh` sepertinya
  menginstal ulang/menimpa torch** -- cek apakah `/venv/main/bin/activate`
  benar-benar berhasil di-source (baris pertama log `entrypoint.sh` seharusnya
  mencetak "activating vast.ai PyTorch template venv"); kalau instance Anda
  bukan dari template PyTorch resmi, path venv-nya bisa berbeda/tidak ada,
  dan `pip` akan jatuh ke Python sistem lalu menginstal torch dari PyPI
  sendiri (tetap berfungsi, hanya lebih lambat di run pertama).
