# Kenali Kucing

Aplikasi Streamlit untuk mengidentifikasi jenis kucing: **hitam**, **kampung**, atau **tiga belang**.
Input dari unggah gambar atau kamera. Projek UTS Praktikum Pemrograman Big Data.

## Model

| File | Jenis | Cara kerja |
|---|---|---|
| `models/model.pt` | YOLOv8 (deteksi, ultralytics 8.4.197) | Menemukan kucing di gambar, menggambar kotak, memberi kelas dan keyakinan. Kelas: 0 Hitam, 1 Kampung, 2 Tiga Belang |
| `models/model.h5` | CNN Keras 3.13 (klasifikasi) | Menilai seluruh gambar (224x224) dan memberi probabilitas 3 kelas |

## Struktur

```
cat-classifier/
├── app.py            # halaman & navigasi
├── views.py          # markup HTML tiap bagian (nama pembuat: CREDIT_*)
├── classifier.py     # memuat model & prediksi
├── config.py         # LABELS, path model, parameter
├── requirements.txt
├── packages.txt      # paket sistem untuk Streamlit Cloud (OpenCV)
├── .streamlit/config.toml
├── assets/style.css  # gaya & animasi
├── static/cats/      # video kucing beranda (hitam, tiga-belang, kampung .mp4)
└── models/           # model.pt dan model.h5
```

## Jalankan lokal

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy lewat GitHub + Streamlit Community Cloud

```bash
git init
git add .
git commit -m "Kenali Kucing"
git branch -M main
git remote add origin https://github.com/USERNAME/NAMA-REPO.git
git push -u origin main
```

Lalu buka https://share.streamlit.io, **New app**, pilih repo, branch `main`, file `app.py`.
Di **Advanced settings** pilih **Python 3.11** atau 3.12.

Catatan: TensorFlow + PyTorch sekaligus cukup berat untuk memori server gratis. Jika aplikasi crash saat
memuat model, jalankan satu model saja (hapus library model lainnya dari `requirements.txt`).

## Pengaturan penting (`config.py`)

- `LABELS`: urutan kelas. Untuk `.pt` sudah pasti (Hitam, Kampung, Tiga Belang). Untuk `.h5`
  urutannya diasumsikan sama (alfabetis); jika hasil `.h5` tertukar, ubah urutan ini.
- `H5_RESCALE`: `"div255"` atau `"none"`. Harus sama dengan preprocessing saat training. Bisa dicoba
  langsung di aplikasi lewat *Pengaturan lanjutan*.
- `YOLO_CONF`: ambang keyakinan deteksi awal (bisa digeser di aplikasi).

## Video kucing di beranda

Taruh file `hitam`, `tiga-belang`, `kampung` (`.mp4`, `.webm`, `.webp`, `.gif`, `.jpg`, `.png`) di `static/cats/`.
Jika tidak ada, dipakai gambar kucing SVG. Pastikan Anda berhak memakai materinya. Kompres dengan ffmpeg:

```bash
ffmpeg -i asli.mp4 -an -t 6 -vf "scale=720:-2,fps=24" -c:v libx264 -crf 28 -preset slow -movflags +faststart hitam.mp4
```
