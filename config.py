"""Konfigurasi aplikasi. Sesuaikan nilai di sini dengan model Anda."""
from pathlib import Path

BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"
PT_PATH = BASE_DIR / "models" / "model.pt"    # YOLOv8 (deteksi): Hitam, Kampung, Tiga Belang
H5_PATH = BASE_DIR / "models" / "model.h5"    # CNN Keras (klasifikasi), input 224x224x3
HERO_PATH = ASSETS_DIR / "hero.jpg"           # foto banner (opsional)

# Urutan kelas. Model YOLO Anda: 0=Hitam, 1=Kampung, 2=Tiga Belang.
# Untuk model .h5 urutannya DIASUMSIKAN sama (alfabetis, bawaan flow_from_directory / image_dataset_from_directory).
# Jika hasil .h5 tertukar, ubah urutan ini.
LABELS = ["hitam", "kampung", "tiga belang"]

YOLO_IMGSZ = 640           # ukuran gambar saat training YOLO
YOLO_CONF = 0.25           # ambang keyakinan deteksi (bisa diubah di aplikasi)
H5_RESCALE = "div255"      # "div255" (0-1) atau "none" (0-255). Bisa diubah di aplikasi.
LOW_CONFIDENCE = 0.60      # di bawah ini hasil ditandai kurang yakin

# Warna kotak deteksi pada gambar
BOX_COLORS = {"hitam": "#F5D400", "kampung": "#7CC36B", "tiga belang": "#EE7B07"}

CLASS_INFO = {
    "hitam": {
        "title": "Kucing hitam",
        "desc": "Bulu hitam pekat hampir di seluruh tubuh.",
        "ciri": "Warna seragam gelap; di bawah sinar terang bulunya bisa tampak kecokelatan.",
        "color": "#1A1A1A",
    },
    "tiga belang": {
        "title": "Kucing tiga belang",
        "desc": "Bulu tiga warna: putih, hitam, dan oranye.",
        "ciri": "Bercak warna terpisah jelas, hampir selalu berjenis kelamin betina.",
        "color": "#EE7B07",
    },
    "kampung": {
        "title": "Kucing kampung",
        "desc": "Kucing domestik berbulu pendek dengan corak beragam.",
        "ciri": "Corak bervariasi: polos, lurik, atau dua warna, tanpa ras tertentu.",
        "color": "#5E8B4F",
    },
}

# Video kucing untuk beranda (opsional). Nama file: hitam, tiga-belang, kampung (.mp4/.webm/.webp/.gif/.jpg/.png)
STATIC_DIR = BASE_DIR / "static" / "cats"
MEDIA_URL = "app/static/cats"
