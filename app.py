"""Kenali Kucing: klasifikasi jenis kucing (hitam / tiga belang / kampung).

Navigasi memakai query param (?page=...), tampilan ada di views.py dan assets/style.css.
"""
from __future__ import annotations

import base64
import io

import streamlit as st
from PIL import Image, ImageOps

import views
from classifier import run_model
from config import ASSETS_DIR, H5_PATH, H5_RESCALE, HERO_PATH, LABELS, PT_PATH, YOLO_CONF

st.set_page_config(page_title="Kenali Kucing", page_icon="🐈", layout="wide", initial_sidebar_state="collapsed")


def md(html: str) -> None:
    st.markdown(html, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_css() -> str:
    return (ASSETS_DIR / "style.css").read_text(encoding="utf-8")


@st.cache_data(show_spinner=False)
def hero_background() -> str:
    """Foto di assets/hero.jpg jika ada, kalau tidak gradien hijau."""
    if HERO_PATH.exists():
        img = ImageOps.exif_transpose(Image.open(HERO_PATH)).convert("RGB")
        img.thumbnail((1600, 1600))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=82)
        return f"url(data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()})"
    return views.GRADIENT


def describe(p) -> str:
    """Ringkasan deteksi YOLO, mis. '2 kucing terdeteksi: 1 Hitam, 1 Kampung'."""
    counts = {}
    for d in p.detections:
        counts[LABELS[d.idx].title()] = counts.get(LABELS[d.idx].title(), 0) + 1
    return f"{len(p.detections)} kucing terdeteksi: " + ", ".join(f"{n} {k}" for k, n in counts.items())


def identify_section() -> None:
    available = {}
    if PT_PATH.exists():
        available["YOLO (.pt) · deteksi"] = ("pt", PT_PATH)
    if H5_PATH.exists():
        available["Keras (.h5) · klasifikasi"] = ("h5", H5_PATH)
    if not available:
        st.error(
            "File model tidak ditemukan. Letakkan file di folder `models/` dengan nama "
            "`model.pt` dan/atau `model.h5`, atau ubah `PT_PATH` / `H5_PATH` di `config.py`."
        )
        return

    default_src = 1 if st.query_params.get("src") == "kamera" else 0
    col_a, col_b = st.columns([1.2, 1], gap="large")
    with col_a:
        source = st.radio("Sumber gambar", ["Unggah gambar", "Kamera"], index=default_src, horizontal=True)
    with col_b:
        options = list(available) + (["Bandingkan keduanya"] if len(available) > 1 else [])
        choice = st.selectbox("Model", options, disabled=len(options) == 1)
    selected = list(available) if choice == "Bandingkan keduanya" else [choice]

    conf, rescale = YOLO_CONF, H5_RESCALE
    with st.expander("Pengaturan lanjutan"):
        if any(available[n][0] == "pt" for n in selected):
            conf = st.slider("Ambang keyakinan deteksi (YOLO)", 0.05, 0.90, YOLO_CONF, 0.05)
        if any(available[n][0] == "h5" for n in selected):
            label = st.radio(
                "Skala piksel untuk model Keras",
                ["Bagi 255 (0–1)", "Tanpa skala (0–255)"],
                index=0 if H5_RESCALE == "div255" else 1,
                help="Pilih sama dengan saat training. Jika hasil .h5 terasa janggal, coba opsi lainnya.",
            )
            rescale = "div255" if label.startswith("Bagi") else "none"

    if source == "Unggah gambar":
        file = st.file_uploader("Pilih foto kucing", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    else:
        file = st.camera_input("Arahkan kamera ke kucing, lalu ambil foto")
    if file is None:
        st.info("Belum ada gambar. Unggah foto atau ambil foto dengan kamera untuk mulai.")
        return

    image = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
    results = []
    for name in selected:
        kind, path = available[name]
        try:
            with st.spinner(f"Menganalisis dengan {name}…"):
                results.append((name, run_model(kind, path, image, conf, rescale), None))
        except Exception as e:  # noqa: BLE001
            results.append((name, None, e))

    shown, caption = image, None
    for _, p, _ in results:
        if p is not None and p.annotated is not None:
            shown, caption = p.annotated, "Kotak menunjukkan hasil deteksi YOLO."
            break

    col_img, col_res = st.columns([1, 1.15], gap="medium")
    with col_img:
        st.image(shown, use_container_width=True, caption=caption)
    with col_res:
        for name, p, err in results:
            if err is not None:
                st.error(f"Gagal menjalankan model {name}.")
                st.exception(err)
            elif len(p.probs) != len(LABELS):
                st.error(f"Model mengeluarkan {len(p.probs)} kelas, tetapi LABELS berisi {len(LABELS)}. Periksa config.py.")
            elif p.detections is not None and not p.detections:
                md(views.no_detection_html(name))
            else:
                md(views.result_html(name, p.probs, describe(p) if p.detections else ""))


st.markdown(f"<style>{load_css()}</style>", unsafe_allow_html=True)
page = st.query_params.get("page", "beranda")
if page not in views.PAGES:
    page = "beranda"

md('<div id="top"></div>')
md(views.header(page))

if page == "beranda":
    md(views.hero(hero_background()))
    md(views.marquee())
    md(views.title("Mulai dari sini"))
    md(views.tiles())
    md(views.title("Kenali tiga sahabat"))
    md(views.pals())
elif page == "identifikasi":
    md(views.banner("Identifikasi kucing", "Unggah foto atau gunakan kamera. Hasil muncul dalam beberapa detik."))
    md('<div style="height:1.2rem"></div>')
    identify_section()
elif page == "jenis":
    md(views.jenis_page(st.query_params.get("k")))
elif page == "panduan":
    md(views.panduan_page())
else:
    md(views.tentang_page())

md(views.footer())
