"""Memuat model dan prediksi.

- .pt  : YOLOv8 (ultralytics). Model deteksi: menemukan kucing, memberi kotak + kelas + keyakinan.
- .h5  : CNN Keras 3 (klasifikasi seluruh gambar).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from config import BOX_COLORS, LABELS, YOLO_IMGSZ


@dataclass
class Detection:
    idx: int                                  # indeks di LABELS
    conf: float
    box: tuple[float, float, float, float]    # x1, y1, x2, y2 (piksel)


@dataclass
class Prediction:
    probs: np.ndarray                         # skor per kelas, urutan = LABELS
    detections: list[Detection] | None = None  # None = klasifikasi seluruh gambar
    annotated: Image.Image | None = None      # gambar dengan kotak (khusus YOLO)


def to_probs(raw: np.ndarray) -> np.ndarray:
    """Pastikan output berupa probabilitas (softmax jika masih logits)."""
    raw = np.asarray(raw, dtype=np.float64).reshape(-1)
    if np.all(raw >= 0) and abs(raw.sum() - 1.0) < 1e-3:
        return raw
    e = np.exp(raw - raw.max())
    return e / e.sum()


# ─────────────────────────────── YOLO (.pt) ───────────────────────────────
@st.cache_resource(show_spinner="Memuat model YOLO…")
def load_pt(path: str):
    from ultralytics import YOLO

    return YOLO(path)


def draw_boxes(img: Image.Image, dets: list[Detection]) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    big = max(out.size)
    width = max(3, round(big / 200))
    size = max(14, round(big / 38))
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:  # Pillow lama
        font = ImageFont.load_default()
    for d in dets:
        label = LABELS[d.idx]
        color = BOX_COLORS.get(label, "#F5D400")
        x1, y1, x2, y2 = d.box
        draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
        text = f"{label.title()} {d.conf * 100:.0f}%"
        tb = draw.textbbox((0, 0), text, font=font)
        tw, th, pad = tb[2] - tb[0], tb[3] - tb[1], 5
        ty = max(0, y1 - th - 2 * pad)
        draw.rectangle((x1, ty, x1 + tw + 2 * pad, ty + th + 2 * pad), fill=color)
        draw.text((x1 + pad, ty + pad - tb[1]), text, fill="#111111", font=font)
    return out


def predict_pt(model, img: Image.Image, conf: float) -> Prediction:
    res = model.predict(img, imgsz=YOLO_IMGSZ, conf=conf, verbose=False, device="cpu")[0]
    names = {int(k): str(v).strip().lower() for k, v in model.names.items()}

    def to_idx(cls_id: int) -> int:
        name = names.get(cls_id, "")
        return LABELS.index(name) if name in LABELS else min(cls_id, len(LABELS) - 1)

    scores = np.zeros(len(LABELS))
    if getattr(res, "probs", None) is not None:  # model klasifikasi YOLO
        data = res.probs.data.cpu().numpy() if hasattr(res.probs.data, "cpu") else np.asarray(res.probs.data)
        for cid, p in enumerate(data):
            scores[to_idx(cid)] = float(p)
        return Prediction(scores)

    dets: list[Detection] = []
    boxes = res.boxes
    if boxes is not None and len(boxes) > 0:
        for cid, c, xyxy in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()):
            d = Detection(to_idx(int(cid)), float(c), tuple(float(v) for v in xyxy))
            dets.append(d)
            scores[d.idx] = max(scores[d.idx], d.conf)
    dets.sort(key=lambda d: -d.conf)
    return Prediction(scores, dets, draw_boxes(img, dets) if dets else img)


# ─────────────────────────────── Keras (.h5) ───────────────────────────────
@st.cache_resource(show_spinner="Memuat model Keras…")
def load_h5(path: str):
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    import keras

    return keras.models.load_model(path, compile=False)


def predict_h5(model, img: Image.Image, rescale: str) -> Prediction:
    shape = model.input_shape
    h, w = shape[1] or 224, shape[2] or 224
    arr = np.asarray(img.resize((w, h)), dtype=np.float32)
    if rescale == "div255":
        arr = arr / 255.0
    return Prediction(to_probs(model.predict(arr[None, ...], verbose=0)[0]))


# ─────────────────────────────── pintu masuk ───────────────────────────────
def run_model(kind: str, path: Path, img: Image.Image, conf: float, rescale: str) -> Prediction:
    if kind == "pt":
        return predict_pt(load_pt(str(path)), img, conf)
    return predict_h5(load_h5(str(path)), img, rescale)
