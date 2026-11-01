"""Pembuat HTML untuk tiap bagian halaman. Murni string, tanpa dependensi Streamlit."""
from __future__ import annotations

import numpy as np

from config import CLASS_INFO, LABELS, LOW_CONFIDENCE, MEDIA_URL, STATIC_DIR

A = 'target="_self"'
PAGES = ["beranda", "identifikasi", "jenis", "panduan", "tentang"]
GRADIENT = "linear-gradient(135deg,#17331F 0%,#3F6B45 55%,#8AA45A 100%)"

NAV = [
    ("beranda", "BERANDA", []),
    ("identifikasi", "IDENTIFIKASI", [
        ("Unggah gambar", "?page=identifikasi&src=upload"),
        ("Ambil foto dengan kamera", "?page=identifikasi&src=kamera"),
    ]),
    ("jenis", "JENIS KUCING", [(v["title"], f"?page=jenis&k={k}") for k, v in CLASS_INFO.items()]),
    ("panduan", "PANDUAN", []),
    ("tentang", "TENTANG", []),
]


def header(page: str) -> str:
    logo = (
        '<svg class="logo" width="42" height="42" viewBox="0 0 48 48" aria-hidden="true">'
        '<path d="M6 7 18 15h12L42 7v21c0 10-8 16-18 16S6 38 6 28Z" fill="#111111"/>'
        '<circle class="eye" cx="17" cy="28" r="3.2" fill="#F5D400"/>'
        '<circle class="eye" cx="31" cy="28" r="3.2" fill="#F5D400"/></svg>'
    )
    nav = mobile = ""
    for key, label, subs in NAV:
        chev = '<i class="chev"></i>' if subs else ""
        dd = (
            '<div class="dd">' + "".join(f'<a href="{h}" {A}>{t}</a>' for t, h in subs) + "</div>"
            if subs else ""
        )
        on = " on" if key == page else ""
        nav += f'<div class="nv{on}"><a href="?page={key}" {A}>{label}{chev}</a>{dd}</div>'
        mobile += f'<a href="?page={key}" {A}>{label}</a>'
    return (
        '<div class="fullbleed"><div class="strip"></div><div class="site-header">'
        f'<a class="brand" href="?page=beranda" {A}>{logo}<span>KENALI<br>KUCING</span></a>'
        f"<nav>{nav}</nav>"
        f'<div class="cta"><a class="y" href="?page=panduan" {A}>PANDUAN</a>'
        f'<a class="o" href="?page=identifikasi" {A}>COBA SEKARANG<i class="ar"></i></a></div></div>'
        f'<details class="mnav"><summary>☰ MENU</summary><div class="mlist">{mobile}</div></details></div>'
    )


TAIL_D = "M150 196C196 198 198 132 172 120"
_TABBY_HEAD = ('<g stroke="#6B4A2A" stroke-width="5" stroke-linecap="round" fill="none">'
               '<path d="M100 42v15M86 45l3 12M114 45l-3 12M58 88h12M60 98l11-3M142 88h-12M140 98l-11-3"/></g>')
_TABBY_BODY = ('<g stroke="#6B4A2A" stroke-width="5" stroke-linecap="round" fill="none">'
               '<path d="M58 150q14 8 28 2M56 170q16 8 32 2M142 150q-14 8-28 2M144 170q-16 8-32 2"/></g>')
COATS = {
    "hitam": dict(fur="#1A1A1A", edge="rgba(255,255,255,.12)", eye="#F5D400", earl="#1A1A1A", earr="#1A1A1A",
                  whisk="#A8A8A8", mouth="#8A8A8A", tail="#1A1A1A", head="", body="", tailx=""),
    "tiga belang": dict(fur="#FBFAF4", edge="rgba(0,0,0,.16)", eye="#7FB83E", earl="#EE7B07", earr="#1A1A1A",
                        whisk="#8A8A8A", mouth="#777", tail="#1A1A1A",
                        head='<ellipse cx="70" cy="60" rx="32" ry="28" fill="#EE7B07"/><ellipse cx="134" cy="62" rx="28" ry="24" fill="#1A1A1A"/>',
                        body='<ellipse cx="66" cy="150" rx="30" ry="30" fill="#EE7B07"/><ellipse cx="138" cy="190" rx="34" ry="28" fill="#1A1A1A"/>',
                        tailx=""),
    "kampung": dict(fur="#B88A55", edge="rgba(0,0,0,.2)", eye="#E8A21A", earl="#B88A55", earr="#B88A55",
                    whisk="#F1E6D6", mouth="#5A3A1E", tail="#B88A55", head=_TABBY_HEAD, body=_TABBY_BODY,
                    tailx=f'<path d="{TAIL_D}" fill="none" stroke="#6B4A2A" stroke-width="16" stroke-dasharray="7 13"/>'),
}


def cat_svg(kind: str, delay: float = 0.0, uid: str = "") -> str:
    """Kucing duduk beranimasi (ekor, kedip, telinga, napas) dalam satu SVG."""
    c = COATS[kind]
    k = uid + kind.replace(" ", "-")  # id clipPath harus unik per instance
    eyes = "".join(
        f'<g class="blink"><ellipse cx="{x}" cy="86" rx="9" ry="10" fill="{c["eye"]}"/>'
        f'<ellipse cx="{x}" cy="86" rx="3.2" ry="8" fill="#111"/><circle cx="{x + 3}" cy="82" r="2.2" fill="#fff"/></g>'
        for x in (80, 120)
    )
    whiskers = (f'<g stroke="{c["whisk"]}" stroke-width="1.6" stroke-linecap="round">'
                '<path d="M70 102 40 97M70 107 40 111M130 102 160 97M130 107 160 111"/></g>')
    return (
        f'<svg class="catsvg" viewBox="0 0 220 236" style="--d:{delay}s" role="img" aria-label="Kucing {kind}">'
        f'<defs><clipPath id="hc-{k}"><circle cx="100" cy="85" r="45"/></clipPath>'
        f'<clipPath id="bc-{k}"><ellipse cx="100" cy="170" rx="50" ry="52"/></clipPath></defs>'
        '<ellipse cx="100" cy="227" rx="66" ry="6" fill="rgba(0,0,0,.28)"/>'
        f'<g class="tail"><path d="{TAIL_D}" fill="none" stroke="{c["tail"]}" stroke-width="16" stroke-linecap="round"/>{c["tailx"]}</g>'
        f'<g class="body"><g clip-path="url(#bc-{k})"><ellipse cx="100" cy="170" rx="50" ry="52" fill="{c["fur"]}"/>{c["body"]}</g>'
        f'<ellipse cx="78" cy="206" rx="15" ry="20" fill="{c["fur"]}" stroke="{c["edge"]}"/>'
        f'<ellipse cx="122" cy="206" rx="15" ry="20" fill="{c["fur"]}" stroke="{c["edge"]}"/></g>'
        '<g class="head">'
        f'<polygon points="58,64 55,10 99,42" fill="{c["earl"]}"/><polygon points="66,52 65,26 86,43" fill="#F4B6B6" opacity=".85"/>'
        f'<g class="ear-r"><polygon points="142,64 145,10 101,42" fill="{c["earr"]}"/><polygon points="134,52 136,26 115,43" fill="#F4B6B6" opacity=".85"/></g>'
        f'<g clip-path="url(#hc-{k})"><circle cx="100" cy="85" r="45" fill="{c["fur"]}"/>{c["head"]}</g>'
        f'{eyes}<polygon points="95,100 105,100 100,106" fill="#E88C9A"/>'
        f'<path d="M100 106v4M100 110q-6 6-12 2M100 110q6 6 12 2" fill="none" stroke="{c["mouth"]}" stroke-width="1.8" stroke-linecap="round"/>'
        f"{whiskers}</g></svg>"
    )


MEDIA_EXTS = (".mp4", ".webm", ".webp", ".gif", ".jpg", ".jpeg", ".png")


def cat_visual(kind: str, delay: float = 0.0, uid: str = "") -> tuple[str, bool]:
    """Foto/video asli jika ada di static/cats/, kalau tidak gambar SVG. Mengembalikan (html, asli?)."""
    slug = kind.replace(" ", "-")
    for ext in MEDIA_EXTS:
        f = STATIC_DIR / f"{slug}{ext}"
        if f.exists():
            src = f"{MEDIA_URL}/{slug}{ext}?v={int(f.stat().st_mtime)}"
            if ext in (".mp4", ".webm"):
                m = (f'<video class="cat-media" src="{src}" autoplay muted loop playsinline '
                     f'preload="metadata" aria-label="Kucing {kind}"></video>')
            else:
                m = f'<img class="cat-media" src="{src}" alt="Kucing {kind}" loading="lazy">'
            return f'<div class="frame">{m}</div>', True
    return cat_svg(kind, delay, uid), False


def hero_cats() -> str:
    slides = ""
    for i, k in enumerate(CLASS_INFO):
        vis, real = cat_visual(k, i * 0.7, "hero-")
        slides += f'<div class="cat{" real" if real else ""}">{vis}<span class="lbl">{k.upper()}</span></div>'
    return f'<div class="hero-cat" aria-hidden="true">{slides}</div>'


def pals() -> str:
    body = "".join(
        f'<a class="pal" style="--c:{v["color"]}" href="?page=jenis&k={k}" {A}>'
        f'<div class="pal-cat">{cat_visual(k, i * 0.9, "pal-")[0]}</div><h3>{v["title"]}</h3>'
        f'<span class="go">LIHAT CIRI<i class="ar"></i></span></a>'
        for i, (k, v) in enumerate(CLASS_INFO.items())
    )
    return f'<div class="grid">{body}</div>'


def hero(bg: str = GRADIENT) -> str:
    return (
        f'<section class="hero" style="--hero-bg:{bg}"><div class="hero-bg"></div>{hero_cats()}<div class="hero-text">'
        "<h1>Kenali jenis kucingmu</h1>"
        "<p>Unggah foto atau ambil gambar dengan kamera, lalu lihat apakah kucingmu "
        "termasuk hitam, tiga belang, atau kampung.</p>"
        '<div class="hero-actions">'
        f'<a class="btn" href="?page=identifikasi&src=upload" {A}>UNGGAH FOTO<i class="ar"></i></a>'
        f'<a class="btn ghost" href="?page=identifikasi&src=kamera" {A}>BUKA KAMERA<i class="ar"></i></a>'
        "</div></div></section>"
    )


def marquee() -> str:
    group = "".join(f"<span>{k.upper()} ✦</span>" for k in ["hitam", "tiga belang", "kampung"] * 3)
    return f'<div class="fullbleed marquee"><div class="mq-track">{group}{group}</div></div>'


def title(text: str) -> str:
    return f'<div class="title">{text}</div>'


def banner(heading: str, sub: str) -> str:
    return f'<div class="banner"><h1>{heading}</h1><p>{sub}</p></div>'


def tiles() -> str:
    items = [
        ("01", "IDENTIFIKASI", "Unggah foto atau pakai kamera, hasil keluar dalam hitungan detik.", "identifikasi", "#EE7B07"),
        ("02", "JENIS KUCING", "Kenali ciri kucing hitam, tiga belang, dan kampung.", "jenis", "#F5D400"),
        ("03", "PANDUAN FOTO", "Tips cahaya dan posisi agar hasil lebih akurat.", "panduan", "#5E8B4F"),
    ]
    body = "".join(
        f'<a class="tile" style="--c:{c}" href="?page={p}" {A}><span class="no">{n}</span>'
        f'<h3>{h}</h3><p>{d}</p><span class="go">BUKA<i class="ar"></i></span></a>'
        for n, h, d, p, c in items
    )
    return f'<div class="grid">{body}</div>'


def jenis_page(highlight: str | None = None) -> str:
    cards = ""
    for k, v in CLASS_INFO.items():
        hl = " hl" if k == highlight else ""
        cards += (
            f'<div class="card{hl}" data-k="{k}" style="--c:{v["color"]}"><h3>{v["title"]}</h3>'
            f'<p>{v["desc"]}</p><p>{v["ciri"]}</p>'
            f'<a class="more" href="?page=identifikasi&src=upload" {A}>COBA IDENTIFIKASI<i class="ar"></i></a></div>'
        )
    return banner("Jenis kucing", "Tiga jenis yang dikenali model, beserta ciri utamanya.") + f'<div class="title">Pilih jenis</div><div class="grid">{cards}</div>'


def panduan_page() -> str:
    do = [
        ("Cahaya cukup", "Foto di tempat terang agar warna bulu terlihat jelas.", "#F5D400"),
        ("Kucing terlihat utuh", "Pastikan tubuh kucing mengisi sebagian besar gambar.", "#EE7B07"),
        ("Satu kucing per foto", "Foto dengan banyak kucing bisa membingungkan model.", "#5E8B4F"),
    ]
    dont = [
        ("Jangan terlalu gelap", "Bayangan tebal membuat belang tiga sulit dibedakan.", "#1A1A1A"),
        ("Jangan pakai filter", "Filter warna mengubah corak bulu yang dibaca model.", "#1A1A1A"),
        ("Jangan terlalu jauh", "Kucing yang kecil di sudut gambar sulit dikenali.", "#1A1A1A"),
    ]

    def grid(items):
        return '<div class="grid">' + "".join(
            f'<div class="card" style="--c:{c}"><h3>{t}</h3><p>{d}</p></div>' for t, d, c in items
        ) + "</div>"

    return (
        banner("Panduan foto", "Beberapa kebiasaan kecil yang membuat hasil lebih akurat.")
        + title("Lakukan") + grid(do) + title("Hindari") + grid(dont)
    )


def tentang_page() -> str:
    steps = [
        ("1. Unggah", "Pilih foto dari perangkat atau ambil langsung dengan kamera.", "#EE7B07"),
        ("2. Analisis", "Model membandingkan pola warna dan corak bulu pada foto.", "#F5D400"),
        ("3. Hasil", "Anda mendapat jenis kucing beserta persentase keyakinan.", "#5E8B4F"),
    ]
    cards = "".join(f'<div class="card" style="--c:{c}"><h3>{t}</h3><p>{d}</p></div>' for t, d, c in steps)
    return (
        banner("Tentang", "Aplikasi ini memakai model pembelajaran mesin untuk membedakan kucing hitam, tiga belang, dan kampung. Hasilnya berupa perkiraan, bukan kepastian.")
        + title("Cara kerja") + f'<div class="grid">{cards}</div>'
    )


CREDIT_TITLE = "Projek UTS Praktikum Pemrograman Big Data"
CREDIT_NAME = "Rizki Aula Fikri"
CREDIT_NIM = "2308108010084"


def footer() -> str:
    links = "".join(f'<a href="?page={k}" {A}>{l}</a>' for k, l, _ in NAV)
    return (
        '<div class="fullbleed"><div class="site-footer"><strong>KENALI KUCING</strong>'
        "<p>Identifikasi jenis kucing dari foto. Hasil berupa perkiraan model.</p>"
        f'<div class="flinks">{links}</div>'
        f'<p class="fcredit">{CREDIT_TITLE} · {CREDIT_NAME} | {CREDIT_NIM}</p></div></div>'
        f'<div class="credit"><b>{CREDIT_TITLE}</b><span>Dibuat oleh {CREDIT_NAME} | {CREDIT_NIM}</span></div>'
        '<a class="totop" href="#top" target="_self" aria-label="Kembali ke atas"><i></i></a>'
    )


def result_html(model_name: str, probs: np.ndarray, note: str = "") -> str:
    order = np.argsort(probs)[::-1]
    top = LABELS[order[0]]
    conf = float(probs[order[0]]) * 100
    info = CLASS_INFO.get(top, {"title": top.title(), "desc": "", "color": "#111111"})
    ip, dp = f"{conf:.1f}".split(".")
    warn = ""
    if conf / 100 < LOW_CONFIDENCE:
        warn = '<div class="warn">Model kurang yakin. Coba foto dengan cahaya lebih terang dan kucing terlihat jelas.</div>'
    note_html = f'<p class="d">{note}</p>' if note else ""
    rows = ""
    for i in order:
        label = LABELS[i]
        color = CLASS_INFO.get(label, {}).get("color", "#111111")
        pct = probs[i] * 100
        rows += (
            f'<div class="row"><div class="t"><span>{label.title()}</span><span>{pct:.1f}%</span></div>'
            f'<div class="track"><div class="fill" style="width:{pct:.1f}%;background:{color}"></div></div></div>'
        )
    return (
        f'<div class="result" style="--c:{info["color"]}"><div class="m">Model: {model_name}</div>'
        f'<h2>{info["title"]}</h2><p class="d">{info["desc"]}</p>'
        f'<div class="conf">Keyakinan <span class="num" style="--v:{ip}"></span>.{dp}%</div>{note_html}{warn}{rows}</div>'
    )


def no_detection_html(model_name: str) -> str:
    return (
        '<div class="result" style="--c:#8A8A85"><div class="m">Model: ' + model_name + "</div>"
        "<h2>Kucing tidak terdeteksi</h2>"
        '<p class="d">Model tidak menemukan kucing pada gambar ini. Coba foto yang lebih jelas, '
        "atau turunkan ambang keyakinan di Pengaturan lanjutan.</p></div>"
    )
