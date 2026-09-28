import io
import os
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st

try:
    import torch
except Exception:
    torch = None

try:
    import nibabel as nib
except Exception:
    nib = None

from models.unet import UNet
from inference.predict import count_lesions


st.set_page_config(
    page_title="МРТешечка — AI-анализ МРТ",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
<style>
:root { --bg:#06111b; --panel:#0a1724; --panel2:#0d1d2d; --cyan:#39e8ff; --cyan2:#19bdd4; --purple:#a34cff; --text:#edf7ff; --muted:#7e9aae; }
.stApp { background: radial-gradient(circle at 70% 10%, #10283a 0, #06111b 36%, #040b12 100%); color:var(--text); }
.block-container { max-width: 1500px; padding-top: 1.4rem; padding-bottom: 2rem; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#07131f,#050d15); border-right:1px solid #173040; }
[data-testid="stSidebar"] * { color:#d9edf7 !important; }
.hero { display:flex; align-items:center; justify-content:space-between; margin-bottom:1.2rem; }
.brand { display:flex; gap:16px; align-items:center; }
.logo { width:58px;height:58px;border-radius:18px;display:flex;align-items:center;justify-content:center;font-size:34px;background:linear-gradient(135deg,#19e4ff,#9a48ff);box-shadow:0 0 30px #2bdcf955; }
.brand h1 { margin:0; font-size:34px; letter-spacing:-1.5px; }
.brand p { margin:2px 0 0; color:#83a7bb; }
.badge { border:1px solid #1e4155; background:#091a29; border-radius:999px; padding:11px 18px; color:#bfeaf4; font-size:13px; }
.card { background:linear-gradient(180deg,rgba(13,31,46,.96),rgba(7,20,31,.96)); border:1px solid #1b394b; border-radius:18px; padding:20px; box-shadow:0 16px 50px #0005; }
.card h3 { margin:0 0 12px; }
.muted { color:#7893a5; }
.metric { border:1px solid #17475b; background:linear-gradient(135deg,#092738,#071b29); border-radius:16px; padding:18px; }
.metric .label { color:#38e5f5; font-size:14px; }
.metric .value { font-size:46px; font-weight:800; line-height:1.05; margin-top:6px; }
.result-row { display:flex; justify-content:space-between; align-items:center; padding:10px 0; border-bottom:1px solid #173141; }
.dot { width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:9px; box-shadow:0 0 10px currentColor; }
.c1 { color:#a84cff; } .c2 { color:#4da5ff; } .c3 { color:#2ee3c0; }
.tip { color:#78a1b5; font-size:13px; line-height:1.5; }
.stButton > button { border-radius:12px !important; border:1px solid #27cce3 !important; background:linear-gradient(90deg,#35d9ee,#42c8ff) !important; color:#03111a !important; font-weight:800 !important; }
.stDownloadButton > button { border-radius:12px !important; background:#0b1c2b !important; color:#dff9ff !important; border:1px solid #1e5a70 !important; }
[data-testid="stFileUploaderDropzone"] { background:#071a29; border:1px dashed #2b7f99; border-radius:14px; }
hr { border-color:#173141; }
.small-title { font-size:12px; color:#6f91a4; text-transform:uppercase; letter-spacing:1.5px; }
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Helpers
# -----------------------------
def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo)


def read_uploaded(uploaded):
    """Return display image and model-ready 3-channel image."""
    name = uploaded.name.lower()
    raw = uploaded.getvalue()

    if name.endswith(".nii") or name.endswith(".nii.gz"):
        if nib is None:
            raise RuntimeError("Для NIfTI установите nibabel: pip install nibabel")
        nii = nib.load(io.BytesIO(raw)) if hasattr(nib, "load") else None
        if nii is None:
            # nibabel.load expects a path; save temporary file.
            tmp = Path(".streamlit_tmp.nii.gz" if name.endswith(".gz") else ".streamlit_tmp.nii")
            tmp.write_bytes(raw)
            try:
                nii = nib.load(str(tmp))
            finally:
                tmp.unlink(missing_ok=True)
        vol = np.asarray(nii.get_fdata(), dtype=np.float32)
        if vol.ndim == 4:
            vol = vol[..., 0]
        axis = int(np.argmax(vol.shape))
        sl = np.take(vol, vol.shape[axis] // 2, axis=axis)
    else:
        sl = np.asarray(Image.open(io.BytesIO(raw)).convert("L"), dtype=np.float32)

    sl = normalize(sl)
    pil = Image.fromarray((sl * 255).astype(np.uint8)).convert("RGB")
    return pil, sl


def model_input(gray, size=256):
    img = Image.fromarray((normalize(gray) * 255).astype(np.uint8)).resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    # The project's U-Net expects [DWI, ADC, FLAIR]. For a single uploaded image
    # we use the same normalized slice in all three channels.
    return np.stack([arr, arr, arr], axis=0)


@st.cache_resource(show_spinner=False)
def load_checkpoint(path):
    if torch is None:
        raise RuntimeError("PyTorch не установлен")
    model = UNet()
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def run_inference(model, arr, min_size):
    if model is None:
        # Demo mode: create a small soft center region so the UI remains usable
        # without shipping medical model weights.
        h, w = arr.shape[1:]
        yy, xx = np.mgrid[:h, :w]
        cx, cy = w * 0.56, h * 0.49
        sigma = min(h, w) * 0.12
        prob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)).astype(np.float32)
        mask = prob > 0.56
    else:
        with torch.no_grad():
            x = torch.from_numpy(arr).float().unsqueeze(0)
            prob = torch.sigmoid(model(x)).squeeze().cpu().numpy()
        mask = prob > 0.5

    count, labels, sizes = count_lesions(mask, min_size=max(1, int(min_size)))
    confidence = float(np.clip(prob[mask].mean() if mask.any() else prob.max(), 0, 1))
    return prob, mask, labels, sizes, count, confidence


def overlay_image(gray, labels):
    base = np.stack([gray, gray, gray], axis=-1)
    base = np.clip(base * 255, 0, 255).astype(np.uint8)
    # Resize label map to display size.
    lab = Image.fromarray(labels.astype(np.int32)).resize((base.shape[1], base.shape[0]), Image.Resampling.NEAREST)
    lab = np.asarray(lab)
    out = base.astype(np.float32)
    colors = np.array([[163,76,255], [77,165,255], [46,227,192]], dtype=np.float32)
    for i in range(1, int(lab.max()) + 1):
        idx = lab == i
        if idx.any():
            col = colors[(i - 1) % len(colors)]
            out[idx] = 0.45 * out[idx] + 0.55 * col
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## 🧠 **МРТешечка**")
    st.caption("AI-анализ МРТ")
    st.markdown("---")
    page = st.radio("Навигация", ["Главная", "Анализ", "История", "О проекте"], index=0)
    st.markdown("---")
    st.markdown("### ⚙️ Параметры")
    min_size = st.slider("Минимальный размер очага, px", 1, 500, 20)
    checkpoint = st.text_input("Путь к весам U-Net", value="", placeholder="например: weights/best.pth")
    st.markdown("<div class='tip'>Без файла весов приложение запускается в демо-режиме интерфейса. Для реального анализа укажите обученные веса модели.</div>", unsafe_allow_html=True)


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
<div class="hero">
  <div class="brand">
    <div class="logo">🧠</div>
    <div><h1>МРТешечка</h1><p>AI-анализ МРТ</p></div>
  </div>
  <div class="badge">Считаем очаги. Смотрим внимательнее. ✨</div>
</div>
""",
    unsafe_allow_html=True,
)


if page == "О проекте":
    st.markdown("<div class='card'><h2>Что умеет МРТешечка?</h2><p class='muted'>Интерфейс для сегментации очагов на МРТ с использованием U-Net из этого проекта. Результат включает маску, число отдельных связных очагов и их размеры.</p></div>", unsafe_allow_html=True)
    st.info("Важно: интерфейс не является медицинской системой диагностики. Результат модели должен оцениваться специалистом.")
    st.stop()

if page == "История":
    st.markdown("<div class='card'><h2>История анализов</h2><p class='muted'>В этой версии история хранится только в текущей сессии браузера.</p></div>", unsafe_allow_html=True)
    if "history" not in st.session_state or not st.session_state.history:
        st.info("Пока анализов нет. Загрузите изображение на главной странице.")
    else:
        for item in reversed(st.session_state.history):
            st.markdown(f"**{item['name']}** — найдено очагов: **{item['count']}**, уверенность: **{item['confidence']:.0%}**")
            st.divider()
    st.stop()


# -----------------------------
# Main upload / analysis
# -----------------------------
left, center, right = st.columns([0.95, 1.65, 1.0], gap="large")

with left:
    st.markdown("<div class='card'><div class='small-title'>Загрузить изображение</div><h3>Кидай МРТ сюда 👇</h3>", unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["png", "jpg", "jpeg", "nii", "gz"], label_visibility="collapsed")
    st.markdown("<div class='tip'>Поддерживаются PNG, JPG и NIfTI (.nii / .nii.gz). Для NIfTI берётся центральный срез.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='card'><div class='small-title'>Быстрый старт</div><h3>Примеры</h3><p class='tip'>Загрузите любой тестовый срез, чтобы посмотреть интерфейс.</p></div>", unsafe_allow_html=True)

with center:
    if uploaded is None:
        st.markdown("<div class='card' style='min-height:560px;display:flex;align-items:center;justify-content:center;text-align:center'><div><div style='font-size:70px'>🧠</div><h2>Загрузите МРТ</h2><p class='muted'>И посмотрим, что там нашёл наш маленький AI-детектив.</p></div></div>", unsafe_allow_html=True)
        display_pil = None
        result = None
    else:
        try:
            display_pil, gray = read_uploaded(uploaded)
            arr = model_input(gray)
            model = None
            if checkpoint.strip():
                if not os.path.exists(checkpoint.strip()):
                    st.warning(f"Файл весов не найден: {checkpoint}")
                else:
                    model = load_checkpoint(checkpoint.strip())

            prob, mask, labels, sizes, count, confidence = run_inference(model, arr, min_size)
            overlay = overlay_image(np.asarray(display_pil.convert("L"), dtype=np.float32) / 255.0, labels)
            st.markdown("<div class='card'><div class='small-title'>Изображение</div><h3>Срез МРТ + сегментация</h3></div>", unsafe_allow_html=True)
            tabs = st.tabs(["Оригинал", "Маска", "Наложение"])
            with tabs[0]: st.image(display_pil, use_container_width=True)
            with tabs[1]: st.image(mask.astype(np.uint8) * 255, use_container_width=True)
            with tabs[2]: st.image(overlay, use_container_width=True)

            if "history" not in st.session_state:
                st.session_state.history = []
            if not st.session_state.history or st.session_state.history[-1].get("name") != uploaded.name:
                st.session_state.history.append({"name": uploaded.name, "count": count, "confidence": confidence})

        except Exception as exc:
            st.error(f"Не удалось обработать файл: {exc}")
            display_pil = None
            result = None
            count = 0
            sizes = []
            confidence = 0.0

with right:
    st.markdown("<div class='card'><div class='small-title'>Результаты анализа</div><h3>✨ Что нашли</h3>", unsafe_allow_html=True)
    count_value = locals().get("count", 0)
    confidence_value = locals().get("confidence", 0.0)
    st.markdown(f"<div class='metric'><div class='label'>Найдено очагов</div><div class='value'>{count_value}</div></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("**Размеры очагов**")
    sizes = locals().get("sizes", [])
    if sizes:
        for i, s in enumerate(sizes, 1):
            cls = ["c1", "c2", "c3"][(i - 1) % 3]
            st.markdown(f"<div class='result-row'><span><span class='dot {cls}'>●</span>Очаг {i}</span><b>{s} px</b></div>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='muted'>Загрузите изображение для подсчёта.</p>", unsafe_allow_html=True)

    st.markdown("<br><b>Оценка модели</b>", unsafe_allow_html=True)
    st.progress(float(confidence_value), text=f"Уверенность: {confidence_value:.0%}")

    st.markdown("<br>", unsafe_allow_html=True)
    if uploaded is not None and display_pil is not None:
        if st.button("▶  Анализировать ещё раз", use_container_width=True):
            st.rerun()
        report = {
            "file": uploaded.name,
            "lesion_count": count_value,
            "lesion_sizes_px": sizes,
            "confidence": round(float(confidence_value), 4),
        }
        import json
        st.download_button("↓  Скачать результаты", json.dumps(report, ensure_ascii=False, indent=2), file_name="results.json", mime="application/json", use_container_width=True)
    else:
        st.button("▶  Анализировать ещё раз", disabled=True, use_container_width=True)
        st.download_button("↓  Скачать результаты", "{}", file_name="results.json", disabled=True, use_container_width=True)

st.markdown("<br><div class='tip'>⚠️ Демо-режим используется, если веса U-Net не подключены. Не используйте демо-результат для медицинских решений.</div>", unsafe_allow_html=True)
