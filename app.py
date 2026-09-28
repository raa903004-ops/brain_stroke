import json
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import tempfile
from pathlib import Path
import time
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import streamlit as st
import torch
from scipy.ndimage import label as connected_components
from scipy.ndimage import center_of_mass
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    ResampleToMatchd,
    NormalizeIntensityd,
    ConcatItemsd,
    EnsureTyped,
)

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from models.unet3d import create_unet3d
from prepare_flair import resample_flair_to_dwi
from dicom_upload import (
    convert_series, describe_series, inspect_local_study, inspect_study, suggested_series,
)


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="МРТ Анализ",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATHS = {
    "ISLES · DWI + ADC + FLAIR": PROJECT_DIR / "weights" / "best_unet3d.pth",
    "SOOP · DWI + ADC": PROJECT_DIR / "weights" / "soop_dwi_adc_best.pth",
}
SOOP_NAME = "SOOP · DWI + ADC"
SOOP_FLAIR_NAME = "SOOP · DWI + ADC + FLAIR"
SOOP_FLAIR_EVAL = PROJECT_DIR / "soop_flair_evaluation.json"
if SOOP_FLAIR_EVAL.is_file() and (PROJECT_DIR / "weights" / "soop_flair_best.pth").is_file():
    MODEL_PATHS[SOOP_FLAIR_NAME] = PROJECT_DIR / "weights" / "soop_flair_best.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
POSTPROCESS_MODES = {
    "Баланс": {"threshold": 0.4, "dice": 0.625, "iou": 0.510},
    "Меньше ложных": {"threshold": 0.8, "dice": 0.615, "iou": 0.501},
}
MIN_COMPONENT_VOXELS = 10
DWI_SIGNAL_FLOOR = 0.5
if "result" in st.session_state and "voxel_volume_mm3" not in st.session_state["result"]:
    del st.session_state["result"]


# ============================================================
# CSS
# ============================================================
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 50% 15%, rgba(0,255,240,0.05), transparent 30%),
            linear-gradient(135deg, #061111 0%, #0a1b1b 50%, #061010 100%);
        color: #efffff;
    }

    .block-container {
        max-width: 1500px;
        padding: 22px 30px 35px 30px;
    }

    h1, h2, h3 {
        color: #efffff !important;
    }

    [data-testid="stFileUploader"] section {
        background: #0b2020 !important;
        border: 1px dashed #2d6664 !important;
        border-radius: 5px !important;
    }

    [data-testid="stFileUploader"] button {
        background: #102f2f !important;
        color: #72fff6 !important;
        border: 1px solid #367875 !important;
    }

    .stButton > button {
        background: #102f2f;
        color: #72fff6;
        border: 1px solid #397875;
        border-radius: 4px;
        min-height: 42px;
        font-weight: 600;
    }

    .stButton > button:hover {
        background: #173c3c;
        color: white;
        border-color: #72fff6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL
# ============================================================
@st.cache_resource
def load_model(model_name):
    device = torch.device(DEVICE)

    model = create_unet3d(in_channels=2 if model_name == SOOP_NAME else 3)

    checkpoint = torch.load(
        MODEL_PATHS[model_name],
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return model, device


# ============================================================
# INFERENCE PREPROCESSING
# ============================================================
def make_load_transform(keys):
    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
    ])

flair_resample = ResampleToMatchd(
    keys=["flair"],
    key_dst="dwi",
    mode="bilinear",
)

adc_resample = ResampleToMatchd(
    keys=["adc"],
    key_dst="dwi",
    mode="bilinear",
)

def make_finish_transform(keys):
    return Compose([
        NormalizeIntensityd(keys=keys, nonzero=True, channel_wise=True),
        ConcatItemsd(keys=keys, name="image", dim=0),
        EnsureTyped(keys=["image", *keys]),
    ])
def same_grid(image1, image2):
    shape1 = tuple(image1.shape[1:])
    shape2 = tuple(image2.shape[1:])

    affine1 = image1.affine
    affine2 = image2.affine

    if torch.is_tensor(affine1):
        affine1 = affine1.cpu().numpy()

    if torch.is_tensor(affine2):
        affine2 = affine2.cpu().numpy()

    return (
        shape1 == shape2
        and np.allclose(
            affine1,
            affine2,
            atol=1e-4
        )
    )


def save_uploaded_file(uploaded_file, folder):
    path = os.path.join(folder, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def _voxel_geometry(flair_tensor):
    """Return voxel volume (mm^3) and in-plane voxel area (mm^2)."""
    affine = flair_tensor.affine
    if torch.is_tensor(affine):
        affine = affine.detach().cpu().numpy()
    affine = np.asarray(affine, dtype=float)

    voxel_volume_mm3 = float(abs(np.linalg.det(affine[:3, :3])))
    voxel_sizes = np.linalg.norm(affine[:3, :3], axis=0)
    inplane_area_mm2 = float(voxel_sizes[0] * voxel_sizes[1])
    return voxel_volume_mm3, inplane_area_mm2


def postprocess_probability(probability, dwi_volume, threshold, use_signal_gate=True):
    """Require a DWI signal and remove isolated 3D predictions."""
    mask = probability >= threshold
    if use_signal_gate:
        mask &= dwi_volume > DWI_SIGNAL_FLOOR
    components, component_count = connected_components(mask)
    if component_count:
        sizes = np.bincount(components.ravel(), minlength=component_count + 1)
        mask &= sizes[components] >= MIN_COMPONENT_VOXELS
    return mask.astype(np.uint8)


def predict(dwi_path, adc_path, flair_path, model_name, _flair_prepared=False):
    use_flair = model_name != SOOP_NAME

    # Avoid decoding very large FLAIR volumes into RAM and then copying them to CUDA.
    if use_flair and not _flair_prepared and np.prod(nib.load(flair_path).shape[:3]) > 8_000_000:
        with tempfile.TemporaryDirectory() as prepared_dir:
            compact_flair = str(Path(prepared_dir) / "flair_on_dwi.nii.gz")
            resample_flair_to_dwi(flair_path, dwi_path, compact_flair)
            return predict(dwi_path, adc_path, compact_flair, model_name, _flair_prepared=True)

    keys = ["dwi", "adc", "flair"] if use_flair else ["dwi", "adc"]
    data = {"dwi": dwi_path, "adc": adc_path}
    if use_flair:
        data["flair"] = flair_path

    # ========================================================
    # 1. ЗАГРУЗКА + ORIENTATION
    # ========================================================

    t0 = time.perf_counter()

    data = make_load_transform(keys)(data)

    t1 = time.perf_counter()


    # ========================================================
    # 2. ПЕРЕНОС PREPROCESSING НА GPU
    # ========================================================

    preprocess_device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    data["dwi"] = data["dwi"].to(preprocess_device)
    data["adc"] = data["adc"].to(preprocess_device)
    if use_flair:
        data["flair"] = data["flair"].to(preprocess_device)

    if preprocess_device.type == "cuda":
        torch.cuda.synchronize()

    t2 = time.perf_counter()


    # ========================================================
    # 3. ПРОВЕРКА ГЕОМЕТРИИ
    # ========================================================

    flair_need_resample = use_flair and not same_grid(
        data["flair"],
        data["dwi"]
    )

    adc_need_resample = not same_grid(
        data["adc"],
        data["dwi"]
    )


    # ========================================================
    # 4. RESAMPLE
    # ========================================================

    if flair_need_resample:
        data = flair_resample(data)

    if adc_need_resample:
        data = adc_resample(data)

    # Важно для правильного измерения времени CUDA
    if preprocess_device.type == "cuda":
        torch.cuda.synchronize()

    t3 = time.perf_counter()


    # ========================================================
    # 5. NORMALIZATION + CONCAT
    # ========================================================

    data = make_finish_transform(keys)(data)

    if preprocess_device.type == "cuda":
        torch.cuda.synchronize()

    t4 = time.perf_counter()


    # ========================================================
    # 6. MODEL
    # ========================================================

    model, device = load_model(model_name)

    image = data["image"].unsqueeze(0)

    # Если preprocessing уже был на CUDA,
    # этот вызов практически ничего не стоит.
    image = image.to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t5 = time.perf_counter()


    # ========================================================
    # 7. INFERENCE
    # ========================================================

    with torch.inference_mode():

        if device.type == "cuda":

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                logits = sliding_window_inference(
                    inputs=image,
                    roi_size=(96, 96, 32) if not use_flair else (96, 96, 96),
                    sw_batch_size=1 if not use_flair else 2,
                    predictor=model,
                    overlap=0.25,
                )

        else:

            logits = sliding_window_inference(
                inputs=image,
                roi_size=(96, 96, 32) if not use_flair else (96, 96, 96),
                sw_batch_size=1,
                predictor=model,
                overlap=0.25,
            )


        probability = torch.sigmoid(logits)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t6 = time.perf_counter()


    # ========================================================
    # ВРЕМЯ
    # ========================================================

    st.write(
        f"Загрузка + Orientation: "
        f"{t1 - t0:.1f} сек"
    )

    st.write(
        f"Перенос на GPU: "
        f"{t2 - t1:.1f} сек"
    )

    st.write(
        f"Resample: "
        f"{t3 - t2:.1f} сек"
    )

    st.write(
        f"Normalize + объединение: "
        f"{t4 - t3:.1f} сек"
    )

    st.write(
        f"Подготовка модели: "
        f"{t5 - t4:.1f} сек"
    )

    st.write(
        f"Работа модели: "
        f"{t6 - t5:.1f} сек"
    )

    st.write(
        f"FLAIR resample: "
        f"{'да' if flair_need_resample else 'не требуется'}"
    )

    st.write(
        f"ADC resample: "
        f"{'да' if adc_need_resample else 'не требуется'}"
    )


    # ========================================================
    # RESULTS
    # ========================================================

    probability_np = (
        probability[0, 0]
        .float()
        .cpu()
        .numpy()
    )
    flair_volume = (
        data["flair"][0].float().cpu().numpy()
        if use_flair else None
    )

    dwi_volume = (
        data["dwi"][0]
        .float()
        .cpu()
        .numpy()
    )

    adc_volume = (
        data["adc"][0]
        .float()
        .cpu()
        .numpy()
    )

    # Candidate voxels must also show a DWI signal increase; reject tiny 3D islands.
    mask_np = postprocess_probability(
        probability_np, dwi_volume,
        0.5 if model_name in (SOOP_NAME, SOOP_FLAIR_NAME)
        else POSTPROCESS_MODES["Баланс"]["threshold"],
        use_signal_gate=model_name not in (SOOP_NAME, SOOP_FLAIR_NAME),
    )


    voxel_volume_mm3, inplane_area_mm2 = (
        _voxel_geometry(
            data["dwi"]
        )
    )


    return (
        flair_volume,
        dwi_volume,
        adc_volume,
        mask_np,
        probability_np,
        voxel_volume_mm3,
        inplane_area_mm2,
    )


def analyze_mask(mask, voxel_volume_mm3, inplane_area_mm2):

    labeled, raw_count = connected_components(mask > 0)

    # Быстро считаем размер всех компонент за один проход
    if raw_count > 0:
        component_sizes = np.bincount(
            labeled.ravel(),
            minlength=raw_count + 1
        )[1:]
    else:
        component_sizes = np.array([], dtype=np.int64)

    lesion_count = raw_count

    total_voxels = int(mask.sum())

    total_volume_cm3 = (
        total_voxels
        * voxel_volume_mm3
        / 1000.0
    )

    if component_sizes.size > 0:
        largest_voxels = int(component_sizes.max())

        largest_volume_cm3 = (
            largest_voxels
            * voxel_volume_mm3
            / 1000.0
        )
    else:
        largest_volume_cm3 = 0.0

    area_per_slice = (
        mask.sum(axis=(0, 1))
        * inplane_area_mm2
    )

    max_area_mm2 = (
        float(area_per_slice.max())
        if area_per_slice.size
        else 0.0
    )

    coords_text = "—"

    if total_voxels > 0:
        coords = center_of_mass(mask > 0)

        coords_text = (
            f"({coords[0]:.1f}, "
            f"{coords[1]:.1f}, "
            f"{coords[2]:.1f}) voxel"
        )

    return {
        "lesion_count": lesion_count,
        "total_volume_cm3": total_volume_cm3,
        "largest_volume_cm3": largest_volume_cm3,
        "max_area_mm2": max_area_mm2,
        "coords": coords_text,
    }


# ============================================================
# HEADER
# ============================================================
st.title("🧠 МРТ АНАЛИЗ")
st.caption("Сегментация возможных ишемических очагов")
st.caption(f"Устройство: {DEVICE.upper()}")
if DEVICE == "cuda":
    st.caption(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    st.warning(
        "CUDA недоступна в текущем Python-окружении. "
        "Запустите приложение командой: python -m streamlit run app.py"
    )

left, center, right = st.columns([0.95, 1.65, 1.0], gap="medium")


# ============================================================
# LEFT — UPLOAD + RUN
# ============================================================
with left:
    st.subheader("ЗАГРУЗКА МРТ")
    model_name = st.selectbox("Модель", list(MODEL_PATHS), key="model_name_v2")
    use_flair = model_name != SOOP_NAME
    input_format = st.radio("Формат снимков", ["NIfTI", "DICOM папка", "DICOM ZIP"],
                            horizontal=True, key="input_format")
    if "result" in st.session_state and (
        st.session_state["result"].get("model_name") != model_name
        or st.session_state["result"].get("input_format", "NIfTI") != input_format
    ):
        del st.session_state["result"]
    if model_name == SOOP_NAME:
        st.caption("SOOP: DWI + ADC, экспериментальная модель, лучший чекпоинт 60-й эпохи.")
    elif model_name == SOOP_FLAIR_NAME:
        st.caption("SOOP с совмещённым FLAIR: экспериментальная модель, внешний тест в отчёте ниже.")
    else:
        st.caption("ISLES: DWI + ADC + FLAIR.")

    dwi = adc = flair = None
    dicom_series = {}
    selected_series = {}
    if input_format == "NIfTI":
        st.markdown("**01 / DWI**")
        st.caption("Диффузионно-взвешенное изображение")
        dwi = st.file_uploader("DWI", type=["nii", "gz"], key="dwi",
                               label_visibility="collapsed")
        st.divider()
        st.markdown("**02 / ADC**")
        st.caption("Карта коэффициента диффузии")
        adc = st.file_uploader("ADC", type=["nii", "gz"], key="adc",
                               label_visibility="collapsed")
        st.divider()
        if use_flair:
            st.markdown("**03 / FLAIR**")
            st.caption("FLAIR изображение")
            flair = st.file_uploader("FLAIR", type=["nii", "gz"], key="flair",
                                     label_visibility="collapsed")
    else:
        st.caption("Выбери папку со всеми .dcm, включая подпапки, или ZIP исследования. "
                   "Затем проверь серии DWI, ADC и FLAIR.")
        if input_format == "DICOM папка":
            local_dicom_path = st.text_input(
                "Путь к папке DICOM на этом компьютере",
                placeholder=r"D:\путь\к\исследованию",
                help="Можно указать папку одного исследования; файлы остаются на вашем компьютере.",
            )
            study_files = st.file_uploader("Папка DICOM", type=["dcm", "dmc", "dicom"],
                                           accept_multiple_files="directory", key="dicom_folder")
            study_zip = None
        else:
            local_dicom_path = ""
            study_files = None
            study_zip = st.file_uploader("ZIP с DICOM", type=["zip"], key="dicom_zip",
                                         max_upload_size=1024)
        if study_files or study_zip or local_dicom_path.strip():
            try:
                dicom_series = (inspect_local_study(local_dicom_path.strip())
                                if local_dicom_path.strip() else inspect_study(study_files, study_zip))
                studies = sorted({group["study_uid"] for group in dicom_series.values()})
                if len(studies) > 1:
                    study_labels = {}
                    for index, study_uid in enumerate(studies, 1):
                        groups = [group for group in dicom_series.values()
                                  if group["study_uid"] == study_uid]
                        parts = Path(groups[0]["files"][0][0]).parts
                        folder_hint = next((part for part in parts
                                            if part.upper().startswith("P") and part[1:].isdigit()), None)
                        study_labels[study_uid] = (f"{folder_hint or f'Исследование {index}'} · "
                                                   f"{len(groups)} серии")
                    selected_study = st.selectbox(
                        "Исследование", studies, format_func=lambda key: study_labels[key],
                        key="dicom_study",
                    )
                    dicom_series = {key: group for key, group in dicom_series.items()
                                    if group["study_uid"] == selected_study}
                st.caption(f"Найдено {len(dicom_series)} DICOM-серий в исследовании.")
                for modality in (["dwi", "adc", "flair"] if use_flair else ["dwi", "adc"]):
                    suggested = suggested_series(dicom_series, modality)
                    choices = [None, *dicom_series]
                    selected_series[modality] = st.selectbox(
                        modality.upper(), choices,
                        index=choices.index(suggested) if suggested else 0,
                        format_func=lambda key: "Выберите серию" if key is None
                        else describe_series(dicom_series[key]),
                        key=f"dicom_series_{modality}",
                    )
            except Exception as exc:
                st.error(f"Не удалось прочитать DICOM: {exc}")

    st.divider()

    if st.button("ЗАПУСТИТЬ АНАЛИЗ", use_container_width=True):
        required = ["dwi", "adc", "flair"] if use_flair else ["dwi", "adc"]
        if input_format == "NIfTI" and (dwi is None or adc is None or (use_flair and flair is None)):
            st.error("Загрузите DWI и ADC" + (" и FLAIR." if use_flair else "."))
        elif input_format != "NIfTI" and not all(selected_series.get(key) for key in required):
            st.error("Загрузите DICOM и выберите серии " + ", ".join(key.upper() for key in required) + ".")
        elif input_format != "NIfTI" and len(set(selected_series[key] for key in required)) != len(required):
            st.error("Для DWI, ADC и FLAIR нужно выбрать разные DICOM-серии.")
        else:
            try:
                with st.spinner("Модель анализирует МРТ..."):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        if input_format == "NIfTI":
                            dwi_path = save_uploaded_file(dwi, temp_dir)
                            adc_path = save_uploaded_file(adc, temp_dir)
                            flair_path = save_uploaded_file(flair, temp_dir) if use_flair else None
                        else:
                            paths = {key: convert_series(dicom_series[selected_series[key]],
                                                         Path(temp_dir) / f"{key}.nii.gz")
                                     for key in required}
                            dwi_path, adc_path = paths["dwi"], paths["adc"]
                            flair_path = paths.get("flair")

                        (
                            flair_volume,
                            dwi_volume,
                            adc_volume,
                            mask,
                            probability,
                            voxel_volume_mm3,
                            inplane_area_mm2,
                        ) = predict(dwi_path, adc_path, flair_path, model_name)

                        t_stats = time.perf_counter()

                        stats = analyze_mask(
                            mask,
                            voxel_volume_mm3,
                            inplane_area_mm2,
                        )

                        st.write(
                            f"Анализ маски: "
                            f"{time.perf_counter() - t_stats:.1f} сек"
                        )

                        st.session_state["result"] = {
                            "flair": flair_volume,
                            "dwi": dwi_volume,
                            "adc": adc_volume,
                            "mask": mask,
                            "probability": probability,
                            "stats": stats,
                            "voxel_volume_mm3": voxel_volume_mm3,
                            "inplane_area_mm2": inplane_area_mm2,
                            "model_name": model_name,
                            "input_format": input_format,
                        }

                st.success("Анализ завершён")
            except Exception as e:
                st.exception(e)

    if input_format == "NIfTI" and (dwi is not None or adc is not None or flair is not None):
        st.subheader("СТАТУС ЗАГРУЗКИ")
        if dwi is not None:
            st.success("DWI загружен")
        if adc is not None:
            st.success("ADC загружен")
        if flair is not None:
            st.success("FLAIR загружен")
    elif input_format != "NIfTI" and dicom_series:
        st.caption("DICOM загружен. Проверь названия серий перед анализом.")


# ============================================================
# CENTER — VISUALIZATION & MODALITY SELECTOR
# ============================================================
with center:
    st.subheader("МРТ / Показ изображения")

    if "result" in st.session_state:
        result = st.session_state["result"]
        mode_name = st.radio(
            "Режим выделения:",
            options=list(POSTPROCESS_MODES),
            horizontal=True,
            index=1 if result["model_name"] in (SOOP_NAME, SOOP_FLAIR_NAME) else 0,
            key=("postprocess_mode_soop" if result["model_name"] == SOOP_NAME else
                 "postprocess_mode_soop_flair" if result["model_name"] == SOOP_FLAIR_NAME else
                 "postprocess_mode_isles"),
        )
        result["mask"] = postprocess_probability(
            result["probability"], result["dwi"],
            (0.5 if mode_name == "Баланс" else 0.8)
            if result["model_name"] in (SOOP_NAME, SOOP_FLAIR_NAME)
            else POSTPROCESS_MODES[mode_name]["threshold"],
            use_signal_gate=result["model_name"] not in (SOOP_NAME, SOOP_FLAIR_NAME),
        )
        result["stats"] = analyze_mask(
            result["mask"], result["voxel_volume_mm3"], result["inplane_area_mm2"]
        )
        mask = result["mask"]

        # Кнопка переключения модальности
        selected_modality = st.radio(
            "Режим отображения:",
            options=(["DWI", "ADC"] if result["model_name"] == SOOP_NAME
                     else ["FLAIR", "DWI", "ADC"]),
            horizontal=True,
            key="modality_selector"
        )

        selected_volume = result[selected_modality.lower()]

        if mask.any():
            lesion_per_slice = mask.sum(axis=(0, 1))
            slice_id = int(np.argmax(lesion_per_slice))
        else:
            slice_id = int(selected_volume.shape[2] // 2)

        vol_slice = selected_volume[:, :, slice_id]
        mask_slice = mask[:, :, slice_id]

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(vol_slice.T, cmap="gray", origin="lower")

        if mask_slice.any():
            ax.contour(
                mask_slice.T,
                levels=[0.5],
                colors="red",
                linewidths=2,
            )
            ax.set_title(f"Режим: {selected_modality} • Предсказанная область на срезе {slice_id}")
        else:
            ax.set_title(f"Режим: {selected_modality} • Маска пуста • Срез {slice_id}")

        ax.axis("off")
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

        st.info(f"Отображается: **{selected_modality}**. Красный контур — область, выделенная моделью; он не подтверждает диагноз.")
    else:
        st.info("Загрузите NIfTI или DICOM-серии и нажмите «ЗАПУСТИТЬ АНАЛИЗ».")


# ============================================================
# RIGHT — RESULTS + VALIDATION METRICS
# ============================================================
with right:
    st.subheader("РЕЗУЛЬТАТ")

    if "result" in st.session_state:
        stats = st.session_state["result"]["stats"]

        st.write(f"**Предсказанных областей:** {stats['lesion_count']}")
        st.write(
            f"**Общий объём маски:** {stats['total_volume_cm3']:.3f} см³"
        )
        st.write(
            f"**Объём крупнейшего очага:** {stats['largest_volume_cm3']:.3f} см³"
        )
        st.write(
            f"**Макс. площадь на срезе:** {stats['max_area_mm2']:.1f} мм²"
        )
        st.write(f"**Центр маски:** {stats['coords']}")
    else:
        st.write("**Предсказанных областей:** —")
        st.write("**Общий объём маски:** —")
        st.write("**Объём крупнейшего очага:** —")
        st.write("**Макс. площадь на срезе:** —")
        st.write("**Центр маски:** —")

    st.divider()
    st.markdown("#### Качество модели")

    mode_key = ("postprocess_mode_soop" if model_name == SOOP_NAME else
                "postprocess_mode_soop_flair" if model_name == SOOP_FLAIR_NAME else
                "postprocess_mode_isles")
    mode_name = st.session_state.get(
        mode_key, "Меньше ложных" if model_name in (SOOP_NAME, SOOP_FLAIR_NAME) else "Баланс")
    if model_name == SOOP_NAME:
        val_dice, val_iou = (0.6763, 0.5529) if mode_name == "Баланс" else (None, None)
    elif model_name == SOOP_FLAIR_NAME:
        report = json.loads(SOOP_FLAIR_EVAL.read_text(encoding="utf-8"))
        val_dice = report["soop"]["mean_dice"] if mode_name == "Баланс" else None
        val_iou = None
    else:
        val_dice = POSTPROCESS_MODES[mode_name]["dice"]
        val_iou = POSTPROCESS_MODES[mode_name]["iou"]
    if val_dice is not None:
        pro = val_dice * 100
        st.markdown(
            f"""
        <div style="
            width:100%;
            height:22px;
            background:#173737;
            border:1px solid #315b59;
            position:relative;
            overflow:hidden;
            margin-top:8px;
        ">
            <div style="
                width:{pro}%;
                height:100%;
                background:#72fff6;
            "></div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        st.markdown(f"**Dice на валидации: {val_dice:.3f} ({val_dice * 100:.1f}%)**")
        if val_iou is not None:
            st.markdown(f"**IoU на валидации: {val_iou:.3f} ({val_iou * 100:.1f}%)**")
    else:
        st.caption("Для режима «Меньше ложных» отдельная оценка качества ещё не проводилась.")
    if model_name == SOOP_NAME:
        st.caption("SOOP: лучший чекпоинт 60-й эпохи; метрики сырой маски при пороге 0.5 "
                   "на 143 отложенных размеченных случаях. После фильтрации и при другом пороге "
                   "показатели могут отличаться. Проверки на снимках без очагов не было.")
        st.warning("Модель может ошибочно выделять нормальные ткани, особенно на снимках "
                   "другого протокола. На другой выборке качество заметно хуже. "
                   "Наличие красного контура не подтверждает очаг.")
    elif model_name == SOOP_FLAIR_NAME:
        st.caption(f"SOOP: {report['soop']['cases']} отложенных случаев, Dice сырой маски "
                   f"при пороге 0.5. Внешняя проверка на {report['isles']['cases']} ISLES случаях: "
                   f"Dice {report['isles']['mean_dice']:.1%}. Проверки на снимках без очагов не было.")
        st.warning("Модель экспериментальная. Выделения на новых снимках могут быть ложными.")
    else:
        st.caption("ISLES: метрики сегментации на 38 размеченных случаях с инсультом. "
                   "Проверки на снимках без очагов не было.")

st.caption(
    "Исследовательский прототип. Результат модели не является медицинским заключением."
)
