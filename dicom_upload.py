"""Read a DICOM study uploaded as a folder or ZIP and convert selected series."""

from collections import defaultdict
import io
from pathlib import Path
import zipfile

import pydicom
import SimpleITK as sitk


EXTENSIONS = (".dcm", ".dmc", ".dicom")


def _members_from_zip(uploaded_zip):
    with zipfile.ZipFile(io.BytesIO(uploaded_zip.getvalue())) as archive:
        members = []
        for entry in archive.infolist():
            if entry.is_dir() or not entry.filename.lower().endswith(EXTENSIONS):
                continue
            if entry.file_size > 128 * 1024 * 1024:
                raise ValueError(f"Слишком большой DICOM-файл в ZIP: {entry.filename}")
            members.append((entry.filename, archive.read(entry)))
        return members


def inspect_study(uploaded_files=None, uploaded_zip=None):
    if uploaded_zip is not None:
        members = _members_from_zip(uploaded_zip)
    else:
        members = [(file.name, file.getvalue()) for file in uploaded_files or []]
    if not members:
        return {}
    return _inspect_members(members)


def inspect_local_study(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Папка не найдена: {folder}")
    files = [path for path in folder.rglob("*") if path.is_file()
             and path.suffix.lower() in EXTENSIONS]
    if not files:
        raise ValueError("В папке нет DICOM-файлов (.dcm).")
    if len(files) > 20000:
        raise ValueError("Слишком много файлов. Выберите папку одного исследования.")
    return _inspect_members([(str(path.relative_to(folder)), path) for path in files])


def _inspect_members(members):
    series = defaultdict(lambda: {"files": [], "description": "", "protocol": "", "image_type": ""})
    invalid = 0
    for name, data in members:
        try:
            source = data if isinstance(data, Path) else io.BytesIO(data)
            header = pydicom.dcmread(source, stop_before_pixels=True, force=True)
            series_uid = str(header.get("SeriesInstanceUID", ""))
            study_uid = str(header.get("StudyInstanceUID", ""))
            if not series_uid or not header.get("Rows") or not header.get("Columns"):
                invalid += 1
                continue
            key = f"{study_uid}|{series_uid}"
            group = series[key]
            group["files"].append((name, data))
            group["description"] = str(header.get("SeriesDescription", ""))
            group["protocol"] = str(header.get("ProtocolName", ""))
            group["image_type"] = str(header.get("ImageType", ""))
            group["number"] = str(header.get("SeriesNumber", ""))
            group["study_uid"] = study_uid
            group["series_uid"] = series_uid
            group["b_value"] = float(header.get("DiffusionBValue", 0) or 0)
            group["inversion_time"] = float(header.get("InversionTime", 0) or 0)
            group["echo_time"] = float(header.get("EchoTime", 0) or 0)
            group["repetition_time"] = float(header.get("RepetitionTime", 0) or 0)
            group["scanning_sequence"] = str(header.get("ScanningSequence", ""))
        except Exception:
            invalid += 1
    if not series:
        raise ValueError("В загрузке нет читаемых DICOM-серий (.dcm).")
    return dict(series)


def describe_series(group):
    label = group["description"] or group["protocol"]
    if not label:
        kind = inferred_modality(group)
        label = {"dwi": "DWI (b=1000)", "adc": "ADC", "flair": "вероятно FLAIR"}.get(
            kind, "неизвестная серия")
    return f"№{group.get('number', '?')} · {label} · {len(group['files'])} файл(ов)"


def inferred_modality(group):
    text = " ".join((group["description"], group["protocol"], group["image_type"])).lower()
    if "adc" in text or "apparent diffusion" in text:
        return "adc"
    if "flair" in text or "dark fluid" in text:
        return "flair"
    if any(hint in text for hint in ("dwi", "diff", "trace", "b1000", "b 1000")):
        return "dwi"
    if group.get("b_value", 0) >= 800:
        return "dwi"
    if (group.get("inversion_time", 0) >= 1000
            and group.get("echo_time", 0) >= 80
            and group.get("repetition_time", 0) >= 3000):
        return "flair"
    return None


def suggested_series(series, modality):
    matching = []
    for key, group in series.items():
        if inferred_modality(group) == modality:
            matching.append((len(group["files"]), key))
    return max(matching)[1] if matching else None


def convert_series(group, output_path):
    output_path = Path(output_path)
    folder = output_path.parent / (output_path.stem + "_dicom")
    if any(not isinstance(source, Path) for _, source in group["files"]):
        folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, (_, source) in enumerate(group["files"]):
        if isinstance(source, Path):
            path = source
        else:
            path = folder / f"slice_{index:06d}.dcm"
            path.write_bytes(source)
        paths.append(path)
    if len(paths) == 1:
        image = sitk.ReadImage(str(paths[0]))
    else:
        ordered = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
            str(paths[0].parent), group["series_uid"])
        if len(ordered) != len(paths):
            raise ValueError("DICOM-серия неполная или содержит несовместимые срезы.")
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(ordered)
        image = reader.Execute()
    if image.GetDimension() != 3 or image.GetSize()[2] < 2:
        raise ValueError("Нужна полная 3D DICOM-серия из нескольких срезов.")
    sitk.WriteImage(image, str(output_path), useCompression=True)
    return str(output_path)
