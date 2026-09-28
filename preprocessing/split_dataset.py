"""
Разбиение датасета МРТ-снимков ишемического инсульта (BIDS-подобная структура)
на обучающую / валидационную / тестовую выборки НА УРОВНЕ ПАЦИЕНТА.

Ожидаемая структура (как в structure.txt):

dataset_root/
├── sub-strokecase0001/
│   └── ses-0001/
│       ├── anat/
│       │   ├── sub-strokecase0001_ses-0001_FLAIR.nii.gz
│       │   └── sub-strokecase0001_ses-0001_FLAIR.json
│       └── dwi/
│           ├── sub-strokecase0001_ses-0001_adc.nii.gz
│           ├── sub-strokecase0001_ses-0001_adc.json
│           ├── sub-strokecase0001_ses-0001_dwi.nii.gz
│           └── sub-strokecase0001_ses-0001_dwi.json
├── ...
└── derivatives/
    └── sub-strokecase0001/
        └── ses-0001/
            ├── sub-strokecase0001_ses-0001_msk.nii.gz   # маска очага (ground truth)
            └── sub-strokecase0001_ses-0001_snp.png      # превью

Почему делим по пациенту, а не по файлу:
    Если снимки одного и того же пациента (FLAIR/ADC/DWI/маска) окажутся
    в разных выборках, модель на этапе валидации/теста будет неявно видеть
    анатомию, которую уже "запомнила" на обучении -> завышенная, нечестная
    точность. Поэтому уникальный ключ разбиения — sub-strokecase####,
    а не отдельные .nii.gz файлы.

Требуется: pandas, scikit-learn
    pip install pandas scikit-learn --break-system-packages
"""

import argparse
import csv
import shutil
import gzip
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import nibabel as nib  # опционально, только для стратификации по наличию очага
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False


# --------------------------------------------------------------------------
# 1. Сбор манифеста: для каждого пациента находим все нужные файлы
# --------------------------------------------------------------------------

def find_one(folder: Path, pattern: str) -> str | None:
    """Возвращает путь к первому файлу, совпавшему с паттерном, либо None."""
    matches = sorted(folder.glob(pattern))
    return str(matches[0]) if matches else None


def build_manifest(dataset_root: Path) -> pd.DataFrame:
    rows = []
    subject_dirs = sorted(dataset_root.glob("sub-strokecase*"))

    for sub_dir in subject_dirs:
        if not sub_dir.is_dir():
            continue
        subject_id = sub_dir.name  # напр. sub-strokecase0001

        ses_dirs = sorted(sub_dir.glob("ses-*"))
        if not ses_dirs:
            continue
        # берём первую (и обычно единственную) сессию
        ses_dir = ses_dirs[0]
        session_id = ses_dir.name

        anat_dir = ses_dir / "anat"
        dwi_dir = ses_dir / "dwi"

        flair_path = find_one(anat_dir, "*_FLAIR.nii.gz") if anat_dir.exists() else None
        adc_path = find_one(dwi_dir, "*_adc.nii.gz") if dwi_dir.exists() else None
        dwi_path = find_one(dwi_dir, "*_dwi.nii.gz") if dwi_dir.exists() else None

        # маска лежит в параллельном дереве derivatives/
        mask_dir = dataset_root / "derivatives" / subject_id / session_id
        mask_path = find_one(mask_dir, "*_msk.nii.gz") if mask_dir.exists() else None
        snapshot_path = find_one(mask_dir, "*_snp.png") if mask_dir.exists() else None

        rows.append({
            "subject_id": subject_id,
            "session_id": session_id,
            "flair_path": flair_path,
            "adc_path": adc_path,
            "dwi_path": dwi_path,
            "mask_path": mask_path,
            "snapshot_path": snapshot_path,
        })

    df = pd.DataFrame(rows)
    return df


def report_completeness(df: pd.DataFrame) -> None:
    """Печатает статистику по недостающим модальностям, чтобы вы могли
    осознанно решить: исключать такие кейсы или использовать частично."""
    total = len(df)
    print(f"Всего найдено пациентов: {total}")
    for col in ["flair_path", "adc_path", "dwi_path", "mask_path"]:
        missing = df[col].isna().sum()
        print(f"  нет {col:15s}: {missing} ({missing / total:.1%})")

    core_ok = df[["flair_path","adc_path","dwi_path","mask_path"]].notna().all(axis=1)
    print(f"Пациентов с полным набором данных для обучения: {core_ok.sum()}")


# --------------------------------------------------------------------------
# 2. Опциональная стратификация: есть ли непустой очаг в маске
#    (полезно, чтобы train/val/test имели похожее соотношение
#     "с поражением / без поражения" или похожий размер очага)
# --------------------------------------------------------------------------

def has_lesion(mask_path: str | None) -> int:
    if not mask_path or not HAS_NIBABEL:
        return 0
    try:
        img = nib.load(mask_path)
        data = img.get_fdata()
        return int((data > 0).any())
    except Exception:
        return 0


# --------------------------------------------------------------------------
# 3. Разбиение по пациентам
# --------------------------------------------------------------------------

def split_patients(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    stratify_by_lesion: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
        "Доли train/val/test должны в сумме давать 1.0"

    df = df.copy()

    if stratify_by_lesion:
        df["has_lesion"] = df["mask_path"].apply(has_lesion)
        strat_col = df["has_lesion"]
    else:
        strat_col = None

    # шаг 1: отделяем test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_col,
    )

    # шаг 2: из оставшегося отделяем val (пропорция пересчитывается)
    val_relative_size = val_size / (train_size + val_size)
    strat_col_2 = train_val_df["has_lesion"] if stratify_by_lesion else None
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_size,
        random_state=random_state,
        stratify=strat_col_2,
    )

    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")

    result = pd.concat([train_df, val_df, test_df]).sort_values("subject_id")
    return result.reset_index(drop=True)


# --------------------------------------------------------------------------
# 4. Материализация: разложить файлы по папкам train/val/test
#    (по умолчанию — symlink, экономит место; можно copy)
# --------------------------------------------------------------------------

def materialize_split(df: pd.DataFrame, output_root: Path, mode: str = "symlink") -> None:
    modality_cols = ["flair_path", "adc_path", "dwi_path", "mask_path"]

    for _, row in df.iterrows():
        split = row["split"]
        subject_id = row["subject_id"]
        session_id = row["session_id"]

        for col in modality_cols:
            src = row[col]
            if not src or pd.isna(src):
                continue
            src_path = Path(src)

            # сохраняем относительную BIDS-подпапку (anat/dwi/derivatives)
            if col == "flair_path":
                rel_sub = Path(subject_id) / session_id / "anat"
            elif col in ("adc_path", "dwi_path"):
                rel_sub = Path(subject_id) / session_id / "dwi"
            else:  # mask_path
                rel_sub = Path("derivatives") / subject_id / session_id

            dest_dir = output_root / split / rel_sub
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / src_path.name

            if dest_path.exists():
                continue
            if mode == "symlink":
                dest_path.symlink_to(src_path.resolve())
            elif mode == "copy":
                shutil.copy2(src_path, dest_path)
            else:
                raise ValueError("mode должен быть 'symlink' или 'copy'")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path, help="Путь к корню датасета (где лежат sub-strokecase*)")
    parser.add_argument("--output_root", type=Path, default=Path("./split_dataset"),
                         help="Куда разложить train/val/test")
    parser.add_argument("--train_size", type=float, default=0.70)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--no_materialize", action="store_true",
                         help="Только сгенерировать csv-манифесты, без раскладки файлов по папкам")
    parser.add_argument("--no_stratify", action="store_true",
                         help="Не стратифицировать по наличию очага (например, если nibabel не установлен)")
    args = parser.parse_args()

    df = build_manifest(args.dataset_root)
    report_completeness(df)

    # для обучения оставляем только пациентов с FLAIR + DWI + ADC+ маской
    usable_df = df[df["flair_path"].notna() &df["adc_path"].notna()&df["dwi_path"].notna()&df["mask_path"].notna()].reset_index(drop=True)
    print(f"\nБудет разбито пациентов: {len(usable_df)} из {len(df)}")

    split_df = split_patients(
        usable_df,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        stratify_by_lesion=not args.no_stratify and HAS_NIBABEL,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    for split_name in ["train", "val", "test"]:
        part = split_df[split_df["split"] == split_name]
        out_csv = args.output_root / f"{split_name}.csv"
        part.to_csv(out_csv, index=False)
        print(f"{split_name}: {len(part)} пациентов -> {out_csv}")

    if not args.no_materialize:
        materialize_split(split_df, args.output_root, mode=args.mode)
        print(f"\nФайлы разложены в {args.output_root}/{{train,val,test}}/... (режим: {args.mode})")


if __name__ == "__main__":
    main()