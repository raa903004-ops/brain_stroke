"""Rigidly register each SOOP FLAIR volume to its DWI TRACE grid."""

import argparse
import json
import os
from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np
import SimpleITK as sitk


ROOT = Path(r"D:\SOOP")
OUTPUT = Path(r"D:\SOOP-registered-flair")


def register(subject):
    dwi_path = ROOT / subject / "dwi" / f"{subject}_rec-TRACE_dwi.nii.gz"
    flair_path = ROOT / subject / "anat" / f"{subject}_FLAIR.nii.gz"
    output_path = OUTPUT / f"{subject}_flair_on_trace.nii.gz"
    if output_path.is_file() and output_path.stat().st_size > 1000:
        return subject, "cached"
    if not flair_path.is_file():
        return subject, "missing"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=OUTPUT) as temporary:
        dwi_nii = nib.load(str(dwi_path))
        dwi_3d = Path(temporary) / "dwi.nii.gz"
        nib.save(nib.Nifti1Image(np.asarray(dwi_nii.dataobj).squeeze(), dwi_nii.affine), str(dwi_3d))
        fixed = sitk.ReadImage(str(dwi_3d), sitk.sitkFloat32)
        moving = sitk.ReadImage(str(flair_path), sitk.sitkFloat32)
        if fixed.GetDimension() != 3 or moving.GetDimension() != 3:
            raise ValueError(f"{subject}: expected 3D images")
        initial = sitk.Euler3DTransform()
        initial.SetCenter(fixed.TransformContinuousIndexToPhysicalPoint(
            [(size - 1) / 2 for size in fixed.GetSize()]))
        # Keep registration inexpensive even when the source FLAIR is a large 3D scan.
        fixed_small = sitk.Shrink(fixed, [max(1, round(2.5 / s)) for s in fixed.GetSpacing()])
        moving_small = sitk.Shrink(moving, [max(1, round(2.5 / s)) for s in moving.GetSpacing()])
        registration = sitk.ImageRegistrationMethod()
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(0.04)
        registration.SetInterpolator(sitk.sitkLinear)
        registration.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0, minStep=0.05, numberOfIterations=60,
            relaxationFactor=0.5, gradientMagnitudeTolerance=1e-5)
        registration.SetOptimizerScalesFromPhysicalShift()
        registration.SetShrinkFactorsPerLevel([2, 1])
        registration.SetSmoothingSigmasPerLevel([1, 0])
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        registration.SetInitialTransform(initial, inPlace=False)
        transform = registration.Execute(fixed_small, moving_small)
        if isinstance(transform, sitk.CompositeTransform):
            transform = transform.GetNthTransform(transform.GetNumberOfTransforms() - 1)
        translation_mm = np.linalg.norm(transform.GetTranslation())
        rotation_rad = np.linalg.norm(transform.GetParameters()[:3])
        if translation_mm > 30 or rotation_rad > 0.5:
            raise ValueError(f"{subject}: implausible registration shift {translation_mm:.1f} mm, "
                             f"rotation {rotation_rad:.2f} rad")
        aligned = sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        if sitk.GetArrayViewFromImage(aligned).std() < 1e-3:
            raise ValueError(f"{subject}: empty registered FLAIR")
        temporary_output = Path(temporary) / "aligned.nii.gz"
        sitk.WriteImage(aligned, str(temporary_output), useCompression=True)
        os.replace(temporary_output, output_path)
        return subject, f"registered shift={translation_mm:.1f}mm rotation={rotation_rad:.2f}rad"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (ROOT / "subjects.jsonl").read_text(encoding="utf-8").splitlines()]
    subjects = [row["subject_id"] for row in rows if row.get("evaluable_lesionAcute") and row.get("adc_usable")]
    if args.subject:
        subjects = [args.subject]
    if args.limit:
        subjects = subjects[:args.limit]
    failed = []
    for index, subject in enumerate(subjects, 1):
        try:
            name, status = register(subject)
            if index <= 5 or index % 25 == 0 or index == len(subjects):
                print(f"FLAIR registration {index}/{len(subjects)}: {name} {status}", flush=True)
        except Exception as exc:
            failed.append((subject, str(exc)))
            print(f"FLAIR registration ERROR {index}/{len(subjects)}: {exc}", flush=True)
    (OUTPUT / "registration_failures.json").write_text(
        json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Registration finished: {len(subjects) - len(failed)}/{len(subjects)} successful", flush=True)


if __name__ == "__main__":
    main()
