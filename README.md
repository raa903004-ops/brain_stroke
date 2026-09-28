# Brain stroke MRI segmentation

Local Streamlit interface for exploratory segmentation of ischemic stroke lesions in MRI. The app supports NIfTI volumes and DICOM studies (a local folder or ZIP). It groups DICOM files by study and series, then lets you select DWI, ADC, and FLAIR series before inference.

## Run locally

Use Python 3.10+ and install PyTorch appropriate for your CPU/GPU first. Then install the remaining dependencies:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local address printed by Streamlit, usually `http://127.0.0.1:8501/`. The two current model checkpoints are in `weights/`: `best_unet3d.pth` (ISLES, DWI + ADC + FLAIR) and `soop_dwi_adc_best.pth` (SOOP, DWI + ADC). The app selects a compatible checkpoint based on the available modalities.

For DICOM, select one study and check the proposed series before running inference. Series identification uses DICOM metadata; filenames such as `IMG-0001` are not reliable modality labels. Patient test scans are not used for training.

This project is experimental and its masks require expert review. The displayed validation Dice/IoU are dataset-level segmentation metrics, not a diagnostic probability for an individual scan.

Training and dataset preparation scripts are included, but the source datasets and local training logs are not part of this repository.
