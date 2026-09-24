# Data 

> Note - Data files (MRI images) are not stored in this repo (too large), please download via Kaggle


## 1. Classification 

Source: `masoudnickparvar/brain-tumor-mri-dataset`

- 7,023 MRI images
- 4 Classes: `glioma`, `meningioma`, `pituitary`, `notumor`
- Layout: `Training/` (5,712) + `Testing/` (1,311) = 7,023
- Validation (per report Ch. 6): 1,143 images held out from Training

```bash
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
unzip brain-tumor-mri-dataset.zip -d data/classification
```

Expected:

```text
data/classification/Training/{glioma,meningioma,notumor,pituitary}/
data/classification/Testing/{glioma,meningioma,notumor,pituitary}/
```

## 2. Segmentation

Source: `mateuszbuda/lgg-mri-segmentation`

- 3,929 MRI + mask pairs (LGG)
- 2,556 no-tumor + 1,373 tumor images, each with a `.tif` mask (`kaggle_3m/...`)
- `data.csv` maps `patient_id -> image_path, mask_path`

```bash
kaggle datasets download -d mateuszbuda/lgg-mri-segmentation
unzip lgg-mri-segmentation.zip -d data/segmentation
```

## Notes

- All notebooks assume this layout or a Kaggle `/kaggle/input/...` mount - adjust the first data cell if your path differs.
- Preprocessing used: resize to 256×256 (B3/ResUNet/Attention-UNet) or 128×128 (UNet baseline), rescale to [0,1], grayscale→3ch where needed, train-time flips.
- No patient metadata is included; see `docs/project-report_TEAM-1_SKCET_2024.pdf` Ch. 4.4.2–4.4.3 for the full protocol.
