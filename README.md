# MRI Brain Tumor Classification and Segmentation Using Deep Learning

> **Final Year Project**
> B.Tech Artificial Intelligence and Data Science
> Sri Krishna College of Engineering and Technology, Coimbatore (Anna University)
> March 2024

**Authors**

- Varun S (20EUAI051)
- Tharun S (20EUAI048)
- Sathish Kumar M (20EUAI034)

**Supervisor**

- Ms. B. Kiruba, Assistant Professor, AI & DS

---

### Project Description

Joint brain-tumor classification (glioma / meningioma / pituitary / no-tumor) and
pixel-level tumor segmentation from MRI, comparing transfer-learned classifiers
(EfficientNet-B3, EfficientNetV2, ViT, CNN) against UNet-family segmenters
(ResUNet, Attention-UNet, UNet). All notebooks are archived with outputs as the
record of results.

---

## Contents

- [1. Overview](#1-overview)
- [2. Results](#2-results-report-ch-53-tables-51-52)
- [3. Repository Structure](#3-repository-structure)
- [4. Installation](#4-installation)
- [5. Usage](#5-usage)
- [6. Method](#6-method)
- [7. Testing](#7-testing)
- [8. Limitations & Future Work](#8-limitations--future-work)
- [9. References](#9-references)
- [10. License & Disclaimer](#10-license--disclaimer)

## 1. Overview

Brain-tumor MRI analysis is hard because of modality and illumination variation,
irregular tumor shapes, small datasets, and overfitting in plain CNN/UNet
baselines. This project tests whether ImageNet transfer learning plus
residual/attention mechanisms close that gap.

**What is demonstrated:**

- Kaggle MRI fetching + resizing/normalization + train/val/test splits (no binaries committed)
- Four classifiers (EfficientNet-B3 / EfficientNetV2-L / ViT-L-32 / CNN, 30 epochs) + three segmenters (ResUNet / Attention-UNet / UNet)
- Six runnable notebooks with loss/accuracy curves, confusion matrices, and sample predictions
- Report, conference paper, per-epoch CSV logs, and shared metrics module archived verbatim

**Datasets:**

| Task | Source | Size | Labels |
|---|---|---|---|
| Classification | [`masoudnickparvar/brain-tumor-mri-dataset`](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) | **7,023** images: `Training/` 5,712 + `Testing/` 1,311 | glioma, meningioma, pituitary, no-tumor |
| Segmentation | [`mateuszbuda/lgg-mri-segmentation`](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) (TCGA-LGG) | **3,929** pairs: 2,556 no-tumor + 1,373 tumor | pixel mask (`.tif`), `data.csv` maps image/mask |

> **Data setup:** images are not committed. Run the two Kaggle download commands in
> `data/README.md` to recreate the `data/` folder layout.
>
> **Validation splits:** for classification, 1,143 images from `Training/` are held
> out as validation (report Ch. 6) and the separate `Testing/` folder (1,311 images)
> is used only for final scores. For segmentation, 15% of images are held out as
> test, then 10% of the remainder as validation, both handled inside the data
> generators.

## 2. Results (report Ch. 5.3, Tables 5.1–5.2)

**Setup:**

| Setting | Classification | Segmentation |
|---|---|---|
| Input | MRI, 256×256 (128×128 UNet baseline) | MRI + mask, same sizes |
| Scaling | rescale to [0,1], grayscale→3ch where needed | same |
| Split | train + 1,143 val + 1,311 test | 85/15 + 10% val |
| Augment | horizontal/vertical flips (train) | same |
| Optimizer / Loss | Adamax/AdamW/Adam + cross-entropy | Adam (lr 1e-3) + focal-Tversky |
| Epochs | 30 | 60 (Attention-UNet 35+) |
| Best | **EfficientNet-B3** | **ResUNet** |

<br/>
<br/>

| Model | Architecture | Loss | Acc. | Prec. | Rec. | F1 |
|---|---|---|---|---|---|---|
| EfficientNet-B3 | B3-ImageNet, max-pool + BN + Dense-256 + Dropout 0.45, Adamax 1e-3 | 0.0713 | **0.9993** | 0.99 | 0.99 | 0.99 |
| EfficientNetV2 | V2-L frozen features, head 1280→256→4, GELU, AdamW, label-smooth 0.1 | 0.4683 | 0.9527 | 0.95 | 0.95 | 0.95 |
| ViT | ViT-L-32 frozen, head 1000→256→4, same optimizer/loss | 0.4803 | 0.9509 | 0.94 | 0.93 | 0.93 |
| CNN | Custom TF-CNN baseline, Adam, 30 epochs | 0.1625 | 0.9568 | 0.94 | 0.94 | 0.94 |

<br/>
<br/>

| Model | Focal-Tversky ↓ | Tversky ↑ | IoU ↑ | Dice ↑ |
|---|---|---|---|---|
| ResUNet | **0.2007** | **0.8819** | **0.8243** | 0.8589 |
| Attention-UNet | 0.2879 | 0.8056 | 0.6577 | 0.7901 |
| UNet | 0.2189 | 0.8217 | 0.7907 | **0.8590** |

#### Loss curves

- Figs. 5.1–5.12 in `docs/project-report-team1.pdf` (loss/accuracy, focal-Tversky, Tversky, IoU, Dice).
- Per-epoch CSVs in `results/training-logs/` (cnn, efficientnetv2, vit); B3 curves live inside notebook 01.
- Raw notebook cells match this ranking with small split/rounding deltas (e.g. B3 cell: 1.00 on its 656-image test generator; EffV2 0.9497 / ViT 0.9329; CNN 0.9383; ResUNet eval 0.2472 / Tversky 0.8443). Tables above are canonical.
- Accepted for presentation at NCIECIT-2024 — see `docs/conference-paper-NCIECIT.pdf`.

## 3. Repository Structure

```text
├── notebooks/
│   ├── classification/
│   │   ├── 01_classification_efficientnet_b3.ipynb        # BEST — 99.93%
│   │   ├── 02_classification_efficientnetv2_vit_30epochs.ipynb  # V2 + ViT head-to-head
│   │   └── 03_classification_cnn_baseline_30epochs.ipynb # baseline
│   └── segmentation/
│       ├── 01_segmentation_resunet.ipynb                 # BEST — loss 0.2007
│       ├── 02_segmentation_attention_unet.ipynb          # attention gates
│       └── 03_segmentation_unet_baseline.ipynb           # lightweight baseline
├── src/
│   ├── metrics.py      # Tversky / focal-Tversky / Dice / IoU (Keras + NumPy)
│   └── README.md
├── results/training-logs/  # per-epoch CSVs + README
├── data/README.md          # Kaggle download + layout (no binaries committed)
├── docs/
│   ├── project-report-team1.pdf       # full B.Tech report (TEAM 1.pdf)
│   └── conference-paper-NCIECIT.pdf   # accepted paper (PAPER.pdf)
├── requirements.txt
├── CITATION.cff
├── LICENSE (MIT)
└── .gitignore
```

> **Design rule enforced:**
>
> - Notebooks are stored *with outputs* as the result record; reruns are optional.
> - No MRI binaries or `.h5`/`.hdf5` weights are committed (see `.gitignore`).
> - Shared metric code lives *only* in `src/metrics.py`.
> - Superseded `CNN_94_accuracy(20_epochs).ipynb` is excluded (weaker duplicate of the 30-epoch CNN).

## 4. Installation

```powershell
cd repo
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Reproducibility:** 
- report baseline was Intel i5 / 4 GB RAM (Jupyter + VS Code);
actual runs used Kaggle GPU runtimes. 
- Splits, seeds, and hyperparameters are fixed
in the notebooks; per-epoch logs in `results/training-logs/` let you verify curves without a GPU rerun.

## 5. Usage

```powershell
jupyter notebook notebooks/classification/01_classification_efficientnet_b3.ipynb
jupyter notebook notebooks/segmentation/01_segmentation_resunet.ipynb
```

1. Start with **notebook 01 (B3)**: preprocessing → 30-epoch training → loss/accuracy curves → confusion matrix → sample predictions.
2. Then **notebook 02 (V2+ViT)**: same data, head-to-head transfer comparison.
3. Then **notebook 03 (CNN)**: baseline reference.
4. Switch to **segmentation 01 (ResUNet)**: masks → focal-Tversky training → Tversky/IoU/Dice → overlay predictions; 02/03 for attention/UNet ablations.

> If your data path differs from Kaggle `/kaggle/input/...`, edit the first data cell
> per `data/README.md`.

## 6. Method

- **Collection:** Kaggle downloads (links in §1); classification via `flow_from_directory` / `flow_from_dataframe`, segmentation via `DataGenerator` over `data.csv`.
- **Preprocessing:** NumPy/pandas arrays → resize (256; 128 for UNet baseline) → rescale [0,1] → grayscale→3ch where needed → 70/30-style splits → `[N, H, W, C]` batches.
- **Training:** Appendix-1 (`docs/project-report-team1.pdf`) and `notebooks/*/*.ipynb`; B3 with Adamax 1e-3 + `categorical_crossentropy`, V2/ViT with AdamW + label-smoothed cross-entropy, UNet family with focal-Tversky + EarlyStopping/ReduceLROnPlateau/Checkpoint.
- **Inference:** argmax class (classification) / 0.5-threshold sigmoid mask (segmentation) + overlay plots.
- **Evaluation:** train/val curves, confusion matrix, precision/recall/F1 (classification); focal-Tversky, Tversky, IoU, Dice (segmentation).

## 7. Testing

Report Ch. 6:

- White-box, conditional, data-flow, loop testing.
- Manual cases: 15-image classification spot-check (Test Case 1: glioma→glioma) and mask overlay check (Test Case 2: segmented tumor region) render correctly.
- No automated suite is shipped.

## 8. Limitations & Future Work

### Limitations

| # | Limitation | Details |
|---|------------|---------|
| 1 | Single-source public sets | No BraTS multi-modal T1/T1ce/T2/FLAIR fusion; no external clinical validation |
| 2 | Overfitting sensitivity | Deep models can overfit small, noisy MRI cohorts without regularization and early stopping |
| 3 | Basic augmentation | Only flips + rescaling; no elastic/GAN augmentation or class-balancing beyond sampling |
| 4 | No calibration | No uncertainty/confidence reporting on predictions or masks |

### Future Work

- **Richer inputs:** multi-modal fusion + radiomic/genomic features
- **Better augmentation:** GAN-based synthesis and stronger geometric/photometric transforms
- **Architecture search:** self-/non-local attention and automated depth/width search
- **Hybrid models:** residual + attention hybrids with calibrated uncertainty
- **Scale-up:** benchmarking on larger, diverse cohorts for generalizability

## 9. References

1. Akter et al., Robust CNN + U-Net for MRI classification/segmentation, *Expert Syst. Appl.* 238 (2023) 122347
2. Zhang et al., Deep fusion of multi-modal features, *Heliyon* 9 (2023) e19266
3. Çetiner & Metlek, DenseUNet+, *J. King Saud Univ. CIS* 35 (2023) 101663
4. Ranjbarzadeh et al., Optimized CNN + chimp optimization, *Comput. Biol. Med.* 168 (2023) 107723

Full [1]–[9] list: report References / `docs/project-report-team1.pdf`.

## 10. License & Disclaimer

MIT — see `LICENSE`. Academic final-year project (2024); predictions are experimental and
**not clinical advice**. MRI interpretation requires qualified radiologists; model outputs
must not guide care without clinical validation.

```bibtex
@techreport{varun2024mribrain,
  title       = {A Comprehensive Approach to MRI Brain Image Segmentation and Classification Using Deep Learning Approaches},
  author      = {Varun, S. and Tharun, S. and Sathish Kumar, M. and Kiruba, B.},
  institution = {Sri Krishna College of Engineering and Technology},
  year        = {2024}
}
```
