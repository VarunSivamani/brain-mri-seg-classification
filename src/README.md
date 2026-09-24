# `src/` — reusable code (lightweight)

Full training logic lives in `notebooks/` (kept with outputs as the record of
results). This folder holds only code worth importing:

- `metrics.py` — Tversky / focal-Tversky / Dice / IoU in Keras + NumPy,
  matching the segmentation notebooks.

```python
from src.metrics import dice_coef_numpy, iou_coef_numpy, focal_tversky_numpy
```
