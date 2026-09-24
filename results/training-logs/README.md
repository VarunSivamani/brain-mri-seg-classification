# Training logs (loss/accuracy curves as CSV)

Exported from the classification notebooks so tables/plots are reproducible
without re-running GPU training:

- `cnn_30epochs.csv`, `cnn_25epochs.csv` — `loss;acc;val_loss;val_acc` per epoch
- `efficientnetv2_30epochs.csv` — `Sno;loss;acc;val_loss;val_acc` per epoch
- `vit_30epochs.csv` — same schema as above

The EfficientNet-B3 run logs live inside
`notebooks/classification/01_classification_efficientnet_b3.ipynb`
(train/val loss + accuracy curves + confusion matrix).
