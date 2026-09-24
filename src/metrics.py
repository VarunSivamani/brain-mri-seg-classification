"""Segmentation metrics used in this repo (ResUNet / Attention-UNet / UNet).

Keras versions mirror the notebook training code (focal Tversky loss,
Tversky score, Dice, IoU). NumPy versions are for offline evaluation.
"""

import numpy as np

EPS = 1e-7
SMOOTH = 1.0


def tversky_coef_numpy(y_true, y_pred, alpha=0.7):
    y_true = np.asarray(y_true).astype(float).ravel()
    y_pred = np.asarray(y_pred).astype(float).ravel()
    tp = np.sum(y_true * y_pred)
    fn = np.sum(y_true * (1 - y_pred))
    fp = np.sum((1 - y_true) * y_pred)
    return (tp + SMOOTH) / (tp + alpha * fn + (1 - alpha) * fp + SMOOTH)


def focal_tversky_numpy(y_true, y_pred, alpha=0.7, gamma=0.75):
    ti = tversky_coef_numpy(y_true, y_pred, alpha=alpha)
    return (1 - ti) ** gamma


def dice_coef_numpy(y_true, y_pred):
    y_true = np.asarray(y_true).astype(float).ravel()
    y_pred = (np.asarray(y_pred).ravel() > 0.5).astype(float)
    inter = np.sum(y_true * y_pred)
    return (2.0 * inter + SMOOTH) / (np.sum(y_true) + np.sum(y_pred) + SMOOTH)


def iou_coef_numpy(y_true, y_pred):
    y_true = (np.asarray(y_true).ravel() > 0.5).astype(float)
    y_pred = (np.asarray(y_pred).ravel() > 0.5).astype(float)
    inter = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - inter
    return (inter + SMOOTH) / (union + SMOOTH)


def keras_metrics_source():
    """Return the exact Keras source used in training (for reference)."""
    return '''
import tensorflow.keras.backend as K
import tensorflow as tf

smooth = 1.0
def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

def focal_tversky(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1 - pt_1), gamma)

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    inter = K.sum(y_true_f * y_pred_f)
    return (2.0 * inter + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    inter = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - inter
    return (inter + smooth) / (union + smooth)
'''
