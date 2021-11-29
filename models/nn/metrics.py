import numpy as np
from scipy import spatial


def fast_hist(pred, label, n=2):
    k = (label >= 0) & (label < n)
    return np.bincount(n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)


def per_class_iou(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-5)


def per_class_dice(hist):
    return 2*np.diag(hist) / (hist.sum(1) + hist.sum(0) + 1e-5)


def iou(pred, label, n_classes=2):
    hist = fast_hist(pred.reshape(1,-1), label.reshape(1, -1), n_classes)
    return np.mean(per_class_iou(hist)[1:])


def dice(pred, label, n_classes=2):
    hist = fast_hist(pred.reshape(1,-1), label.reshape(1, -1), n_classes)
    return np.mean(per_class_dice(hist)[1:])


def accuracy(pred, label, n_classes=2):
    return np.mean(pred == label)

