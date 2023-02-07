import numpy as np


def HR(y_real, y_pre):
    y_real = y_real.reshape(-1)
    y_pre = y_pre.reshape(-1)
    err_temp = np.abs(y_real - y_pre)
    err = np.where(err_temp <= 0.1, 1, 0)
    HR = np.mean(err)*100
    return HR


def mse(preds, labels):
    preds = preds.reshape(-1)
    labels = labels.reshape(-1)
    mse = np.mean(np.square(preds-labels))
    return mse