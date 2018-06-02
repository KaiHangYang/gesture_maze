import numpy as np


def make_gaussian(img_size, center_x, center_y, variance):
    arr_x = np.arange(0, img_size, 1)
    arr_y = arr_x[:, np.newaxis]

    return np.exp(-((arr_x - center_x) ** 2 + (arr_y - center_y) ** 2) / 2.0 / variance / variance)
