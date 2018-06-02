import numpy as np
import cv2

def make_gaussian(img_size, center_x, center_y, variance):
    arr_x = np.arange(0, img_size, 1)
    arr_y = arr_x[:, np.newaxis]

    return np.exp(-((arr_x - center_x) ** 2 + (arr_y - center_y) ** 2) / 2.0 / variance / variance)

def pad_image(image, target_size):
    max_size = max(image.shape[0], image.shape[1])
    resize_scale = target_size * 1.0 / max_size
    image = cv2.resize(image, (int(image.shape[1] * resize_scale), int(image.shape[0] * resize_scale)))

    pad_width = int((-image.shape[1] + target_size) / 2.0)
    pad_height = int((-image.shape[0] + target_size) / 2.0)

    image = cv2.copyMakeBorder(image, top=pad_height, bottom=pad_height, left=pad_width, right=pad_width, borderType=cv2.BORDER_CONSTANT, value=[128, 128, 128])

    return cv2.resize(image, (target_size, target_size))
