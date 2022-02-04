import cv2
import numpy as np

from typing import Tuple


def normalize(img: np.ndarray) -> np.ndarray:
    if img.max() > 0:
        img = (img - img.min()) / (img.max() - img.min())

    return img


def apply_clahe_filter(img: np.ndarray, clip_limit: float = 10.0,
                       tile_grid_size: Tuple[int, int] = (16, 16)) -> np.ndarray:
    clahe_filter = cv2.createCLAHE(clipLimit=clip_limit,
                                   tileGridSize=tile_grid_size)
    if img.max() > 0:
        img = clahe_filter.apply(img)

    return img


def context_aware_pcolor(img_3D: np.ndarray, z: int, thresh: float = 0.2,
                         adjacent_mean: float = 0.03,
                         norm_per_channel: bool = True) -> np.ndarray:
    """Apply Context Aware Pseudocoloring as preprocessing step."""
    # TODO: Doc string explaining adaptive thresholding and adjacent_mean
    if z == 0:
        adjacent_slc1 = np.zeros((img_3D.shape[1], img_3D.shape[2]))
    else:
        adjacent_slc1 = img_3D[z - 1]

    if z == img_3D.shape[0] - 1:
        adjacent_slc2 = np.zeros((img_3D.shape[1], img_3D.shape[2]))
    else:
        adjacent_slc2 = img_3D[z + 1]

    p_red = apply_clahe_filter(adjacent_slc1)
    p_blue = apply_clahe_filter(adjacent_slc2)
    p_red = normalize(p_red)
    p_green = normalize(img_3D[z])
    p_blue = normalize(p_blue)

    # Pseudo red and blue channels - smoothing and adaptive thresholding:
    p_red = cv2.blur(p_red, (8, 8))
    p_blue = cv2.blur(p_blue, (8, 8))

    p_red_thresh = np.zeros(p_red.shape)
    p_blue_thresh = np.zeros(p_blue.shape)

    red_thresh = thresh
    blue_thresh = thresh

    # Adaptive thresholding, choose adjacent_mean based on 2 * mean(GT):
    cnt1 = 0
    while (p_red_thresh.mean() > adjacent_mean or p_red_thresh.mean() == 0) and cnt1 < 30:
        p_red_thresh = cv2.threshold(p_red, red_thresh, 1, cv2.THRESH_BINARY)[1]
        red_thresh += 0.01
        cnt1 += 1
    p_red = p_red_thresh

    cnt2 = 0
    while (p_blue_thresh.mean() > adjacent_mean or p_blue_thresh.mean() == 0) and cnt2 < 30:
        p_blue_thresh = cv2.threshold(p_blue, blue_thresh, 1, cv2.THRESH_BINARY)[1]
        blue_thresh += 0.01
        cnt2 += 1
    p_blue = p_blue_thresh

    # Multiply-accumulate:
    p_red = p_red * p_green + p_green
    p_blue = p_blue * p_green + p_green

    if norm_per_channel:
        p_red = normalize(p_red)
        p_blue = normalize(p_blue)
        p_color = np.dstack((p_red, p_green, p_blue))
    else:
        p_color = np.dstack((p_red, p_green, p_blue))
        p_color = normalize(p_color)

    return p_color