import cv2
import numpy as np

from skimage import io
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from context_aware_pseudocoloring import context_aware_pcolor


def basic_labeling(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.uint8)
    return ndi.label(img)[0].astype(np.uint16)


def distance_watershed(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.bool)
    distance = ndi.distance_transform_edt(image)
    coords = peak_local_max(distance, footprint=np.ones((12, 12)), labels=image)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=image)

    return labels


def inference_2D(img: np.ndarray, ort_session, z: int, thresh: float,
                 adjacent_mean: float, post_processing_type: str,
                 sliding_window_conf: list) -> np.ndarray:
    # Pre-processing:
    _x = context_aware_pcolor(img, z, thresh, adjacent_mean)
    _x = np.expand_dims(_x, axis=0)
    _x = _x.astype(np.float32)

    if sliding_window_conf:
        y_semseg = np.zeros((img.shape[1], img.shape[2]))
        h, w = img.shape[1], img.shape[2]
        patch_w = w // sliding_window_conf[0]
        patch_h = h // sliding_window_conf[1]

        for x in range(sliding_window_conf[0]):
            for y in range(sliding_window_conf[1]):
                y_semseg[0 + y*patch_h:patch_h + y*patch_h,
                         0 + x*patch_w:patch_w + x*patch_w] = \
                    ort_session.run(None, {
                        'x': _x[:, 0 + y*patch_h:patch_h + y*patch_h,
                                0 + x*patch_w:patch_w + x*patch_w, :]
                    })[0][0, ..., 0]
    else:
        y_semseg = ort_session.run(None, {'x': _x})[0][0, ..., 0]

    y_semseg = cv2.threshold(y_semseg, 0.5 , 1, cv2.THRESH_BINARY)[1]

    # Post-processing:
    if post_processing_type == None:
        return y_semseg.astype(np.uint16)
    elif post_processing_type == "distance_ws":
        instance_mask = distance_watershed(y_semseg)
        return instance_mask.astype(np.uint16)


def inference_3D(path2img: str, ort_session, z_offset: int, thresh: float,
                 adjacent_mean: float, post_processing_type: str,
                 sliding_window_conf: list) -> np.ndarray:
    img_3D = io.imread(path2img)
    res_mask = np.zeros(img_3D.shape, dtype=np.uint16)

    for z in range(z_offset, img_3D.shape[0] - z_offset):
        res_mask[z] = inference_2D(img_3D, ort_session, z, thresh,
                                   adjacent_mean, post_processing_type,
                                   sliding_window_conf)
    return res_mask