import cv2
import numpy as np

def scale_aspect(img: np.ndarray, min_size: int, scale: int):
    w           = scale * img.shape[1]
    h           = scale * img.shape[0]
    aspect      = w/h
    if w < h:
        w = min_size
        h = round(w/aspect)
    elif h < w:
        h = min_size
        w = round(h*aspect)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)