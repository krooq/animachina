import gym
from rl import Agent
from typing import Tuple
import cv2
import numpy as np
import torch

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

def show(tensor: torch.Tensor, reshape: Tuple[int,int] = None, resize: Tuple[int,int] = None, label: str = ''):
    img = tensor.numpy()
    if reshape:
        img = img.reshape(reshape)
    if resize:
        img = cv2.resize(img, resize, interpolation=cv2.INTER_NEAREST)
    cv2.imshow(label, img)
