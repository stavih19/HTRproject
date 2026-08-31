from pathlib import Path
import cv2
import numpy as np


def read_image(path: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def ensure_odd(value: int, minimum: int = 3) -> int:
    value = max(minimum, int(value))
    if value % 2 == 0:
        value += 1
    return value
