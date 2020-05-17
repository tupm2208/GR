import cv2
import numpy as np


def normalize_face_image(bgr_images=None, size=112):
    bgr_images = np.array(bgr_images).astype(np.float32) / 255.

    if len(bgr_images.shape) == 3:
        bgr_images = np.expand_dims(bgr_images, axis=0)

    return np.array([cv2.resize(img, (size, size)) for img in bgr_images])
