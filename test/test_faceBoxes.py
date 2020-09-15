import numpy as np
from src.utils.standards import draw_image
from src.utils.image_processing import get_image_face_from_bboxes
from src.config.initializer import Initializer
from glob import glob
import os
import cv2

detector = Initializer().face_detector

image_path = '/home/tupm/SSD/datasets/celebrity/059894.jpg'

image = cv2.imread(image_path)

faces = detector([image])[0]
image_faces = get_image_face_from_bboxes(image, faces)
x1, y1, x2, y2 = faces[0][:4].astype(int)
print(x1, y1, x2, y2)
print(image.shape)
original_face = image[y1:y2, x1:x2, :]

# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.imshow('image', image)
# cv2.imshow('face', image_faces[0])
# cv2.waitKey(0)

cv2.imwrite('/home/tupm/HDD/projects/3dface/facial_verification_system/assets/test_image_result/face.jpg', image_faces[0])
cv2.imwrite('/home/tupm/HDD/projects/3dface/facial_verification_system/assets/test_image_result/original_face.jpg', original_face)
draw_image(faces[0], image)
cv2.imwrite('/home/tupm/HDD/projects/3dface/facial_verification_system/assets/test_image_result/detect.jpg', image)