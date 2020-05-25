import dlib
import cv2
import numpy as np
import os
import imutils
from src.utils.image_processing import preprocess
from src.utils.standards import draw_landmark
from glob import glob
import tqdm


predictor = dlib.shape_predictor('/home/tupm/HDD/projects/children_face/trained_models/shape_predictor_68_face_landmarks.dat')


def get_x_y(point, div=1):
    x = point.x // div
    y = point.y // div

    return [x, y]


def handle_face(img_path):
    image = cv2.imread(img_path)
    image = imutils.resize(image, width=112)
    h, w, _ = image.shape
    rec = dlib.rectangle(0, 0, w, h)
    shape = predictor(image, rec)
    parts = [36, 39, 42, 45, 34, 49, 55]
    left_eye = get_x_y(shape.part(36) + shape.part(39), 2)
    right_eye = get_x_y(shape.part(42) + shape.part(45), 2)
    nose = get_x_y(shape.part(33))
    left_lift = get_x_y(shape.part(48))
    right_lift = get_x_y(shape.part(54))

    landms = np.array([left_eye, right_eye, nose, left_lift, right_lift], dtype=int)
    landms[2, 1] -= 10
    image = preprocess(image, [0, 0, w, h], landms)
    # for x, y in landms:
    #     cv2.circle(image)

    # draw_landmark(image, landms)
    img_path = img_path.replace('VN-celeb', 'VN-celeb_aligned')
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    # cv2.imshow('', image)
    # cv2.waitKey(0)
    cv2.imwrite(img_path, image)


if __name__ == '__main__':
    image_path_list = glob('/home/tupm/datasets/VN-celeb/*/*')
    for e in tqdm.tqdm(image_path_list):
        handle_face(e)
        # break