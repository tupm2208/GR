import os
import cv2
import numpy as np
from src.config.initializer import Initializer

engine = Initializer()
face_detector = engine.face_detector



def read_txt(txt_file):
    with open(txt_file) as _f:
        txt_lines = _f.readlines()
    txt_lines.sort()
    metas = []
    for line in txt_lines:
        line = line.rstrip()
        boxes = []

        _img_path = line.rsplit('| ', 1)[0]
        annos = line.rsplit('| ', 1)[-1]
        labels = annos.split(' ')
        image = cv2.imread(_img_path)
        for label in labels:
            bbox = np.array(label.split(','), dtype=np.float)
            x1, y1, x2, y2 = bbox[:4].astype('int')
            if x2 - x1 < 40:
                continue
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            boxes.append(bbox)
        predicted_boxes = face_detector([image])[0]
        print(predicted_boxes)
        for box in predicted_boxes:
            x1, y1, x2, y2 = box[:4]
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.imshow('', image)
        cv2.waitKey(0)

        current_img_path = _img_path
        metas.append([current_img_path, boxes])

        # print(boxes)
        # break
        ###some change can be made here
    return metas

read_txt('/home/tupm/HDD/projects/3dface/faceboxes_xxx_adam/val.txt')

