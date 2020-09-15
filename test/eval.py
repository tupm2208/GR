import numpy as np
import os
import cv2
from tqdm import tqdm
import random
import shutil
from src.config.initializer import Initializer
from src.utils.standards import draw_image

engine = Initializer()
face_detector = engine.face_detector

IMAGES_DIR = '/home/tupm/SSD/datasets/FDDB/originalPics/'
ANNOTATIONS_PATH = '/home/tupm/SSD/datasets/FDDB/FDDB-folds/'
RESULT_DIR = 'result/'
os.makedirs(RESULT_DIR, exist_ok=True)
MODEL_PATH = 'models/model.pb'

annotations = [s for s in os.listdir(ANNOTATIONS_PATH) if s.endswith('ellipseList.txt')]
image_lists = [s for s in os.listdir(ANNOTATIONS_PATH) if not s.endswith('ellipseList.txt')]
annotations = sorted(annotations)
image_lists = sorted(image_lists)

images_to_use = []
for n in image_lists:
    with open(os.path.join(ANNOTATIONS_PATH, n)) as f:
        images_to_use.extend(f.readlines())

images_to_use = [s.strip() for s in images_to_use]
with open(os.path.join(RESULT_DIR, 'faceList.txt'), 'w') as f:
    for p in images_to_use:
        f.write(p + '\n')

ellipses = []
for n in annotations:
    with open(os.path.join(ANNOTATIONS_PATH, n)) as f:
        ellipses.extend(f.readlines())

i = 0
with open(os.path.join(RESULT_DIR, 'ellipseList.txt'), 'w') as f:
    for p in ellipses:
        
        # check image order
        if 'big/img' in p:
            assert images_to_use[i] in p
            i += 1

        f.write(p)

import cv2
predictions = []
for n in tqdm(images_to_use):
    img_path = os.path.join(IMAGES_DIR, n) + '.jpg'
    image_array = cv2.imread(img_path)
    # image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    # threshold is important to set low
    
    boxes = face_detector([image_array], score_threshold=0.05)[0]
    # for box in boxes:
    #     print(box[-1])
    #     draw_image(box, image_array)
    # cv2.imshow('', image_array)
    # cv2.waitKey(0)
    scores = boxes[:, -1]
    boxes = boxes[:, :4]
    # print(boxes.shape)
    predictions.append((n, boxes, scores))
    
    

with open(os.path.join(RESULT_DIR, 'detections.txt'), 'w') as f:
    for n, boxes, scores in predictions:
        f.write(n + '\n')
        f.write(str(len(boxes)) + '\n')
        for b, s in zip(boxes, scores):
            # ymin, xmin, ymax, xmax = b
            xmin, ymin, xmax, ymax = b
            h, w = int(ymax - ymin), int(xmax - xmin)
            f.write('{0} {1} {2} {3} {4:.4f}\n'.format(int(xmin), int(ymin), w, h, s))

# for n in tqdm(images_to_use):
#     p = os.path.join(RESULT_DIR, 'images', n + '.jpg')
#     os.makedirs(os.path.dirname(p), exist_ok=True)
#     shutil.copy(os.path.join(IMAGES_DIR, n) + '.jpg', p)