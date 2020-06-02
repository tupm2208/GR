import cv2
import numpy as np
from src.config.initializer import Initializer
from src.utils.standards import draw_image
from time import time
from src.core.tracker import Tracker
from dlib import correlation_tracker, rectangle

engine = Initializer()

interval = 10
count = 0
tracker = None
video = cv2.VideoCapture('/home/tupm/projects/GR/assets/test_videos/56354369654362717692.mp4')
embedding = np.empty(512, dtype='float32')

def update_tracker(tracker, image, faces):
    if len(faces) != 0:
        position = faces[0][:4]
        x1, y1, x2, y2 = position
        bbox = (x1, y1, x2-x1, y2-y1)
        tracker.init(image, bbox)
        return True, bbox
    else:
        return tracker.update(image)

def draw_tracker(bbox, image):
    bbox = np.array(bbox, dtype=int)
    x1, y1, w, h = bbox
    
    cv2.rectangle(image, (x1, y1), (x1+w, y1+h), (255, 0, 255), 2)



while True:
    t1 = time()
    ret, image = video.read()
    
    if not ret:
        break

    faces = engine.face_detector([image])[0]
    if len(faces) != 0:
            draw_image(faces[0] + 10, image)
    if count % interval != 0:
        # print('detec')
        faces = []
        
    
    if tracker is None:
        if len(faces) != 0:
            position = faces[0][:4]
            x1, y1, x2, y2 = position
            tracker = cv2.TrackerMIL_create()
            bbox = (x1, y1, x2-x1, y2-y1)
            tracker.init(image, bbox)
            print('init tracker ..................... ok')
        else:
            continue
    else:
        t1 = time()
        ok, bbox = update_tracker(tracker, image, faces)
        draw_tracker(bbox, image)
        pass
        
        

    fps = 1/(time()-t1)
    cv2.putText(image, f'FPS: {fps}', (30, 30), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 255, 0))

    cv2.imshow('test', image)
    k = cv2.waitKey(1)

    if ord('q') == k:
        break
    count += 1
    
    
