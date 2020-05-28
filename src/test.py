import cv2
import numpy as np
from src.config.initializer import Initializer
from src.utils.standards import draw_image
from time import time
from src.core.tracker import Tracker

engine = Initializer()

interval = 2
count = 0
tracker = None
video = cv2.VideoCapture(0)
embedding = np.empty(512, dtype='float32')


while True:
    t1 = time()
    ret, image = video.read()

    if not ret:
        break
    
    faces = engine.face_detector([image])[0]
    draw_image(faces[0], image)
    if count % interval != 0:
        faces = []
        
    
    if tracker is None:
        if len(faces) != 0:
            tracker = Tracker(image, faces[0], embedding, '')
        else:
            continue
    else:
        if len(faces) != 0:
            
            tracker.custom_update(image, faces[0])
        else:
            tracker.custom_update(image)
        
        rec = tracker.current_location

        draw_image(rec, image, 't')

    fps = 1/(time()-t1)
    cv2.putText(image, f'FPS: {fps}', (30, 30), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 255, 0))

    cv2.imshow('test', image)
    k = cv2.waitKey(1)

    if ord('q') == k:
        break
    count += 1
    
    
