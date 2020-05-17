import numpy as np
import cv2
import dlib
from src.config.initializer import Initializer

engine = Initializer()

video = cv2.VideoCapture('/home/tupm/SSD/datasets/video_test/46141157262008171431.mp4')

trackers = []

def draw_image(rec, image):
    x1, y1, x2, y2 = rec

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return image


while True:
    ret, image = video.read()

    if not ret:
        break
    rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if len(trackers) == 0:
        faces = engine.face_detector(image)

        for face in faces:
            face = face.astype('int')[:4]
            x1, y1, x2, y2 = face
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(x1, y1, x2, y2)
            tracker.start_track(rgb, rect)
            trackers.append(tracker)
            draw_image(face, image)
    else:
        for tracker in trackers:
            pos = tracker.get_position()
            face = np.array([pos.left(), pos.top(), pos.right(), pos.bottom()]).astype(int)
            draw_image(face, image)
            cv2.waitKey(10)
            tracker.update(rgb)

    cv2.imshow('', image)
    cv2.waitKey(10)