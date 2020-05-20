import cv2
import time
from src.config.initializer import Initializer
from src.apis.trackers import Trackers
from src.utils.image_processing import get_image_face_from_bboxes
from src.utils.standards import load_vectors, draw_image
from src.apis.face_categorizer import FaceCategorizer

engine = Initializer()

video = cv2.VideoCapture('/home/tupm/SSD/datasets/video_test/64888330720332379881.mp4')

trackers = Trackers()
data = load_vectors()
face_categorizer = FaceCategorizer(data[0], data[1])


while True:
    t1 = time.time()
    ret, image = video.read()

    if not ret:
        break

    faces = engine.face_detector(image)

    face_images = get_image_face_from_bboxes(image, faces)

    vectors = engine.face_verifier.predict(face_images)

    names, scores = face_categorizer.predict(vectors)
    if len(names) != 0:
        names[scores <= 0.9] = 'unknown'

    trackers.update(image, faces, names)
    for tracker in trackers.track_list:
        loc = tracker.current_location
        draw_image(loc, image, tracker.get_identity())

    cv2.imshow('', image)
    print(1/(time.time()-t1))
    cv2.waitKey(1)