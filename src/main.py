import cv2
import time
import numpy as np
from src.config.initializer import Initializer
from src.apis.trackers import Trackers
from src.utils.image_processing import get_image_face_from_bboxes
from src.utils.standards import load_vectors, draw_image, handle_face_result
from src.apis.face_categorizer import FaceCategorizer

engine = Initializer()

gate_video = cv2.VideoCapture('/home/tupm/SSD/datasets/video_test/test/56354369654362717692.mp4')
counter_video = cv2.VideoCapture('/home/tupm/SSD/datasets/video_test/test/23920561543638656951.mp4')

trackers = Trackers()
counter_trackers = Trackers()
data = load_vectors()
face_categorizer = FaceCategorizer(data[0], data[1])

data = load_vectors(0)
cut_face_categorizer = FaceCategorizer(data[0], data[1])

ret1 = True
ret2 = True
shape1 = None

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
vw = cv2.VideoWriter('assets/output_videos/out_cpu.avi', fourcc, 30, (854, 480))

while True:
    t1 = time.time()

    if ret1:
        ret1, image = gate_video.read()

    if not ret1:
        image = np.zeros(shape1, dtype='uint8')
        ret2, image2 = counter_video.read()
    else:
        shape1 = image.shape
        image2 = np.zeros(shape1, dtype='uint8')

    if not ret1 and not ret2:
        break

    faces = engine.face_detector([image, image2])
    gate_faces = faces[0]
    counter_faces = faces[1]

    gate_face_images = get_image_face_from_bboxes(image, gate_faces)
    counter_face_images = get_image_face_from_bboxes(image2, counter_faces)

    face_images = gate_face_images + counter_face_images

    vectors = engine.face_verifier.predict(face_images)

    gate_vectors = vectors[:len(gate_face_images)]
    counter_vectors = vectors[len(gate_face_images):]

    gate_embeddings, gate_names = handle_face_result(gate_faces, face_categorizer, cut_face_categorizer, gate_vectors)

    if ret1:

        trackers.update(image, gate_faces, gate_embeddings, gate_names)
        for tracker in trackers.track_list:
            loc = tracker.current_location
            draw_image(loc, image, tracker.get_identity())
        out_img = image

    else:
        # gate_video = cv2.VideoCapture('/home/tupm/SSD/datasets/video_test/test/56354369654362717692.mp4')
        # face_categorizer = trackers.recognizer
        # ret1 = True
        # continue
        counter_embeddings, counter_names = handle_face_result(counter_faces, trackers.recognizer, trackers.recognizer,
                                                               counter_vectors)
        counter_trackers.update(image2, counter_faces, counter_embeddings, counter_names)
        for tracker in counter_trackers.track_list:
            loc = tracker.current_location
            draw_image(loc, image2, tracker.get_identity())

        out_img = image2
    fps = int(1 / (time.time() - t1))
    cv2.putText(out_img, f'FPS: {fps}', (30, 30), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 255, 0))
    vw.write(out_img)
    cv2.imshow('', out_img)
    cv2.waitKey(1)
