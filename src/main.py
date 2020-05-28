import cv2
import time
import numpy as np
from src.config.initializer import Initializer
from src.apis.trackers import Trackers
from src.apis.counter_trackers import CounterTrackers
from src.utils.image_processing import get_image_face_from_bboxes
from src.utils.standards import load_vectors, draw_image, handle_face_result, draw_track_list
from src.apis.face_categorizer import FaceCategorizer

engine = Initializer()
# counter_video = cv2.VideoCapture('http://192.168.1.222:4747/video')
# gate_video = cv2.VideoCapture('http://192.168.1.120:4747/video')
gate_video = cv2.VideoCapture('assets/test_videos/56354369654362717692.mp4')
counter_video = cv2.VideoCapture('assets/test_videos/23920561543638656951.mp4')

trackers = Trackers()
counter_trackers = CounterTrackers()
data = load_vectors()
face_categorizer = FaceCategorizer(data[0], data[1])


is_record = False
if is_record:
    output_video_path = 'assets/output_videos'
    import os
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    output_path = os.path.join(output_video_path, dt_string)
    os.makedirs(output_path, exist_ok=True)
    recorder_shape = (640, 480)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    raw_gate_recorder = cv2.VideoWriter(os.path.join(output_path, 'raw_gate.avi'), fourcc, 30, recorder_shape)
    im_gate_recorder = cv2.VideoWriter(os.path.join(output_path, 'im_gate.avi'), fourcc, 30, recorder_shape)
    raw_counter_recorder = cv2.VideoWriter(os.path.join(output_path, 'raw_counter.avi'), fourcc, 30, recorder_shape)
    im_counter_recorder = cv2.VideoWriter(os.path.join(output_path, 'im_counter.avi'), fourcc, 30, recorder_shape)

interval = 3
count = 0

while True:

    # ret1, image = gate_video.read()
    # ret2, image2 = counter_video.read()
    ret1, image = gate_video.read()
    ret2, image2 = counter_video.read()
    # ret2, image2 = ret1, image.copy()

    if image is None or image2 is None:
        break
    if is_record:
        raw_gate_recorder.write(image)
        raw_counter_recorder.write(image2)

    if count % interval == 0:
        t1 = time.time()
        faces = engine.face_detector([image, image2], 0.9)
    else:
        faces = [[], []]
    count += 1
    # print(int(1 / (time.time() - t1)))
    gate_faces = faces[0]
    counter_faces = faces[1][:1]

    gate_face_images = get_image_face_from_bboxes(image, gate_faces)
    counter_face_images = get_image_face_from_bboxes(image2, counter_faces)

    face_images = gate_face_images + counter_face_images

    vectors = engine.face_verifier.predict(face_images)

    gate_vectors = vectors[:len(gate_face_images)]
    counter_vectors = vectors[len(gate_face_images):]

    gate_names = handle_face_result(trackers.recognizer, gate_vectors)
    trackers.update(image, gate_faces, vectors[:len(gate_faces)], gate_names)
    draw_track_list(trackers.track_list, image)
    # out_img = image

    counter_names = handle_face_result(trackers.recognizer, counter_vectors)
    counter_trackers.update(image2, counter_faces, vectors[len(gate_faces):], counter_names)
    draw_track_list(counter_trackers.track_list, image2)

    # out_img = image2
    fps = int(interval / (time.time() - t1))
    cv2.putText(image, f'FPS: {fps}', (30, 30), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 255, 0))
    cv2.putText(image2, f'FPS: {fps}', (30, 30), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 255, 0))
    if is_record:
        im_gate_recorder.write(image)
        im_counter_recorder.write(image2)
    cv2.imshow('gate', image)
    cv2.imshow('counter', image2)
    k = cv2.waitKey(1)
    if ord('q') == k:
        break
