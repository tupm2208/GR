import cv2
import time
from src.config.initializer import Initializer
from src.apis.trackers import Trackers
from src.apis.counter_trackers import CounterTrackers
from src.utils.image_processing import get_image_face_from_bboxes
from src.utils.standards import handle_face_result, draw_track_list
from src.config.default import GATE_STREAM, COUNTER_STREAM, TRACKING_INTERVAL, IS_RECORD

engine = Initializer()

if IS_RECORD:
    output_video_path = 'assets/output_videos'
    import os
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    output_path = os.path.join(output_video_path, dt_string)
    os.makedirs(output_path, exist_ok=True)
    recorder_shape = (848, 480)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    raw_gate_recorder = cv2.VideoWriter(os.path.join(output_path, 'raw_gate.avi'), fourcc, 30, recorder_shape)
    im_gate_recorder = cv2.VideoWriter(os.path.join(output_path, 'im_gate.avi'), fourcc, 30, recorder_shape)
    raw_counter_recorder = cv2.VideoWriter(os.path.join(output_path, 'raw_counter.avi'), fourcc, 30, recorder_shape)
    im_counter_recorder = cv2.VideoWriter(os.path.join(output_path, 'im_counter.avi'), fourcc, 30, recorder_shape)


class MainStream:

    def __init__(self):
        self.gate_video = cv2.VideoCapture(GATE_STREAM)
        self.counter_video = cv2.VideoCapture(COUNTER_STREAM)

        self.trackers = Trackers()
        self.counter_trackers = CounterTrackers()
        self.interval = TRACKING_INTERVAL
        self.count = 0

    def process_stream(self):
        ret1, image = self.gate_video.read()
        ret2, image2 = self.counter_video.read()
        # ret2, image2 = ret1, image.copy()
        if image is None or image2 is None:
            return
        if IS_RECORD:
            raw_gate_recorder.write(image)
            raw_counter_recorder.write(image2)

        if self.count % self.interval == 0:
            self.t1 = time.time()
            faces = engine.face_detector([image, image2], 0.9)
        else:
            faces = [[], []]
        self.count += 1
        # print(int(1 / (time.time() - t1)))
        gate_faces = faces[0]
        counter_faces = faces[1]

        gate_face_images = get_image_face_from_bboxes(image, gate_faces)
        counter_face_images = get_image_face_from_bboxes(image2, counter_faces)

        face_images = gate_face_images + counter_face_images

        vectors = engine.face_verifier.predict(face_images)

        gate_vectors = vectors[:len(gate_face_images)]
        counter_vectors = vectors[len(gate_face_images):]
        labels_map = self.trackers.labels_map
        # print(list(labels_map.keys())[-1], list(labels_map.values())[-2:])
        self._hanle_and_update(self.trackers, self.trackers.recognizer, gate_vectors, image, gate_faces, labels_map)
        self._hanle_and_update(self.counter_trackers, self.trackers.recognizer, counter_vectors, image2, counter_faces, labels_map)

        fps = int(self.interval / (time.time() - self.t1))
        self._draw_fps(image, image2, fps)
        if IS_RECORD:
            im_gate_recorder.write(image)
            im_counter_recorder.write(image2)

        cv2.imshow('', image)

        return image, image2, counter_face_images[:1]

    def _hanle_and_update(self, trackers, recognizer, vectors, image, faces, labels_map):
        names = handle_face_result(recognizer, vectors)
        trackers.update(image, faces, vectors[:len(faces)], names)
        draw_track_list(trackers.track_list, image, labels_map)

    def _draw_fps(self, image, image2, fps):
        cv2.putText(image, f'FPS: {fps}', (30, 30), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 255, 0))
        cv2.putText(image2, f'FPS: {fps}', (30, 30), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 255, 0))
