from src.config.default import FACEBOXES_CKPT, ARC_FACE_CKPT
from src.apis.face_detector import FaceDetector
from src.apis.face_verifier import FaceVerifier


class Initializer:
    def __init__(self):
        self.face_detector = FaceDetector(FACEBOXES_CKPT)
        # self.face_verifier = FaceVerifier(ARC_FACE_CKPT)

