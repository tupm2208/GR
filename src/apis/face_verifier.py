import numpy as np
import tensorflow as tf
from src.models.arcface.net import ArcFaceModel
from src.utils.image_processing import normalize_face_image
from src.utils.standards import l2_norm


class FaceVerifier:
    def __init__(self, ckpt_path):
        # self.model = ArcFaceModel(112)
        # self.model.load_weights(ckpt_path)

        # converter = tf.lite.TFLiteConverter.from_saved_model(
        #     '/home/tupm/HDD/projects/3dface/facial_verification_system/trained_model/pb_model')
        # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        # self.model = converter.convert()
        self.model = tf.saved_model.load(ckpt_path)

    def predict(self, bgr_images):
        if len(bgr_images) == 0:
            return np.empty((0, 512))
        bgr_images = normalize_face_image(bgr_images)
        embeddings = self.model(bgr_images)

        return l2_norm(embeddings)


if __name__ == '__main__':
    import numpy as np
    face_verifier = FaceVerifier('/home/tupm/HDD/projects/3dface/facial_verification_system/trained_model/pb_model')
    images = np.ones((112, 112, 3))
    face_verifier.predict(images)