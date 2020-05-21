import cv2
import numpy as np
import pickle
from tqdm import tqdm
from glob import glob
from src.config.initializer import Initializer
from src.config.default import VECTORS_SET_PATH, AUG_VECTORS_SET_PATH
from src.utils.image_processing import augment_face

engine = Initializer()


def convert_image_to_vector(img_file):
    label = img_file.split('/')[-2]
    image = cv2.imread(img_file)
    augmented_face = augment_face(image)

    vectors = engine.face_verifier.predict([image, augmented_face])
    return vectors[:1], vectors[1:], label


def get_vector():
    image_files = glob('/home/tupm/HDD/projects/3dface/Face-Recognition-with-InsightFace/datasets/train/*/*')
    original_vectors = None
    augmented_vectors = None
    labels = []

    for img_path in tqdm(image_files):
        origin, augmented, label = convert_image_to_vector(img_path)
        if original_vectors is None:
            original_vectors = origin
            augmented_vectors = augmented
        else:
            original_vectors = np.concatenate([original_vectors, origin], axis=0)
            augmented_vectors = np.concatenate([augmented_vectors, augmented], axis=0)
        labels.append(label)

    return original_vectors, augmented_vectors, labels


if __name__ == '__main__':
    original_vectors, augmented_vectors, labels = get_vector()
    with open(VECTORS_SET_PATH, 'wb') as f:
        pickle.dump({'vectors': original_vectors, 'labels': labels}, f)

    with open(AUG_VECTORS_SET_PATH, 'wb') as f:
        pickle.dump({'vectors': augmented_vectors, 'labels': labels}, f)
