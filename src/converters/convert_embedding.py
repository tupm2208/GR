import cv2
import numpy as np
import pickle
from tqdm import tqdm
from glob import glob
from src.config.initializer import Initializer
from src.config.default import VECTORS_SET_PATH

engine = Initializer()


def convert_image_to_vector(img_file):
    label = img_file.split('/')[-2]
    image = cv2.imread(img_file)
    vector = engine.face_verifier.predict(image)
    return vector, label


def get_vector():
    image_files = glob('/home/tupm/HDD/projects/3dface/Face-Recognition-with-InsightFace/datasets/train/*/*')
    vectors = None
    labels = []

    for img_path in tqdm(image_files):
        vector, label = convert_image_to_vector(img_path)
        if vectors is None:
            vectors = vector
        else:
            vectors = np.concatenate([vectors, vector], axis=0)
        labels.append(label)

    return vectors, labels


if __name__ == '__main__':
    vectors, labels = get_vector()
    with open(VECTORS_SET_PATH, 'wb') as f:
        pickle.dump({'vectors': vectors, 'labels': labels}, f)
