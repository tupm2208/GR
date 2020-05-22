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

    vectors = engine.face_verifier.predict(image)
    return vectors, label


def get_vector():
    image_files = glob('/home/tupm/HDD/projects/3dface/Face-Recognition-with-InsightFace/datasets/vn/*/*')
    vectors = None
    labels = []

    for img_path in tqdm(image_files):
        origin, label = convert_image_to_vector(img_path)
        if vectors is None:
            vectors = origin
        else:
            vectors = np.concatenate([vectors, origin], axis=0)
        labels.append(label)

    return vectors, labels


if __name__ == '__main__':
    original_vectors, labels = get_vector()
    with open(VECTORS_SET_PATH, 'wb') as f:
        pickle.dump({'vectors': original_vectors, 'labels': labels}, f)

    print(original_vectors.shape)

