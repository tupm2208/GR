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
    print(len(image_files))
    vectors = None
    labels = []
    labels_map = {}
    count = 0

    for img_path in tqdm(image_files):
        origin, label = convert_image_to_vector(img_path)
        if vectors is None:
            vectors = origin
        else:
            vectors = np.concatenate([vectors, origin], axis=0)

        if label not in labels_map.keys():
            count += 1
            labels_map[label] = count

        labels.append(labels_map[label])

    return vectors, labels, labels_map


if __name__ == '__main__':
    original_vectors, labels, labels_map = get_vector()
    with open(VECTORS_SET_PATH, 'wb') as f:
        pickle.dump({'vectors': original_vectors, 'labels': labels, 'labels_map': labels_map}, f)
    print(labels)
    print(labels_map)
    print(original_vectors.shape)

