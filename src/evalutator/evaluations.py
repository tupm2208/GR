import os
import cv2
from glob import glob
from src.apis.face_categorizer import FaceCategorizer
from src.config.initializer import Initializer
from src.utils.standards import load_vectors
from tqdm import tqdm

image_paths = glob('/home/tupm/SSD/datasets/celebrity/main/test/*/*') + glob('/home/tupm/SSD/datasets/celebrity/main/val/*/*')
engine = Initializer()
verifier = engine.face_verifier
data = load_vectors()
categorizer = FaceCategorizer(data[0], data[1], embedding_size=10)
f = open('evaluate.txt', 'w+')
correct = 0
total = 0


def handle_predict(img_path):
    global total, correct, f
    gr = str(int(img_path.split('/')[-2]) + 1)
    img = cv2.imread(img_path)
    embedding = verifier.predict(img)
    names, score = categorizer.predict(embedding)
    f.write(f'{img_path}\n{names[0]}\n{score[0]}\n\n')

    total += 1
    if gr == names[0]:
        correct += 1


for img_path in tqdm(image_paths):
    handle_predict(img_path)
    # print(100 * correct / total)

f.close()
