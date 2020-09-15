import numpy as np
import pickle
import cv2
from src.config.default import VECTORS_SET_PATH, AUG_VECTORS_SET_PATH


def to_xywh(bbox):
    x1, y1, x2, y2 = bbox[:4]

    return np.array([x1, y1, x2 - x1, y2 - y1])


def draw_image(rec, image, name=None):
    x1, y1, x2, y2 = rec[:4]

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # draw_landmark(image, rec)

    if name is not None:
        cv2.putText(image, str(name), (x1, y1), cv2.FONT_ITALIC, 1, (0, 255, 0))


def draw_landmark(image, rec):
    if len(rec) > 10:
        rec = np.reshape(rec[4:14], (5, 2))

    for x, y in rec:
        cv2.circle(image, (x, y), 2, (0, 255, 0), 3)


def load_vectors(ltype=1):
    path = VECTORS_SET_PATH if ltype == 1 else AUG_VECTORS_SET_PATH
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data['vectors'], data['labels'], data['labels_map']


def l2_norm(x, axis=1):
    """l2 norm"""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    output = x / norm

    return output


def iou(boxes1, boxes2):
    """Computes pairwise intersection-over-union between two box collections.

    Arguments:
        boxes1: a float tensor with shape [N, 4].GT
        boxes2: a float tensor with shape [M, 4].ANCHOR
    Returns:
        a float tensor with shape [N, M] representing pairwise iou scores.
    """

    intersections = intersection(boxes1, boxes2)  # transfored wrong

    areas1 = area(boxes1)
    areas2 = area(boxes2)
    unions = areas1 + areas2 - intersections

    return intersections / unions


def intersection(boxes1, boxes2):
    """Compute pairwise intersection areas between boxes.

    Arguments:
        boxes1: a float tensor with shape [N, 4].
        boxes2: a float tensor with shape [M, 4].
    Returns:
        a float tensor with shape [N, M] representing pairwise intersections.
    """
    ##########np transformed wrong need review
    ymin1, xmin1, ymax1, xmax1 = boxes1[:4]
    ymin2, xmin2, ymax2, xmax2 = boxes2[:4]
    # they all have shapes like [None, 1]

    x_min_max = max(xmin1, xmin2)
    y_min_max = max(ymin1, ymin2)

    x_max_min = min(xmax1, xmax2)
    y_max_min = min(ymax1, ymax2)

    iw = max(x_max_min - x_min_max, 0)
    ih = max(y_max_min - y_min_max, 0)

    return iw * ih


def area(boxes):
    """Computes area of boxes.

    Arguments:
        boxes: a float tensor with shape [N, 4].
    Returns:
        a float tensor with shape [N] representing box areas.
    """

    ymin, xmin, ymax, xmax = boxes[:4]
    return (ymax - ymin) * (xmax - xmin)


def handle_face_result(face_categorizer, vectors):
    names, scores = face_categorizer.predict(vectors)
    if len(names) != 0:
        names = names.tolist()
        for idx in range(len(scores)):
            if scores[idx] <= 0.9:
                names[idx] = 'unknown'

    return names


def draw_track_list(track_list, image, labels_map):
    for tracker in track_list:
        loc = tracker.current_location
        identity = tracker.get_identity()
        if identity != 'unknown':
            if int(identity) in labels_map.values():
                identity = list(labels_map.keys())[list(labels_map.values()).index(int(identity))]

        draw_image(loc, image, identity)
