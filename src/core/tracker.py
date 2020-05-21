import cv2
import numpy as np
from dlib import correlation_tracker, rectangle
from src.utils.standards import to_xywh, iou
from src.config.default import TIME_TO_LIVE, BOUNDING_LIMIT


class Tracker(correlation_tracker):
    def __init__(self, bgr, location, embeddings,identity):
        super().__init__()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
        location = np.array(location[:14], dtype=int)
        self._init_variable()
        self.img_h, self.img_w, _ = bgr.shape
        # self.scores = [location[-1]]
        self._convert_ldm_to_scale(location)
        self.current_location = location
        self._update_identity(identity)
        self.track(rgb, location)
        self._update_embeddings(embeddings)

    def _init_variable(self):
        self.ttl = 0
        self.original_names = []
        self.cut_names = []
        self.origin_vectors = None
        self.augmented_vectors = None

    def _update_identity(self, identity):
        # print(identity)
        org_name, cut_name = identity
        self.original_names.append(org_name)
        self.cut_names.append(cut_name)

    def _update_embeddings(self, embeddings):
        origin, augmented = embeddings
        origin = np.expand_dims(origin, 0)
        augmented = np.expand_dims(augmented, 0)
        if self.origin_vectors is None:
            self.origin_vectors = origin
            self.augmented_vectors = augmented
        else:
            self.origin_vectors = np.concatenate([self.origin_vectors, origin])
            self.augmented_vectors = np.concatenate([self.augmented_vectors, augmented])

    def custom_update(self, bgr, location=None, embeddings=None, identity=None):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
        if location is not None:
            location = np.array(location[:14], dtype=int)

            x1, y1, x2, y2 = location[:4]
            rec = rectangle(x1, y1, x2, y2)
            self._convert_ldm_to_scale(location)
            self.current_location = location
            self.update(rgb, rec)
        else:
            self.current_location = self.get_location()
            self.update(rgb)

        if identity is not None:
            self._update_identity(identity)

        if embeddings is not None:
            self._update_embeddings(embeddings)

    def track(self, rgb, location):
        position = location[:4]
        x1, y1, x2, y2 = position
        rec = rectangle(x1, y1, x2, y2)
        self.start_track(rgb, rec)

    def _convert_ldm_to_scale(self, location):
        x, y, w, h = to_xywh(location.copy())
        self.landmarks = np.reshape(location[4:14], (5, 2)).astype(float)
        self.landmarks[:, 0] = (self.landmarks[:, 0] - x) / w
        self.landmarks[:, 1] = (self.landmarks[:, 1] - y) / h

    def _recover_ldm_from_scale(self, position):
        x, y, w, h = to_xywh(position)
        ldmsx = self.landmarks[:, 0] * w + x
        ldmsy = self.landmarks[:, 1] * h + y
        ldmsx = np.reshape(ldmsx, (5, 1))
        ldmsy = np.reshape(ldmsy, (5, 1))

        return np.ravel(np.concatenate([ldmsx, ldmsy], axis=1).astype(int))

    def get_location(self):
        pos = self.get_position()
        x1 = pos.left()
        y1 = pos.top()
        x2 = pos.right()
        y2 = pos.bottom()

        position = np.array([x1, y1, x2, y2])
        ldms = self._recover_ldm_from_scale(position)
        return np.concatenate([position.astype(int), ldms])

    def get_hard_vectors(self, get_id_label):
        name = self.get_identity()
        # if name == 'trachpro':
        #     return None
        step = len(self.origin_vectors)//20
        if name == 'unknown':
            if step == 0:
                return None
            chosen_vectors = self.origin_vectors
        else:
            chosen_idxs = np.where(np.array(self.original_names) != name)
            chosen_vectors = np.take(self.origin_vectors, chosen_idxs, axis=0)
            chosen_vectors = np.squeeze(chosen_vectors, axis=0)
            if min(len(chosen_vectors), 50) == 0:
                return None
            step = len(chosen_vectors)//min(len(chosen_vectors), 50)
        # step = 1
        chosen_vectors = chosen_vectors[0::step]
        # print(chosen_vectors.shape, name)

        identity = get_id_label() if name == 'unknown' else name

        return chosen_vectors, [identity] * len(chosen_vectors)

    def did_not_match(self):
        self.ttl += 1

    def did_match(self):
        self.ttl = 0

    def is_valid(self):
        x1, y1, x2, y2 = self.get_location()[:4]
        x1 -= BOUNDING_LIMIT
        y1 -= BOUNDING_LIMIT
        x1 += BOUNDING_LIMIT
        y1 += BOUNDING_LIMIT
        if self.ttl > TIME_TO_LIVE:
            return False

        if 0 >= x1 or 0 >= y1 or self.img_w <= x1 or self.img_h <= y2:
            return False

        return True

    def get_identity(self):
        trace = self.original_names
        n_names = len(trace)
        unique, count = np.unique(trace, return_counts=True)
        order = np.argsort(count)[::-1]
        count = np.take(count, order)
        unique = np.take(unique, order)
        if len(unique) == 1 or unique[0] != 'unknown' or count[0]/n_names >= 0.88:
            return unique[0]
        else:
            return unique[1]

    def is_match(self, locations):
        if len(locations) == 0:
            return None
        gr_box = self.get_location()[:4]

        areas = [iou(gr_box, pred_box[:4]) for pred_box in locations]
        idx = np.argmax(areas)

        if areas[idx] == 0:
            return None
        else:
            return idx