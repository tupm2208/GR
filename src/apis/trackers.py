import numpy as np
from src.core.tracker import Tracker
from .face_categorizer import FaceCategorizer
from src.utils.standards import load_vectors, handle_face_result

data = load_vectors()


class Trackers:
    def __init__(self):
        self.track_list = []
        self.auto_increase = 0
        self.user_dict_vectors = {}
        self.vector_list = data[0]
        self.label_list = data[1]
        self.recognizer = FaceCategorizer(self.vector_list, self.label_list)

    def update(self, bgr, locations, embeddings, identities):
        matched_idxs = []
        unmatched_idxs = list(range(len(locations)))

        for track in self.track_list:
            location_idx = track.is_match(np.take(locations, unmatched_idxs, axis=0))
            if location_idx is not None:
                idx = unmatched_idxs[location_idx]
                del unmatched_idxs[location_idx]
                location_idx = idx
                # print(identities[location_idx])
                track.custom_update(bgr, locations[location_idx], embeddings[location_idx], identities[location_idx])
                matched_idxs.append(location_idx)
                track.did_match()
            else:
                track.did_not_match()
                track.custom_update(bgr)

        self._delete_tracker()

        self._create_new_trackers(locations, matched_idxs, identities, bgr, embeddings)

    def _get_new_id(self):
        self.auto_increase += 1
        return self.auto_increase

    def _check_duplicate_person(self, vector_labels):
        vectors, labels = vector_labels
        checked_names = handle_face_result(self.recognizer, vectors[:10])
        print(checked_names)

    def _create_new_trackers(self, locations, matched_idxs, identities, bgr, embeddings):
        for i in range(len(locations)):
            if i not in matched_idxs:
                identity = identities[i]

                self.track_list.append(Tracker(bgr, locations[i], embeddings[i], identity))

    def _delete_tracker(self):
        for idx in reversed(range(len(self.track_list))):
            if not self.track_list[idx].is_valid():
                # print(self.track_list[idx].origin_vectors.shape, len(self.track_list[idx].original_names))
                vector_labels = self.track_list[idx].get_hard_vectors(self._get_new_id)
                if vector_labels is not None:
                    self._check_duplicate_person(vector_labels)
                    vectors, labels = vector_labels
                    if self.vector_list is None:
                        self.vector_list = vectors
                    else:
                        self.vector_list = np.concatenate([self.vector_list, vectors], axis=0)

                    self.label_list.extend(labels)
                    # if self.auto_increase == labels[0]:
                    #     self.label_list.extend(labels)
                    #     print(self.vector_list.shape)

                    self.recognizer = FaceCategorizer(self.vector_list, self.label_list)

                del self.track_list[idx]