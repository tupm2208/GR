import numpy as np
from src.core.tracker import Tracker
from .face_categorizer import FaceCategorizer
from src.utils.standards import load_vectors, handle_face_result
from .trackers import Trackers

data = load_vectors()


class CounterTrackers(Trackers):
    def __init__(self):
        super().__init__()
        self.recognizer = None

    # def update(self, bgr, locations, embeddings, identities):
    #     matched_idxs = []
    #     unmatched_idxs = list(range(len(locations)))

    #     for track in self.track_list:
    #         location_idx = track.is_match(np.take(locations, unmatched_idxs, axis=0))
    #         if location_idx is not None:
    #             idx = unmatched_idxs[location_idx]
    #             del unmatched_idxs[location_idx]
    #             location_idx = idx
    #             # print(identities[location_idx])
    #             track.custom_update(bgr, locations[location_idx], embeddings[location_idx], identities[location_idx])
    #             matched_idxs.append(location_idx)
    #             track.did_match()
    #         else:
    #             track.did_not_match()
    #             track.custom_update(bgr)

    #     self._delete_tracker()

    #     self._create_new_trackers(locations, matched_idxs, identities, bgr, embeddings)

    def _delete_tracker(self):
        
        for idx in reversed(range(len(self.track_list))):
            if not self.track_list[idx].is_valid():
                print('deleted')
                del self.track_list[idx]