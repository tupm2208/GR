from src.core.tracker import Tracker


class Trackers:
    def __init__(self):
        self.track_list = []
        self.auto_increase = 0
        self.user_dict_vectors = {}

    def update(self, bgr, locations, identities):
        matched_idxs = []

        for track in self.track_list:
            location_idx = track.is_match(locations)
            if location_idx is not None:
                track.custom_update(bgr, locations[location_idx], identities[location_idx])
                matched_idxs.append(location_idx)
                track.did_match()
            else:
                track.did_not_match()
                track.custom_update(bgr)

        for idx in reversed(range(len(self.track_list))):
            if not self.track_list[idx].is_valid():
                del self.track_list[idx]

        for i in range(len(locations)):
            if i not in matched_idxs:
                identity = identities[i]
                if identity is None or not identity:
                    identity = self._get_new_id()
                self.track_list.append(Tracker(bgr, locations[i], identity))

    def _get_new_id(self):
        self.auto_increase += 1
        return self.auto_increase
