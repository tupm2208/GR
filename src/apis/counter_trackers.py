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
        self.vector_list = None
        self.label_list = None

    def _delete_tracker(self):
        
        for idx in reversed(range(len(self.track_list))):
            if not self.track_list[idx].is_valid():
                del self.track_list[idx]