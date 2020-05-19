import numpy as np


class Vector:
    def __init__(self, embeddings=None, label=None, score=None):
        self.embeddings = embeddings
        self.label = label
        self.score = score

