import numpy as np
from sklearn.neighbors import KDTree


def detect_in_scope(vector_list, label_list):
    X = vector_list

    tree = KDTree(X, leaf_size=min(len(X) // 2, 400))
    return tree