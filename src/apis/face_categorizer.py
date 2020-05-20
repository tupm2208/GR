import numpy as np
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA


class FaceCategorizer:
    def __init__(self, vector_list, labels, embedding_size=None, top=10):
        self.vector_list = vector_list
        self.embedding_size = embedding_size
        self.labels = labels
        if embedding_size is not None:
            self.transposer = self._transpose_pca()
        else:
            self.transposer = None

        self.tree = self._build_tree()
        self.top = top

    def add_vector(self, vector, label):
        self.vector_list.append(vector)
        self.labels.append(label)

    def _build_tree(self):
        X = self.vector_list
        if self.transposer is not None:
            X = self.transposer.transform(X)
        tree = KDTree(X, leaf_size=min(len(X)//2, 400))
        return tree

    def _get_selected_name(self, selected_indexes):
        selected_names = np.take(self.labels, selected_indexes)
        names = []
        counts = []
        for e in selected_names:
            unique, count = np.unique(e, return_counts=True)
            names.append(unique[0])
            counts.append(count[0])

        return np.array(names), np.array(counts)

    def predict(self, embedding):
        if len(embedding) == 0:
            return [], []
        if len(embedding.shape) == 1:
            embedding = np.expand_dims(embedding, 0)
        if self.transposer is not None:
            embedding = self.transposer.transform(embedding)

        top = min(len(self.vector_list), self.top)
        distances, indices = self.tree.query(embedding, k=top)

        # get selected names
        names, counts = self._get_selected_name(indices)

        return names, counts/float(top)

    def _transpose_pca(self):
        pca = PCA(n_components=self.embedding_size)
        pca.fit(self.embeddings)
        return pca