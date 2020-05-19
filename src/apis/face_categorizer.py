import numpy as np
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing.label import _label as preprocessor_label


le = preprocessor_label.LabelEncoder()

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
        self.label_indexes = le.fit_transform(labels)

    def add_vector(self, vector, label):
        self.vector_list.append(vector)
        self.labels.append(label)

    def _build_tree(self):
        X = self.vector_list
        if self.transposer is not None:
            X = self.transposer.transform(X)
        tree = KDTree(X, leaf_size=min(len(X)//2, 400))
        return tree

    def predict(self, embedding):
        if len(embedding.shape) == 1:
            embedding = np.expand_dims(embedding, 0)
        if self.transposer is not None:
            embedding = self.transposer.transform(embedding)

        top = min(len(self.vector_list), self.top)
        print('embedding', len(embedding))
        distances, indices = self.tree.query(embedding, k=top)

        # convert selected name ids from selected vectors
        selected_name_ids = np.take(self.label_indexes, indices)
        # print('indices', indices)
        for e in selected_name_ids:
            print('e', e)
        # count number of appearance of identity index
        identity_count_numbers = np.array([np.bincount(e) for e in selected_name_ids])
        print('identity_count_numbers', identity_count_numbers)
        # get idx of the most frequency
        max_count_indexes = np.argmax(identity_count_numbers, axis=1)
        print('max_count_indexs', max_count_indexes)
        # get max count value
        max_count_value = np.take(max_count_indexes, identity_count_numbers)
        # get number of the most frequency identity
        max_identity_indexes = np.take(selected_name_ids, max_count_indexes)
        # get the real name from identity index
        output_names = np.take(self.labels, np.ravel(max_identity_indexes), axis=1)

        scores = np.ravel(max_count_value/top)

        return output_names, scores

    def _transpose_pca(self):
        pca = PCA(n_components=self.embedding_size)
        pca.fit(self.embeddings)
        return pca