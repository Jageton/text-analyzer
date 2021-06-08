from sklearn.cluster import Birch as skBirch


class Birch:

    def __init__(self, threshold=0.5, branching_factor=50, n_clusters=3, compute_labels=True, copy=True):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters
        self.compute_labels = compute_labels
        self.copy = copy

    def run(self, X):
        brc = skBirch(threshold=self.threshold, branching_factor=self.branching_factor, n_clusters=self.n_clusters,
                      compute_labels=self.compute_labels, copy=self.copy)
        brc.fit(X)
        return brc.predict(X)
