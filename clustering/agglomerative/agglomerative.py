from enum import Enum
from sklearn.cluster import AgglomerativeClustering


class AffinityType(Enum):
    euclidean = "euclidean"
    manhattan = "manhattan"
    cosine = "cosine"
    precomputed = "precomputed"
    l1 = "l1"
    l2 = "l2"


class LinkageType(Enum):
    ward = "ward"
    average = "average"
    complete = "complete"
    single = "single"


class Aglomerative:
    def __init__(self, n_cluster: int = 2, affinity: AffinityType = AffinityType.euclidean.value, memory=None, connectivity=None,
                 compute_full_tree='auto', linkage:LinkageType=LinkageType.ward.value, distance_threshold:float=None,
                 compute_distances:bool=False) -> None:
        self.n_cluster = n_cluster
        self.affinity = affinity
        self.memory = memory
        self.connectivity = connectivity
        self.compute_full_tree = compute_full_tree
        self.linkage = linkage
        self.distance_threshold = distance_threshold
        self.compute_distances = compute_distances

    def run(self, X):
        clustering = AgglomerativeClustering(n_clusters=self.n_cluster, affinity=self.affinity, memory=self.memory,
                                             connectivity=self.connectivity, compute_full_tree=self.compute_full_tree, linkage=self.linkage,
                                             distance_threshold=self.distance_threshold, compute_distances=self.compute_distances)
        return clustering.fit_predict(X=X)
