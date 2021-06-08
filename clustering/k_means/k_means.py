from enum import Enum

import sklearn.cluster as sk_cluster


class KMeansInitType(Enum):
    k_means = 'random'
    k_means_plusplus = 'k-means++'


class KMeansAlgorithmType(Enum):
    auto = 'auto'
    full = 'full'
    elkan = 'elkan'


class KMeans:

    def __init__(self, n_clusters: int = 4, init=KMeansInitType.k_means_plusplus.value, n_init: int = 10,
                 max_iter: int = 300, tol: float = 1e-4, verbose: int = 0, random_state: int = None,
                 copy_x: bool = True, algorithm=KMeansAlgorithmType.auto.value):
        self.init = init
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.algorithm = algorithm

    def run(self, X):
        k_means = sk_cluster.KMeans(init=self.init, n_clusters=self.n_clusters, n_init=self.n_init,
                                    max_iter=self.max_iter, tol=self.tol, verbose=self.verbose,
                                    random_state=self.random_state, copy_x=self.copy_x, algorithm=self.algorithm)
        predict = k_means.fit(X).predict(X)
        return predict
