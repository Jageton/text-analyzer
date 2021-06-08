from enum import Enum

from sklearn.cluster import DBSCAN as SkDBSCAN


class DBSCANMetricType(Enum):
    cityblock = 'cityblock'
    cosine = 'cosine'
    euclidean = 'euclidean'
    l1 = 'l1'
    l2 = 'l2'
    manhattan = 'manhattan'
    braycurtis = 'braycurtis'
    canberra = 'canberra'
    chebyshev = 'chebyshev'
    correlation = 'correlation'
    dice = 'dice'
    hamming = 'hamming'
    jaccard = 'jaccard'
    kulsinski = 'kulsinski'
    mahalanobis = 'mahalanobis'
    minkowski = 'minkowski'
    rogerstanimoto = 'rogerstanimoto'
    russellrao = 'russellrao'
    seuclidean = 'seuclidean'
    sokalmichener = 'sokalmichener'
    sokalsneath = 'sokalsneath'
    sqeuclidean = 'sqeuclidean'
    yule = 'yule'


class DBSCANAlgorithmType(Enum):
    auto = 'auto'
    ball_tree = 'ball_tree'
    kd_tree = 'kd_tree'
    brute = 'brute'


class DBSCAN:

    def __init__(self, eps=0.5, min_samples=5, metric=DBSCANMetricType.euclidean.value, metric_params=None,
                 algorithm=DBSCANAlgorithmType.auto.value, leaf_size=30, p=None, n_jobs=None):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.metric_params = metric_params
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs

    def run(self, X):
        db = SkDBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric, metric_params=self.metric_params,
                      algorithm=self.algorithm, leaf_size=self.leaf_size, p=self.p, n_jobs=self.n_jobs)
        db.fit(X)
        return db.labels_
