from types import FunctionType

from pandas.core.frame import DataFrame

from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from enum import Enum


class KNNAlgorithmType(Enum):
    auto = 'auto'
    ball_tree = 'ball_tree'
    kd_tree = 'kd_tree'
    brute = 'brute'


class KNNWeightType(Enum):
    uniform = 'uniform'
    distance = 'distance'


class KNNMetricsType(Enum):
    euclidean = 'euclidean'
    minkowski = 'minkowski'
    manhattan = 'manhattan'
    chebyshev = 'chebyshev'


class KNN:
    def __init__(self, n_neighbors=5, weights=KNNWeightType.uniform.value, algorithm=KNNAlgorithmType.auto.value,
                 leaf_size=30, p=2, metric=KNNMetricsType.minkowski.value, metric_params=None, n_jobs=None) -> None:
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs

    def run(self, train_x, train_y, test_x):
        classifier = neighbors.KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights,
                                                    algorithm=self.algorithm,
                                                    leaf_size=self.leaf_size, p=self.p, metric=self.metric,
                                                    metric_params=self.metric_params, n_jobs=self.n_jobs)
        classifier.fit(X=train_x, y=train_y)
        return classifier.predict(X=test_x)
