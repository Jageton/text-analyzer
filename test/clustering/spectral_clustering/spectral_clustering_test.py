import unittest

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from sklearn import datasets
from sklearn.datasets import make_blobs

from clustering.spectral_clustering.spectral_clustering import SpectralCluster


class SpectralClusterTest(unittest.TestCase):

    def test_1000_random_points(self):
        # Configuration options
        num_samples_total = 1000
        cluster_centers = [(20, 20), (4, 4)]
        num_classes = len(cluster_centers)

        # Generate data
        X, targets = make_blobs(n_samples=num_samples_total, centers=cluster_centers, n_features=num_classes,
                                center_box=(0, 1), cluster_std=2)
        predict = SpectralCluster(n_clusters=num_classes).run(X)

        # Generate scatter plot for training data
        colors = list(map(lambda x: '#3b4cc0' if x == 1 else '#b40426', predict))
        plt.scatter(X[:, 0], X[:, 1], c=colors, marker="o", picker=True)
        plt.title('Clusterization result')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

        # Asserts:
        first_cluster_cnt = list(predict).count(0)
        second_cluster_cnt = list(predict).count(1)
        self.assertEqual(first_cluster_cnt, second_cluster_cnt)

    def test_6_known_points(self):
        # Generate data
        X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

        predict = SpectralCluster(n_clusters=2).run(X)
        predict_list = list(predict)

        # Asserts:
        self.assertEqual(predict_list[0], 0)
        self.assertEqual(predict_list[1], 0)
        self.assertEqual(predict_list[2], 0)

        self.assertEqual(predict_list[3], 1)
        self.assertEqual(predict_list[4], 1)
        self.assertEqual(predict_list[5], 1)

    def test_iris(self):
        iris = datasets.load_iris()
        iris_frame = DataFrame(iris.data)
        iris_frame.columns = iris.feature_names
        iris_frame['target'] = iris.target
        result = SpectralCluster(n_clusters=3).run(iris_frame)
        target_list = list(iris_frame['target'])

        # Asserts:
        first_target_cluster_cnt = target_list.count(0)
        second_target_cluster_cnt = target_list.count(1)
        third_target_cluster_cnt = target_list.count(2)

        first_cluster_cnt = list(result).count(0)
        second_cluster_cnt = list(result).count(1)
        third_cluster_cnt = list(result).count(2)

        print('Изначальное распределение точек по кластерам: ',
              first_target_cluster_cnt, second_target_cluster_cnt, third_target_cluster_cnt)
        print('Распределение точек по кластерам после кластеризации: ',
              first_cluster_cnt, second_cluster_cnt, third_cluster_cnt)


if __name__ == '__main__':
    unittest.main()
