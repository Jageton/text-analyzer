import unittest

import numpy as np
from pandas import DataFrame
from sklearn import datasets
from sklearn.datasets import make_blobs

from clustering.birch.birch import Birch
from util.visualization import data_2d_visualization


class DBSCANTest(unittest.TestCase):

    def test_450_random_points(self):
        X, clusters = make_blobs(n_samples=450, centers=6, cluster_std=0.70, random_state=0)
        predict = Birch(branching_factor=50, n_clusters=None, threshold=1.5).run(X)
        data_2d_visualization(X, predict)

    def test_6_known_points(self):
        # Generate data
        X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

        predict = Birch(n_clusters=2).run(X)
        data_2d_visualization(X, predict)
        predict_list = list(predict)

        # Asserts:
        self.assertEqual(predict_list[0], predict_list[1])
        self.assertEqual(predict_list[1], predict_list[2])

        self.assertEqual(predict_list[3], predict_list[4])
        self.assertEqual(predict_list[4], predict_list[5])

        self.assertNotEqual(predict_list[0], predict_list[3])

    def test_iris(self):
        iris = datasets.load_iris()
        iris_frame = DataFrame(iris.data)
        iris_frame.columns = iris.feature_names
        iris_frame['target'] = iris.target

        result = Birch(threshold=0.6).run(iris_frame)
        data_2d_visualization(iris_frame, result)
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
