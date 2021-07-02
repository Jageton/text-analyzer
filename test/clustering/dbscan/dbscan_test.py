import unittest

import numpy as np
from pandas import DataFrame
from sklearn import datasets
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from clustering.dbscan.dbscan import DBSCAN
from util.visualization import data_2d_visualization


class DBSCANTest(unittest.TestCase):

    def test_750_random_points(self):
        # Generate sample data
        centers = [[1, 1], [-5, -5], [5, -5]]
        X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4, random_state=0)
        X = StandardScaler().fit_transform(X)

        predict = DBSCAN(eps=0.3).run(X)
        data_2d_visualization(X, predict)
        predict_list = list(predict)

        # Asserts:
        first_cluster_cnt = predict_list.count(0)
        second_cluster_cnt = predict_list.count(1)
        third_cluster_cnt = predict_list.count(2)
        self.assertEqual(first_cluster_cnt, second_cluster_cnt)
        self.assertEqual(second_cluster_cnt, third_cluster_cnt)

    def test_6_random_points(self):
        X = np.array([[1, 2], [2, 2], [2, 3],
                      [8, 7], [8, 8], [25, 80]])
        predict = DBSCAN(eps=3, min_samples=2).run(X)
        data_2d_visualization(X, predict)
        predict_list = list(predict)

        # Asserts:
        self.assertEqual(predict_list[0], 0)
        self.assertEqual(predict_list[1], 0)
        self.assertEqual(predict_list[2], 0)

        self.assertEqual(predict_list[3], 1)
        self.assertEqual(predict_list[4], 1)
        self.assertEqual(predict_list[5], -1)

    def test_iris(self):
        iris = datasets.load_iris()
        iris_frame = DataFrame(iris.data)
        iris_frame.columns = iris.feature_names
        iris_frame['target'] = iris.target

        result = DBSCAN(eps=1).run(iris_frame)
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
