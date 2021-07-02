import unittest

import matplotlib.pyplot as plt
import numpy as np
from pandas.core.frame import DataFrame
from sklearn import datasets
from sklearn.datasets import make_blobs

from clustering.k_means.k_means import KMeans

colors = ['#3b4cc0', '#db2c2c', '#13e700', '#ffdd00', '#ff52d5',
          '#08c7c9', '#5e2594', '#a0ff45', '#f36c1f', '#B3B6B7']


class KMeansTest(unittest.TestCase):

    def test_1000_random_points(self):
        # Configuration options
        num_samples_total = 1000
        cluster_centers = [(20, 20), (4, 4), (20, 4), (4, 20)]
        num_classes = len(cluster_centers)

        # Generate data
        X, targets = make_blobs(n_samples=num_samples_total, centers=cluster_centers, n_features=num_classes,
                                center_box=(0, 1), cluster_std=2)
        predict = KMeans(n_clusters=num_classes).run(X)

        # Generate scatter plot for training data
        plt.scatter(X[:, 0], X[:, 1], c=_get_colors(predict), marker="o", picker=True)
        plt.title('Clusterization result')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

        # Asserts:
        first_cluster_cnt = list(predict).count(0)
        second_cluster_cnt = list(predict).count(1)
        third_cluster_cnt = list(predict).count(2)
        forth_cluster_cnt = list(predict).count(3)

        self.assertEqual(first_cluster_cnt, second_cluster_cnt)
        self.assertEqual(second_cluster_cnt, third_cluster_cnt)
        self.assertEqual(third_cluster_cnt, forth_cluster_cnt)

    def test_6_known_points(self):
        # Generate data
        X = np.array([[1, 2], [1, 4], [1, 0],
                      [10, 2], [10, 4], [10, 0]])
        predict = KMeans(n_clusters=2, random_state=0).run(X)
        predict_list = list(predict)

        # Asserts:
        self.assertEqual(predict_list[0], 1)
        self.assertEqual(predict_list[1], 1)
        self.assertEqual(predict_list[2], 1)

        self.assertEqual(predict_list[3], 0)
        self.assertEqual(predict_list[4], 0)
        self.assertEqual(predict_list[5], 0)

    def test_iris(self):
        iris = datasets.load_iris()
        iris_frame = DataFrame(iris.data)
        iris_frame.columns = iris.feature_names
        iris_frame['target'] = iris.target
        result = KMeans(n_clusters=3).run(iris_frame)
        target_list = list(iris_frame['target'])

        # Asserts:
        first_target_cluster_cnt = target_list.count(0)
        second_target_cluster_cnt = target_list.count(1)
        third_target_cluster_cnt = target_list.count(2)

        first_cluster_cnt = list(result).count(0)
        second_cluster_cnt = list(result).count(1)
        third_cluster_cnt = list(result).count(2)

        run_algorithm_for_2_columns(result, iris_frame)

        print('Изначальное распределение точек по кластерам: ',
              first_target_cluster_cnt, second_target_cluster_cnt, third_target_cluster_cnt)
        print('Распределение точек по кластерам после кластеризации: ',
              first_cluster_cnt, second_cluster_cnt, third_cluster_cnt)


def run_algorithm_for_2_columns(predict, dataframe):
    plt.scatter(dataframe.values[:, 1], dataframe.values[:, 2], c=_get_colors(predict), marker="o", picker=True)
    plt.title('Clusterization result')
    plt.xlabel(dataframe.columns[1])
    plt.ylabel(dataframe.columns[2])
    plt.show()


def _get_colors(predict):
    return list(map(get_color, predict))


def get_color(x):
    if x < 0:
        return '#000000'
    return colors[x % len(colors)]


if __name__ == '__main__':
    unittest.main()
