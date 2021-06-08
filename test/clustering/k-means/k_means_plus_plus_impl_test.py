import math
import random
import unittest

import pylab
from sklearn.datasets import make_blobs

from clustering.k_means.k_means_plus_plus_impl import KMeansPlusPlusImpl, Point


class KMeansImplTest(unittest.TestCase):

    def test_1000_random_points(self):
        # Configuration options
        num_samples_total = 1000
        cluster_centers = [(20, 20), (4, 4)]
        num_classes = len(cluster_centers)

        # Generate data
        X, targets = make_blobs(n_samples=num_samples_total, centers=cluster_centers, n_features=num_classes,
                                center_box=(0, 1), cluster_std=2)
        points = list(map(lambda x: Point(x[0], x[1]), X))
        _, cluster_center_trace = KMeansPlusPlusImpl(points, 2).run()

        self.show_cluster_analysis_results(points, [])

        # Asserts:
        first_cluster_cnt = list(map(lambda x: x.group, points)).count(0)
        second_cluster_cnt = list(map(lambda x: x.group, points)).count(1)
        self.assertEqual(first_cluster_cnt, second_cluster_cnt)

    def test_6_known_points(self):
        # Generate data
        points = [Point(1, 2), Point(1, 4), Point(1, 0), Point(10, 2), Point(10, 4), Point(10, 0)]
        _, cluster_center_trace = KMeansPlusPlusImpl(points, 2).run()

        # Asserts:
        first_cluster_cnt = list(map(lambda x: x.group, points)).count(0)
        second_cluster_cnt = list(map(lambda x: x.group, points)).count(1)
        self.assertEqual(first_cluster_cnt, second_cluster_cnt)

    def test_clustering_visualization(self):
        cluster_center_number = 5
        points_number = 2000
        radius = 50
        points = self.generate_points(points_number, radius)
        _, cluster_center_trace = KMeansPlusPlusImpl(points, cluster_center_number).run()

        self.show_cluster_analysis_results(points, cluster_center_trace)

    def generate_points(self, points_number, radius):
        points = [Point() for _ in range(2 * points_number)]
        count = 0
        for point in points:
            count += 1
            r = random.random() * radius
            angle = random.random() * 2 * math.pi
            point.x = r * math.cos(angle)
            point.y = r * math.sin(angle)
            if count == points_number - 1:
                break
        for index in range(points_number, 2 * points_number):
            points[index].x = 2 * radius * random.random() - radius
            points[index].y = 2 * radius * random.random() - radius
        return points

    def show_cluster_analysis_results(self, points, cluster_center_trace):
        color_store = ['or', 'og', 'ob', 'oc', 'om', 'oy', 'ok']
        pylab.figure(figsize=(9, 9), dpi=80)
        pylab.title('Clusterization result')
        pylab.xlabel('X')
        pylab.ylabel('Y')
        for point in points:
            if point.group >= len(color_store):
                color = color_store[-1]
            else:
                color = color_store[point.group]
            pylab.plot(point.x, point.y, color)
        for single_trace in cluster_center_trace:
            pylab.plot([center.x for center in single_trace], [center.y for center in single_trace], 'k')
        pylab.show()


if __name__ == '__main__':
    unittest.main()
