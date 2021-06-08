import math
import random
import unittest

import pylab
from sklearn.datasets import make_blobs

from clustering.fcm.fcm_impl import FCMImpl, Point


class FCMImplTest(unittest.TestCase):

    def test_1000_random_points(self):
        # Configuration options
        num_samples_total = 1000
        weight = 2
        cluster_number = 2
        cluster_centers = [(20, 20), (4, 4)]

        # Generate data
        X, targets = make_blobs(n_samples=num_samples_total, centers=cluster_centers, n_features=cluster_number,
                                center_box=(0, 1), cluster_std=2)
        points = list(map(lambda x: Point(cluster_number, x[0], x[1]), X))
        _, cluster_center_trace = FCMImpl(points, cluster_number, weight).run()

        self.show_cluster_analysis_results(points, [])

        # Asserts:
        first_cluster_cnt = list(map(lambda x: x.group, points)).count(0)
        second_cluster_cnt = list(map(lambda x: x.group, points)).count(1)
        self.assertEqual(first_cluster_cnt, second_cluster_cnt)

    def test_6_known_points(self):
        # Generate data
        cluster_number = 2
        weight = 2
        points = [Point(cluster_number, 1, 2), Point(cluster_number, 1, 4), Point(cluster_number, 1, 0),
                  Point(cluster_number, 10, 2), Point(cluster_number, 10, 4), Point(cluster_number, 10, 0)]
        _, cluster_center_trace = FCMImpl(points, cluster_number, weight).run()

        # Asserts:
        first_cluster_cnt = list(map(lambda x: x.group, points)).count(0)
        second_cluster_cnt = list(map(lambda x: x.group, points)).count(1)
        self.assertEqual(first_cluster_cnt, second_cluster_cnt)

    def test_clustering_visualization(self):
        cluster_number = 5
        points_number = 2000
        radius = 10
        weight = 2
        points = self.generate_points(points_number, radius, cluster_number)

        _, cluster_center_trace = FCMImpl(points, cluster_number, weight).run()
        self.show_cluster_analysis_results(points, cluster_center_trace)

    def generate_points(self, points_number, radius, cluster_center_number):
        points = [Point(cluster_center_number) for _ in range(2 * points_number)]
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
        colorStore = ['or', 'og', 'ob', 'oc', 'om', 'oy', 'ok']
        pylab.figure(figsize=(9, 9), dpi=80)
        pylab.title('Clusterization result')
        pylab.xlabel('X')
        pylab.ylabel('Y')
        for point in points:
            if point.group >= len(colorStore):
                color = colorStore[-1]
            else:
                color = colorStore[point.group]
            pylab.plot(point.x, point.y, color)
        for single_trace in cluster_center_trace:
            pylab.plot([center.x for center in single_trace], [center.y for center in single_trace], 'k')
        pylab.show()


if __name__ == '__main__':
    unittest.main()
