import copy
import random

FLOAT_MAX = 1e100


class Point:
    __slots__ = ["x", "y", "group"]

    def __init__(self, x=0, y=0, group=0):
        self.x, self.y, self.group = x, y, group


class KMeansPlusPlusImpl:

    def __init__(self, points, cluster_center_number):
        self.points = points
        self.cluster_center_number = cluster_center_number

    def _solve_distance_between_points(self, point_a, point_b):
        return (point_a.x - point_b.x) * (point_a.x - point_b.x) + (point_a.y - point_b.y) * (point_a.y - point_b.y)

    def _get_nearest_center(self, point, cluster_center_group):
        min_index = point.group
        min_distance = FLOAT_MAX
        for index, center in enumerate(cluster_center_group):
            distance = self._solve_distance_between_points(point, center)
            if distance < min_distance:
                min_distance = distance
                min_index = index
        return min_index, min_distance

    def _k_means_plus_plus(self, points, cluster_center_group):
        cluster_center_group[0] = copy.copy(random.choice(points))
        distance_group = [0.0 for _ in range(len(points))]
        sum = 0.0
        for index in range(1, len(cluster_center_group)):
            for i, point in enumerate(points):
                distance_group[i] = self._get_nearest_center(point, cluster_center_group[:index])[1]
                sum += distance_group[i]
            sum *= random.random()
            for i, distance in enumerate(distance_group):
                sum -= distance
                if sum < 0:
                    cluster_center_group[index] = copy.copy(points[i])
                    break
        for point in points:
            point.group = self._get_nearest_center(point, cluster_center_group)[0]
        return

    def run(self):
        cluster_center_group = [Point() for _ in range(self.cluster_center_number)]
        self._k_means_plus_plus(self.points, cluster_center_group)
        cluster_center_trace = [[clusterCenter] for clusterCenter in cluster_center_group]
        tolerable_error, current_error = 5.0, FLOAT_MAX
        count = 0
        while current_error >= tolerable_error:
            count += 1
            count_center_number = [0 for _ in range(self.cluster_center_number)]
            current_center_group = [Point() for _ in range(self.cluster_center_number)]
            for point in self.points:
                current_center_group[point.group].x += point.x
                current_center_group[point.group].y += point.y
                count_center_number[point.group] += 1
            for index, center in enumerate(current_center_group):
                center.x /= count_center_number[index]
                center.y /= count_center_number[index]
            current_error = 0.0
            for index, singleTrace in enumerate(cluster_center_trace):
                singleTrace.append(current_center_group[index])
                current_error += self._solve_distance_between_points(singleTrace[-1], singleTrace[-2])
                cluster_center_group[index] = copy.copy(current_center_group[index])
            for point in self.points:
                point.group = self._get_nearest_center(point, cluster_center_group)[0]
        return cluster_center_group, cluster_center_trace
