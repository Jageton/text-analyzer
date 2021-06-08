import copy
import random

FLOAT_MAX = 1e100


class Point:
    __slots__ = ["x", "y", "group", "membership"]

    def __init__(self, cluster_center_number, x=0, y=0, group=0):
        self.x, self.y, self.group = x, y, group
        self.membership = [0.0 for _ in range(cluster_center_number)]


class FCMImpl:

    def __init__(self, points, cluster_center_number, weight):
        self.points = points
        self.cluster_center_number = cluster_center_number
        self.weight = weight

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
        return

    def run(self):
        cluster_center_group = [Point(self.cluster_center_number) for _ in range(self.cluster_center_number)]
        self._k_means_plus_plus(self.points, cluster_center_group)
        cluster_center_trace = [[clusterCenter] for clusterCenter in cluster_center_group]
        tolerable_error, current_error = 1.0, FLOAT_MAX
        while current_error >= tolerable_error:
            for point in self.points:
                self._get_single_membership(point, cluster_center_group, self.weight)
            current_center_group = [Point(self.cluster_center_number) for _ in range(self.cluster_center_number)]
            for centerIndex, center in enumerate(current_center_group):
                upper_sum_x, upper_sum_y, lower_sum = 0.0, 0.0, 0.0
                for point in self.points:
                    membership_weight = pow(point.membership[centerIndex], self.weight)
                    upper_sum_x += point.x * membership_weight
                    upper_sum_y += point.y * membership_weight
                    lower_sum += membership_weight
                center.x = upper_sum_x / lower_sum
                center.y = upper_sum_y / lower_sum
            # update cluster center trace
            current_error = 0.0
            for index, singleTrace in enumerate(cluster_center_trace):
                singleTrace.append(current_center_group[index])
                current_error += self._solve_distance_between_points(singleTrace[-1], singleTrace[-2])
                cluster_center_group[index] = copy.copy(current_center_group[index])
        for point in self.points:
            max_index, max_membership = 0, 0.0
            for index, singleMembership in enumerate(point.membership):
                if singleMembership > max_membership:
                    max_membership = singleMembership
                    max_index = index
            point.group = max_index
        return cluster_center_group, cluster_center_trace

    def _get_single_membership(self, point, cluster_center_group, weight):
        distance_from_point2_cluster_center_group = [
            self._solve_distance_between_points(point, cluster_center_group[index]) for
            index in
            range(len(cluster_center_group))]
        for center_index, single_membership in enumerate(point.membership):
            sum = 0.0
            is_coincide = [False, 0]
            for index, distance in enumerate(distance_from_point2_cluster_center_group):
                if distance == 0:
                    is_coincide[0] = True
                    is_coincide[1] = index
                    break
                sum += pow(float(distance_from_point2_cluster_center_group[center_index] / distance),
                           1.0 / (weight - 1.0))
            if is_coincide[0]:
                if is_coincide[1] == center_index:
                    point.membership[center_index] = 1.0
                else:
                    point.membership[center_index] = 0.0
            else:
                point.membership[center_index] = 1.0 / sum
