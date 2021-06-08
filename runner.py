import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from sklearn import datasets
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import pylab

from clustering.birch.birch import Birch
from clustering.dbscan.dbscan import DBSCAN
from clustering.fcm.fcm_impl import FCMImpl, Point
from clustering.k_means.k_means import KMeans
from clustering.spectral_clustering.spectral_clustering import SpectralCluster


def run_db_scan():
    centers = [[1, 1], [-5, -5], [5, -5]]
    X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4, random_state=0)
    X = StandardScaler().fit_transform(X)

    predict = DBSCAN(eps=0.3).run(X)
    predict_list = list(predict)

    # Asserts:
    first_cluster_cnt = predict_list.count(0)
    second_cluster_cnt = predict_list.count(1)
    third_cluster_cnt = predict_list.count(2)

    # Plot result
    core_samples_mask = np.zeros_like(predict, dtype=bool)
    core_samples_mask[range(0, 750)] = True

    # Black removed and is used for noise instead.
    unique_labels = set(predict)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (predict == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % 3)
    plt.show()

def run_kmeans():
    # Configuration options
    num_samples_total = 1000
    cluster_centers = [(20, 20), (4, 4)]
    num_classes = len(cluster_centers)

    # Generate data
    X, targets = make_blobs(n_samples=num_samples_total, centers=cluster_centers, n_features=num_classes,
                            center_box=(0, 1), cluster_std=2)
    predict = KMeans(n_clusters=num_classes).run(X)

    # Generate scatter plot for training data
    colors = list(map(lambda x: '#3b4cc0' if x == 1 else '#b40426', predict))
    plt.scatter(X[:, 0], X[:, 1], c=colors, marker="o", picker=True)
    plt.title('Clusterization result')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def run_spectral_clustering():
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

def run_fcm():
    num_samples_total = 1000
    weight = 2
    cluster_number = 2
    cluster_centers = [(20, 20), (4, 4)]

    # Generate data
    X, targets = make_blobs(n_samples=num_samples_total, centers=cluster_centers, n_features=cluster_number,
                            center_box=(0, 1), cluster_std=2)
    points = list(map(lambda x: Point(cluster_number, x[0], x[1]), X))
    _, cluster_center_trace = FCMImpl(points, cluster_number, weight).run()

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

def run_birch():
    X, clusters = make_blobs(n_samples=450, centers=6, cluster_std=0.70, random_state=0)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.7, edgecolors='b')
    plt.show()

    predict = Birch(branching_factor=50, n_clusters=None, threshold=1.5).run(X)

    plt.scatter(X[:, 0], X[:, 1], c=predict, cmap='rainbow', alpha=0.7, edgecolors='b')
    plt.show(block=True)
