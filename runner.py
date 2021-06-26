import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from sklearn import datasets
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pylab

from classification.decision_tree.decision_tree import DecisionTree
from classification.knn.k_nearest_neighbors import KNN
from classification.naive_bayes.naive_bayes import NaiveBayes
from clustering.agglomerative.agglomerative import Aglomerative
from clustering.birch.birch import Birch
from clustering.dbscan.dbscan import DBSCAN
from clustering.fcm.fcm_impl import FCMImpl, Point
from clustering.k_means.k_means import KMeans
from clustering.spectral_clustering.spectral_clustering import SpectralCluster

colors = ['#47a8f2', '#3b4cc0', '#b50525', '#000000']


def get_color(x):
    if 0 > x or x > 2:
        return colors[3]
    return colors[x]


def _get_colors(predict):
    return list(map(get_color, predict))


def run_algorithm_for_2_columns(algorithm, dataframe):
    predict = algorithm()
    plt.scatter(dataframe.values[:, 0], dataframe.values[:, 1], c=_get_colors(
        predict), marker="o", picker=True)
    plt.title('Clusterization result')
    plt.xlabel(dataframe.columns[0])
    plt.ylabel(dataframe.columns[1])
    plt.show()


def run_algorithm_for_3_columns(algorithm, dataframe):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    predict = algorithm()
<<<<<<< Updated upstream
    ax.scatter(dataframe.values[:, 0], dataframe.values[:, 1], dataframe.values[:, 2], c=_get_colors(predict))
    ax.set_xlabel(dataframe.columns[0])
    ax.set_ylabel(dataframe.columns[1])
    ax.set_zlabel(dataframe.columns[2])
=======
    ax.scatter(dataframe.values[:, 0], dataframe.values[:, 1],
               dataframe.values[:, 2], c=_get_colors(predict))
>>>>>>> Stashed changes
    plt.show()


def run_db_scan(nsamples, dataframe, eps, algorithm, metric, leaf_size, p):
    # cluster_std = params['cluster_std']
    # print("////////////////////  DB_SCAN WITH nsamples =", nsamples, ", cluster_std=", cluster_std, " ///////")
    predict = None
    print(dataframe, nsamples)
    if dataframe is None:
        centers = [[1, 1], [-5, -5], [5, -5]]
        X, labels_true = make_blobs(
            n_samples=nsamples, centers=centers, cluster_std=1, random_state=0)
        X = StandardScaler().fit_transform(X)
        predict = DBSCAN(eps=eps, metric=metric,
                         algorithm=algorithm, p=p, leaf_size=leaf_size).run(X)
    else:
        predict = DBSCAN(eps=eps, metric=metric, algorithm=algorithm,
                         p=p, leaf_size=leaf_size).run(dataframe)
        # X = dataframe
    print(predict)
    dataframe.columns[-1].replace(" ", "").split(',')

    # # Plot result
    # core_samples_mask = np.zeros_like(predict, dtype=bool)
    # core_samples_mask[range(0, nsamples)] = True
    #
    # # Black removed and is used for noise instead.
    # unique_labels = set(predict)
    # colors = [plt.cm.Spectral(each)
    #           for each in np.linspace(0, 1, len(unique_labels))]
    # for k, col in zip(unique_labels, colors):
    #     if k == -1:
    #         # Black used for noise.
    #         col = [0, 0, 0, 1]
    #
    #     class_member_mask = (predict == k)
    #
    #     xy = X[class_member_mask & core_samples_mask]
    #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14)
    #
    #     xy = X[class_member_mask & ~core_samples_mask]
    #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)
    #
    # plt.title('Estimated number of clusters: %d' % 3)
    # plt.show()


def run_kmeans(nsamples, dataframe, n_clusters, init, algorithm, n_init, max_iter):
    # Configuration options

    # nsamples = params['nsamples']
    # cluster_std = params['cluster_std']
    print("////////////////////  KMEANS WITH nsamples =", nsamples,
          ", n_clusters =", n_clusters,
          ", init =", init,
          ", algorithm =", algorithm,
          ", n_init =", n_init,
          ", max_iter=", max_iter, " /////////")
    predict = None
    print(dataframe, nsamples)
    if dataframe is None:
        centers = [[1, 1], [-5, -5], [5, -5]]
        X, labels_true = make_blobs(
            n_samples=nsamples, centers=centers, cluster_std=1, random_state=0)
        predict = KMeans(n_clusters=n_clusters,
                         init=init,
                         algorithm=algorithm,
                         n_init=n_init,
                         max_iter=max_iter).run(X)
    else:
        predict = KMeans(n_clusters=n_clusters,
                         init=init,
                         algorithm=algorithm,
                         n_init=n_init,
                         max_iter=max_iter).run(dataframe)
    print(predict)
    # Generate data
    X, targets = make_blobs(n_samples=nsamples, centers=cluster_centers, n_features=num_classes,
                            center_box=(0, 1), cluster_std=cluster_std)
    predict = KMeans(n_clusters=num_classes, verbose=True).run(X)

    # Generate scatter plot for training data
    # colors = list(map(lambda x: '#3b4cc0' if x == 1 else '#b40426', predict))
    # plt.scatter(X[:, 0], X[:, 1], c=colors, marker="o", picker=True)
    # plt.title('Clusterization result')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.show()


def run_spectral_clustering(nsamples,dataframe,n_clusters,eigen_solver):

    print("////////////////////  SPEACTRAL_CLUSTERING WITH nsamples =", nsamples,
    ", n_clusters =",n_clusters,
    ", eigen_solver =",eigen_solver, " ////////")
    predict = None
    print(dataframe, nsamples)
    if dataframe is None:
        centers = [[1, 1], [-5, -5], [5, -5]]
        X, labels_true = make_blobs(
            n_samples=nsamples, centers=centers, cluster_std=1, random_state=0)
        predict = SpectralCluster(n_clusters=n_clusters,
        eigen_solver=eigen_solver ).run(X)
    else:
        predict = SpectralCluster(n_clusters=n_clusters,
        eigen_solver=eigen_solver ).run(dataframe)
    print(predict)
    # Generate data
    # Generate scatter plot for training data
    # colors = list(map(lambda x: '#3b4cc0' if x == 1 else '#b40426', predict))
    # plt.scatter(X[:, 0], X[:, 1], c=colors, marker="o", picker=True)
    # plt.title('Clusterization result')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.show()


def run_fcm(params):
    nsamples = params['nsamples']
    cluster_std = params['cluster_std']
    print("////////////////////  FCM WITH nsamples =",
          nsamples, ", cluster_std=", cluster_std, " ////////")
    weight = 2
    cluster_centers = [(20, 20), (4, 4)]
    cluster_number = len(cluster_centers)

    # Generate data
    X, targets = make_blobs(n_samples=nsamples, centers=cluster_centers, n_features=cluster_number,
                            center_box=(0, 1), cluster_std=cluster_std)
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
        pylab.plot([center.x for center in single_trace], [
                   center.y for center in single_trace], 'k')
    pylab.show()


def run_birch(params):
    nsamples = params['nsamples']
    cluster_std = params['cluster_std']
    branching_factor = params['branching_factor']
    threeshold = params['threeshold']
    print("////////////////////  BIRCH WITH nsamples =",
          nsamples, ", cluster_std=", cluster_std, " ///////")
    X, clusters = make_blobs(
        n_samples=nsamples, centers=6, cluster_std=cluster_std, random_state=0)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.7, edgecolors='b')
    plt.show()

    predict = Birch(branching_factor=branching_factor,
                    n_clusters=None, threshold=threeshold).run(X)

    plt.scatter(X[:, 0], X[:, 1], c=predict,
                cmap='rainbow', alpha=0.7, edgecolors='b')
    plt.show()


def run_agglomerative(nsamples, dataframe, linkage, n_clusters, affinity):

    # nsamples = params['nsamples']
    # cluster_std = params['cluster_std']
    # print("////////////////////  AGGLOMERATIVE WITH nsamples =",
    #       nsamples, ", cluster_std=", cluster_std, " ///////")
    # Configuration options
    predict = None
    print(dataframe, nsamples)
    if(dataframe == None):
        centers = [[1, 1], [-5, -5], [5, -5]]
        # Generate data
        X, targets = make_blobs(
            n_samples=nsamples, centers=centers, cluster_std=1, random_state=0)
        X = StandardScaler().fit_transform(X)
        predict = Aglomerative(
            linkage=linkage, n_clusters=n_clusters, affinity=affinity).run(X)
    else:
        predict = Aglomerative(
            linkage=linkage, n_clusters=n_clusters, affinity=affinity).run(dataframe)
    print(predict)

    # Generate scatter plot for training data
    # colors = list(map(lambda x: '#3b4cc0' if x == 1 else '#b40426', predict))
    # plt.scatter(X[:, 0], X[:, 1], c=colors, marker="o", picker=True)
    # plt.title('Clusterization result')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.show()


def run_decision_tree(test_size, criterion, splitter):
    print("////////////////////  DECISION_TREE WITH test_size =", test_size,
          ", criterion =", criterion, ", splitter =", splitter, "   ///////")
    iris = datasets.load_iris()
    iris_frame = DataFrame(iris.data)
    iris_frame.columns = iris.feature_names
    iris_frame['target'] = iris.target

    x = iris_frame.drop(columns=['target'])
    y = iris_frame['target'].values
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=1)
    predict = DecisionTree(criterion=criterion, splitter=splitter).run(
        train_x=x_train, train_y=y_train, test_x=x_test)
    print(y_test)
    print(predict)


def run_kneares_neighbors(n_neighbords, algorithm, weight, test_size):
    print("////////////////////  KNEAREST_NEIGHBORS WITH  n_neighbords =", n_neighbords,
          "algorithm =", algorithm,
          "weight =", weight,
          "test_size =", test_size, " ///////")
    iris = datasets.load_iris()
    iris_frame = DataFrame(iris.data)
    iris_frame.columns = iris.feature_names
    iris_frame['target'] = iris.target

    x = iris_frame.drop(columns=['target'])
    y = iris_frame['target'].values
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=1)
    predict = KNN(n_neighbors=n_neighbords,
                  algorithm=algorithm,
                  weights=weight).run(x_train, y_train, x_test)
    print(y_test)
    print(predict)


def run_naive_bayes(test_size):
    print("////////////////////  NAIVE_BAYES WITH test_size =",
          test_size, " ///////")
    iris = datasets.load_iris()
    iris_frame = DataFrame(iris.data)
    iris_frame.columns = iris.feature_names
    iris_frame['target'] = iris.target

    x = iris_frame.drop(columns=['target'])
    y = iris_frame['target'].values
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=1)
    predict = NaiveBayes().run(x_train, y_train, x_test)
    print(y_test)
    print(predict)
