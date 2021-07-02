from pandas import DataFrame
from sklearn import datasets
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from classification.decision_tree.decision_tree import DecisionTree
from classification.knn.k_nearest_neighbors import KNN
from classification.naive_bayes.naive_bayes import NaiveBayes
from classification.random_forests.random_forests import RandomForest
from clustering.agglomerative_clustering.agglomerative_clustering import AgglomerativeClustering
from clustering.birch.birch import Birch
from clustering.dbscan.dbscan import DBSCAN
from clustering.k_means.k_means import KMeans
from clustering.mean_shift.mean_shift import MeanShift
from clustering.spectral_clustering.spectral_clustering import SpectralCluster
from util.visualization import data_2d_visualization


def _prepare_iris_data_for_classification(test_size, random_state):
    iris = datasets.load_iris()
    iris_frame = DataFrame(iris.data)
    iris_frame.columns = iris.feature_names
    iris_frame['target'] = iris.target

    x = iris_frame.drop(columns=['target'])
    y = iris_frame['target'].values

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test


def run_db_scan(nsamples, eps, algorithm, metric, leaf_size, p):
    # print("////////////////////  DB_SCAN WITH nsamples =", nsamples, ", cluster_std=", cluster_std, " ///////")
    print(nsamples)
    centers = [[1, 1], [-5, -5], [5, -5]]
    data, labels_true = make_blobs(
        n_samples=nsamples, centers=centers, cluster_std=1, random_state=0)
    data = StandardScaler().fit_transform(data)

    predict = DBSCAN(eps=eps, metric=metric, algorithm=algorithm,
                     p=p, leaf_size=leaf_size).run(data)
    print(predict)
    data_2d_visualization(data, predict)


def run_kmeans(nsamples, n_clusters, init, algorithm, n_init, max_iter):
    print("////////////////////  K-MEANS WITH nsamples =", nsamples,
          ", n_clusters =", n_clusters,
          ", init =", init,
          ", algorithm =", algorithm,
          ", n_init =", n_init,
          ", max_iter=", max_iter, " /////////")
    centers = [[1, 1], [-5, -5], [5, -5]]
    data, labels_true = make_blobs(
        n_samples=nsamples, centers=centers, cluster_std=1, random_state=0)

    predict = KMeans(n_clusters=n_clusters, init=init, algorithm=algorithm,
                     n_init=n_init, max_iter=max_iter).run(data)
    print(predict)
    data_2d_visualization(data, predict)


def run_spectral_clustering(nsamples, n_clusters, eigen_solver):
    print("////////////////////  SPECTRAL_CLUSTERING WITH nsamples =", nsamples,
          ", n_clusters =", n_clusters,
          ", eigen_solver =", eigen_solver, " ////////")
    centers = [[1, 1], [-5, -5], [5, -5]]
    data, labels_true = make_blobs(n_samples=nsamples, centers=centers, cluster_std=1, random_state=0)
    predict = SpectralCluster(n_clusters=n_clusters, eigen_solver=eigen_solver).run(data)
    print(predict)
    data_2d_visualization(data, predict)


def run_birch(nsamples, threshold, branching_factor, n_clusters):
    data, clusters = make_blobs(n_samples=nsamples, centers=6, cluster_std=2, random_state=0)
    predict = Birch(branching_factor=branching_factor,
                    n_clusters=n_clusters, threshold=threshold).run(data)
    data_2d_visualization(data, predict)


def run_agglomerative(nsamples, linkage, n_clusters, affinity):
    # print("////////////////////  AGGLOMERATIVE WITH nsamples =",
    #       nsamples, ", cluster_std=", cluster_std, " ///////")
    print(nsamples)
    centers = [[1, 1], [-5, -5], [5, -5]]
    data, targets = make_blobs(n_samples=nsamples, centers=centers, cluster_std=1, random_state=0)
    data = StandardScaler().fit_transform(data)

    predict = AgglomerativeClustering(
        linkage=linkage, n_clusters=n_clusters, affinity=affinity).run(data)
    print(predict)
    data_2d_visualization(data, predict)


def run_decision_tree(test_size, criterion, splitter):
    print("////////////////////  DECISION_TREE WITH test_size =", test_size,
          ", criterion =", criterion, ", splitter =", splitter, "   ///////")
    x_train, x_test, y_train, y_test = \
        _prepare_iris_data_for_classification(test_size=test_size, random_state=1)
    predict = DecisionTree(criterion=criterion, splitter=splitter).run(
        train_x=x_train, train_y=y_train, test_x=x_test)
    print(y_test)
    print(predict)


def run_mean_shift(max_iter, bin_seeding, cluster_all, nsamples):
    print("////////////////////  MEAN_SHIFT WITH nsamples =", nsamples,
          ", max_iter =", max_iter,
          ", bin_seeding =", bin_seeding,
          ", cluster_all =", cluster_all, " ////////")
    centers = [[1, 1], [-5, -5], [5, -5]]
    data, labels_true = make_blobs(n_samples=nsamples, centers=centers, cluster_std=1, random_state=0)
    predict = MeanShift(max_iter=max_iter, bin_seeding=bin_seeding, cluster_all=cluster_all).run(data)
    print(predict)
    data_2d_visualization(data, predict)


def run_knearest_neighbors(n_neighbors, algorithm, weight, test_size):
    print("////////////////////  KNEAREST_NEIGHBORS WITH  n_neighbors =", n_neighbors,
          "algorithm =", algorithm,
          "weight =", weight,
          "test_size =", test_size, " ///////")
    x_train, x_test, y_train, y_test = \
        _prepare_iris_data_for_classification(test_size=test_size, random_state=1)
    predict = KNN(n_neighbors=n_neighbors,
                  algorithm=algorithm,
                  weights=weight).run(x_train, y_train, x_test)
    print(y_test)
    print(predict)


def run_random_forest(n_estimators, criterion, verbose, random_state, test_size):
    print("////////////////////  RANDOM_FOREST WITH n_estimators =", n_estimators,
          ", criterion =", criterion,
          ", verbose =", verbose,
          ", random_state =", random_state,
          ", test_size", test_size, " ///////")
    x_train, x_test, y_train, y_test = \
        _prepare_iris_data_for_classification(test_size=test_size, random_state=1)
    predict = RandomForest(n_estimators=n_estimators, criterion=criterion,
                           verbose=verbose).run(x_train, y_train, x_test)
    print(y_test)
    print(predict)


def run_naive_bayes(test_size):
    print("////////////////////  NAIVE_BAYES WITH test_size =",
          test_size, " ///////")
    x_train, x_test, y_train, y_test = \
        _prepare_iris_data_for_classification(test_size=test_size, random_state=1)
    predict = NaiveBayes().run(x_train, y_train, x_test)
    print(y_test)
    print(predict)
