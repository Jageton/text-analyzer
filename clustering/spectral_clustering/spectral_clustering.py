from enum import Enum
from sklearn.cluster import SpectralClustering


class EigenSolver(Enum):
    arpack = "arpack"
    lpbpcg = "lpbpcg"
    amg = "amg"


class AssignLabels(Enum):
    kmeans = "kmeans"
    discretize = "discretize"


class Affinity(Enum):
    nearest_neighbors = "nearest_neighbors"
    rbf = "rbf"
    precomputed = "precomputed"
    precomputed_nearest_neighbors = "precomputed_nearest_neighbors"


class SpectralCluster:
    def __init__(self, n_clusters: int = 8, eigen_solver: EigenSolver = None, n_components: int = None,
                 random_state=None, n_init: int = 10, gamma: float = 1.0, affinity=Affinity.rbf.value,
                 n_neighbors: int = 10, eigen_tol: float = 0.0, assign_labels: AssignLabels = AssignLabels.kmeans.value,
                 degree: float = 0.0, coef0: float = 1, kernel_params=None, n_jobs: int = None, verbose: bool = False):

        self.n_clusters = n_clusters
        self.eigen_solver = eigen_solver
        self.n_components = n_clusters if n_components == None else n_components
        self.random_state = random_state
        self.n_init = n_init
        self.gamma = gamma
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.eigen_tol = eigen_tol
        self.assign_labels = assign_labels
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.n_jobs = n_jobs
        self.verbose = verbose

    def run(self, X):
        return SpectralClustering(n_clusters=self.n_clusters, eigen_solver=self.eigen_solver, n_components=self.n_components,
                                  random_state=self.random_state, n_init=self.n_init, gamma=self.gamma, affinity=self.affinity,
                                  n_neighbors=self.n_neighbors, eigen_tol=self.eigen_tol, assign_labels=self.assign_labels,
                                  degree=self.degree, coef0=self.coef0, kernel_params=self.kernel_params,
                                  n_jobs=self.n_jobs, verbose=self.verbose).fit_predict(X=X)
