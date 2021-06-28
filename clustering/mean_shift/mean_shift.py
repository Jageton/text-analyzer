from sklearn.cluster import MeanShift as skMeanShift


class MeanShift:

    def __init__(self, bandwidth=None, seeds=None, bin_seeding=False,
                 min_bin_freq=1, cluster_all=True, n_jobs=None, max_iter=300):
        self.bandwidth = bandwidth
        self.seeds = seeds
        self.bin_seeding = bin_seeding
        self.cluster_all = cluster_all
        self.min_bin_freq = min_bin_freq
        self.n_jobs = n_jobs
        self.max_iter = max_iter

    def run(self, x):
        model = skMeanShift(bandwidth=self.bandwidth, seeds=self.seeds, bin_seeding=self.bin_seeding,
                            min_bin_freq=self.min_bin_freq, cluster_all=self.cluster_all,
                            n_jobs=self.n_jobs, max_iter=self.max_iter)
        return model.fit_predict(X=x)
