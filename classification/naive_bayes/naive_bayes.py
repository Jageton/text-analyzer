from sklearn.naive_bayes import GaussianNB


class NaiveBayes:
    def __init__(self, priors: int = None, var_smoothing: float = 1e-9):
        self.priors = priors
        self.var_smoothing = var_smoothing

    def run(self, x_train, y_train, x_test):
        model = GaussianNB(priors=self.priors, var_smoothing=self.var_smoothing)
        model.fit(X=x_train, y=y_train)
        return model.predict(X=x_test)
