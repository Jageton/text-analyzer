from enum import Enum

from sklearn.ensemble import RandomForestClassifier


class RandomForestCriterion(Enum):
    gini = 'gini'
    entropy = 'entropy'


class RandomForestMaxFeatures(Enum):
    auto = 'auto'
    sqrt = 'sqrt'
    log2 = 'log2'


class RandomForest:

    def __init__(self, n_estimators=100, criterion=RandomForestCriterion.gini.value, max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_features=RandomForestMaxFeatures.auto.value, max_leaf_nodes=None, min_impurity_decrease=0.,
                 bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0,
                 warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.max_samples = max_samples

    def run(self, x_train, y_train, x_test):
        model = RandomForestClassifier(
            n_estimators=self.n_estimators, criterion=self.criterion, max_depth=self.max_depth,
            min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf, max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes, min_impurity_decrease=self.min_impurity_decrease,
            bootstrap=self.bootstrap, oob_score=self.oob_score, n_jobs=self.n_jobs, random_state=self.random_state,
            verbose=self.verbose, warm_start=self.warm_start, class_weight=self.class_weight,
            ccp_alpha=self.ccp_alpha, max_samples=self.max_samples)
        model.fit(x_train, y_train)
        return model.predict(x_test)
