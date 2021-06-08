from enum import Enum

from sklearn.tree import DecisionTreeClassifier


class DecisionTreeCriterion(Enum):
    gini = "gini"
    entropy = "entropy"


class DecisionTreeSplitter(Enum):
    best = "best"
    random = "random"


class DecisionTreeMaxFeatures(Enum):
    auto = "auto"
    sqrt = "sqrt"
    log2 = "log2"


class DecisionTree:
    def __init__(self, criterion=DecisionTreeCriterion.gini.value, splitter=DecisionTreeSplitter.best.value,
                 max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                 max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
                 class_weight=None, ccp_alpha=0.0):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_feature = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha

    def run(self, train_x, train_y, test_x):
        clf = DecisionTreeClassifier(criterion=self.criterion, splitter=self.splitter,
                                     max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                     min_samples_leaf=self.min_samples_leaf,
                                     min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                     max_features=self.max_feature, random_state=self.random_state,
                                     max_leaf_nodes=self.max_leaf_nodes,
                                     min_impurity_decrease=self.min_impurity_decrease, class_weight=self.class_weight,
                                     ccp_alpha=self.ccp_alpha)
        return clf.fit(train_x, train_y).predict(test_x)
