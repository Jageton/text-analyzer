import unittest
from classification.decision_tree.decision_tree import DecisionTree
from pandas.core.frame import DataFrame
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np


class DecisionTreeTest(unittest.TestCase):

    def test_iris(self):
        iris = datasets.load_iris()
        iris_frame = DataFrame(iris.data)
        iris_frame.columns = iris.feature_names
        iris_frame['target'] = iris.target

        x = iris_frame.drop(columns=['target'])
        y = iris_frame['target'].values
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.25, random_state=1)
        predict = DecisionTree().run(train_x=x_train,
                                     train_y=y_train,
                                     test_x=x_test)
        print(y_test)
        print(predict)
        pass


if __name__ == '__main__':
    unittest.main()
# [0 1 1 0 2 1 2 0 0 2 1 0 2 1 1 0 1 1 0 0 1 1 1 0 2 1 0 0 1 2 1 2 1 2 2 0 1 0]
# [0 1 1 0 2 1 2 0 0 2 1 0 2 1 1 0 1 1 0 0 1 1 2 0 2 1 0 0 1 2 1 2 1 2 2 0 1 0]
