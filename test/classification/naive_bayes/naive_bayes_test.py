import unittest

import numpy as np
from pandas.core.frame import DataFrame
from sklearn import datasets
from sklearn.model_selection import train_test_split

from classification.naive_bayes.naive_bayes import NaiveBayes


class NaiveBayesTest(unittest.TestCase):

    def test_alg(self):
        iris = datasets.load_iris()
        iris_frame = DataFrame(iris.data)
        iris_frame.columns = iris.feature_names
        iris_frame['target'] = iris.target

        x = iris_frame.drop(columns=['target'])
        y = iris_frame['target'].values
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=1)
        predict = NaiveBayes().run(x_train, y_train, x_test)
        print(y_test)
        print(predict)
        self.assertEqual(np.array_equal(y_test, predict), True)


if __name__ == '__main__':
    unittest.main()
