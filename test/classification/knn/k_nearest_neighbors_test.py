from classification.knn.k_nearest_neighbors import KNN
import unittest
from pandas.core.frame import DataFrame
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
class KNearestNeighborsTest(unittest.TestCase):

    def test_alg(self):

        iris = datasets.load_iris()
        iris_frame = DataFrame(iris.data)
        iris_frame.columns = iris.feature_names
        iris_frame['target'] = iris.target
        iris_frame['name'] = iris_frame.target.apply(
            lambda x: iris.target_names[x])
        iris_frame = iris_frame.drop(columns=["name"])

        X = iris_frame.drop(columns=['target'])
        Y = iris_frame.values
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.25, random_state=1)
        classifier=KNN(n_neighbors=10)
        predict=classifier.run(X_train,Y_train,X_test)
        self.assertEqual(np.array_equal(Y_test,predict),True)


if __name__ == '__main__':
    unittest.main()
