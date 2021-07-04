import unittest

from pandas import DataFrame
from sklearn import datasets
from sklearn.model_selection import train_test_split

from classification.naive_bayes.naive_bayes import NaiveBayes
from clustering.k_means.k_means import KMeans
from pio.output import Output


class XlsxOutputTest(unittest.TestCase):

    def test_clusterization_iris(self):
        iris = datasets.load_iris()
        iris_frame = DataFrame(iris.data)
        iris_frame.columns = iris.feature_names

        prediction = KMeans(n_clusters=3).run(iris_frame)

        iris_frame_rows = iris_frame.values
        rows = [iris_frame_rows[i] for i in range(len(iris_frame_rows))]
        dictionary = {}
        for i in range(max(prediction) + 1):
            indexes = [j for j, x in enumerate(prediction) if x == i]
            points = [rows[i] for i in indexes]
            dictionary['Кластер %d:' % (i + 1)] = points

        Output.write_to_xlsx_file('./ClusterizationResult.xlsx', iris_frame, dictionary)

    def test_classification_iris(self):
        iris = datasets.load_iris()
        iris_frame = DataFrame(iris.data)
        iris_frame.columns = iris.feature_names
        x_train, x_test, y_train, y_test = train_test_split(iris_frame, iris.target, test_size=0.10, random_state=1)

        prediction = NaiveBayes().run(x_train, y_train, x_test)
        print(y_test)
        print(prediction)

        iris_frame_rows = iris_frame.values
        rows = [iris_frame_rows[i] for i in range(len(iris_frame_rows))]
        dictionary = {}
        for i in range(max(prediction) + 1):
            indexes = [j for j, x in enumerate(prediction) if x == i]
            points = [rows[i] for i in indexes]
            dictionary['Класс %d:' % (i + 1)] = points

        Output.write_to_xlsx_file('./ClassificationResult.xlsx', iris_frame, dictionary)


if __name__ == '__main__':
    unittest.main()
