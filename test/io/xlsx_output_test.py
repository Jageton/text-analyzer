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

        alg = KMeans(n_clusters=3)
        prediction = alg.run(iris_frame)

        iris_frame_rows = iris_frame.values
        rows = [iris_frame_rows[i] for i in range(len(iris_frame_rows))]
        dictionary = {}
        for i in range(max(prediction) + 1):
            indexes = [j for j, x in enumerate(prediction) if x == i]
            points = {i + 1: rows[i] for i in indexes}
            dictionary['Кластер %d:' % (i + 1)] = points

        attributes = self.get_attributes(alg)
        attributes['alg_name'] = alg.__class__.__name__
        Output.write_to_xlsx_file('./ClusterizationResult.xlsx', iris_frame, dictionary, attributes)

    def test_classification_iris(self):
        iris = datasets.load_iris()
        iris_frame = DataFrame(iris.data)
        iris_frame.columns = iris.feature_names
        x_train, x_test, y_train, y_test = train_test_split(iris_frame, iris.target, test_size=0.10, random_state=1)

        alg = NaiveBayes()
        prediction = alg.run(x_train, y_train, x_test)
        print(y_test)
        print(prediction)

        iris_frame_rows = iris_frame.values
        rows = [iris_frame_rows[i] for i in range(len(iris_frame_rows))]
        dictionary = {}
        for i in range(max(prediction) + 1):
            indexes = [j for j, x in enumerate(prediction) if x == i]
            points = {i + 1: rows[i] for i in indexes}
            dictionary['Класс %d:' % (i + 1)] = points

        attributes = self.get_attributes(alg)
        attributes['alg_name'] = alg.__class__.__name__
        Output.write_to_xlsx_file('./ClassificationResult.xlsx', iris_frame, dictionary, attributes)

    @staticmethod
    def get_attributes(alg):
        return dict((k, v) for (k, v) in alg.__dict__.items() if not str(k).startswith('__') and v is not None)


if __name__ == '__main__':
    unittest.main()
