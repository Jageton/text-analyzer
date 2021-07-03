import unittest

from pandas import DataFrame
from sklearn import datasets

from clustering.k_means.k_means import KMeans
from pio.output import Output


class XlsxOutputTest(unittest.TestCase):

    def test_iris(self):
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


if __name__ == '__main__':
    unittest.main()
