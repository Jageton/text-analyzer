from types import FunctionType
import numpy as np
from numpy.lib.function_base import copy
from pandas.core.frame import DataFrame
from sklearn import neighbors,datasets
from sklearn.model_selection import train_test_split


class KnnResult:
    def __init__(self,predictionResult,scopeResult,testData,classifier) -> None:
        self.predictionResult=predictionResult
        self.scope=scopeResult
        self.knnClassifier=classifier
        self.testData=testData
        pass

def knn(n_neighbors:int,weights:FunctionType,data:DataFrame,targetColumn='target',testSize=0.2)->KnnResult:
    
    X=data.drop(columns=[targetColumn])
    Y=data[targetColumn].values
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=testSize,random_state=1,stratify=Y)

    classifier=neighbors.KNeighborsClassifier(n_neighbors=n_neighbors,weights=weights)
    classifier.fit(X=X_train,y=Y_train)
    return KnnResult(predictionResult=classifier.predict(X=X_test),
        scopeResult=classifier.score(X=X_test,y=Y_test),
        classifier=classifier,testData=X_test)

def main():
    iris=datasets.load_iris()
    iris_frame=DataFrame(iris.data)
    iris_frame.columns=iris.feature_names
    iris_frame['target']=iris.target
    iris_frame['name']=iris_frame.target.apply(lambda x:iris.target_names[x])
    iris_frame=iris_frame.drop(columns=["name"])
    result=knn(10,'uniform',iris_frame,'target',0.25)

    print(result.scope)
    print(result.predictionResult)

main()