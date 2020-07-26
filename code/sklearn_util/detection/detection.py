import numpy as np
from sklearn.ensemble import IsolationForest
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
import sys
import os
sys.path.append(os.getcwd())


def test_lof(X, Y):
    clf = LocalOutlierFactor(n_neighbors=20, contamination=0.19)
    y_pred = clf.fit_predict(X)
    print(classification_report(Y, y_pred))


def test_ocs(X, Y, normal_vecs, abnormal_vecs):

    clf = OneClassSVM(kernel='linear', nu=0.002, degree=2)
    clf.fit(normal_vecs)
    y_pred = clf.predict(X)
    print(classification_report(Y, y_pred))


def test_iforest(X, Y):

    clf = IsolationForest(
        max_samples=5000, random_state=np.random.RandomState(42), contamination=0.19)
    clf.fit(X)
    y_pred = clf.predict(X)
    print(classification_report(Y, y_pred))
    print(y_pred)


def test():
    from data.util import load_data
    normal_data_set, abnormal_data = load_data()
    normal_data = normal_data_set
    data = []
    data.extend(abnormal_data)
    data.extend(normal_data)
    random.shuffle(data)
    commands = []
    labels = []
    for item in data:
        if item[0]:
            commands.append(item[0])
            labels.append(item[1])
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(commands)
    Y = [1 if item == 0 else -1 for item in labels]
    normal_vecs = vectorizer.transform([item[0] for item in normal_data_set])
    abnormal_vecs = vectorizer.transform([item[0] for item in abnormal_data])
    print("result for One class svm:")
    test_ocs(X, Y, normal_vecs, abnormal_vecs)
    print("result for iforest:")
    test_iforest(X, Y)
    print("result for LOF")
    test_lof(X, Y)


if __name__ == '__main__':
    test()
