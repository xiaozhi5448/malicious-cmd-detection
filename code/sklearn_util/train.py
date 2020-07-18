#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib
from sklearn import svm
import glob
import os, sys
import json
import logging
from datetime import datetime
import random
import pickle
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit, cross_val_score, KFold
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import warnings
import settings
from data.util import load_data, get_dataset
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(filename)s %(lineno)d: %(levelname)s  %(message)s')
data_dirs = ['data/normal_all', 'data/abnormal_all']
meta_data_dir = settings.meta_data_dir
dataset_bin = settings.dataset_bin

plt.figure(figsize=(10, 20))


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.title(title)
    if ylim:
        plt.ylim(*ylim)
    plt.xlabel("Train example")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_score_mean = np.mean(train_scores, axis=1)
    train_score_std = np.std(train_scores, axis=1)
    test_score_mean = np.mean(test_scores, axis=1)
    test_score_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_score_mean-train_score_std, train_score_mean+train_score_std, alpha=0.1, color='r')

    plt.fill_between(train_sizes, test_score_mean-test_score_std, test_score_mean+test_score_std, alpha=0.1, color='g')

    plt.plot(train_sizes, train_score_mean, 'o-', color='r', label="training score")
    plt.plot(train_sizes, test_score_mean, 'o-', color='g', label="cross-validate score")
    plt.legend(loc="best")
    return plt


def test_knn(X, Y):
    clfs = list()
    clfs.append(('KNN', KNeighborsClassifier(n_neighbors=3)))
    clfs.append(('RadiusKNN', RadiusNeighborsClassifier(n_neighbors=3, radius=500)))
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    for clf in clfs:
        t1 = datetime.now()
        logging.info("training model {} started at {}".format(clf[0], t1))
        model = clf[1]
        model.fit(X_train, Y_train)
        # joblib.dump(model, "meta_data/{}.m".format(clf[0]))
        t2 = datetime.now()
        logging.info("training model {} end at {}, cost {} seconds".format(clf[0], t2, (t2 - t1).seconds))
        score = model.score(X_test, Y_test)
        logging.info("{} score: {}".format(clf[0], score))
        Y_pred = model.predict(X_test)
        print("report for {}".format(clf[0]))
        print(classification_report(Y_test, Y_pred))

def test_svm(X, Y):

    logging.info("svm result:")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    clf = GridSearchCV(svm.SVC(), cv=5, param_grid=[
        {
            'kernel':['rbf'],
            'gamma': np.linspace(0, 1, 50)
        },
        {
            'kernel':['poly'],
            'degree': list(range(1, 10))
        }
    ]
    )
    clf = svm.SVC(C=1.0, kernel='rbf', gamma=0.5)
    clf.fit(X, Y)
    # logging.info("best svm param: {}\nbest score: {}".format(clf.best_params_, clf.best_score_))
    # estimator = clf.best_estimator_
    y_pred = clf.predict(X_test)
    logging.info("svm score: {}".format(clf.score(X_test, Y_test)))
    print(classification_report(Y_test, y_pred))
    plt.subplot(3, 1, 2)
    plot_learning_curve(clf, "svm", X, Y, cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=0))
    return clf


def test_decision_tree(X, Y):
    logging.info("test decision tree:")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    clf = GridSearchCV(DecisionTreeClassifier(), cv=5, param_grid=[
        {
            'criterion': ['entropy'],
            'min_impurity_decrease': np.linspace(0, 1, 50)
        },{
            'criterion':['gini'],
            'min_impurity_decrease': np.linspace(0, 0.5, 50)
        },{
            'max_depth': range(1, 15)
        },{
            'min_samples_split': range(2, 30, 2)
        }
    ])
    clf.fit(X, Y)
    logging.info("best decision tree param: {}\nbest score: {}".format(clf.best_params_, clf.best_score_))
    estimator = clf.best_estimator_
    y_pred = estimator.predict(X_test)
    logging.info("decision tree score: {}".format(clf.score(X_test, Y_test)))
    print(classification_report(Y_test, y_pred))
    print(cross_val_score(clf, X, Y, cv=KFold(n_splits=5)))
    plt.subplot(3, 1, 3)
    plot_learning_curve(clf, "decision tree", X, Y, cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=0))
    return estimator

def test():
    normal_data_set, abnormal_data = load_data()

    normal_data = normal_data_set
    data = []
    data.extend(abnormal_data)
    data.extend(normal_data)
    random.shuffle(data)
    commands = [item[0] for item in data]
    labels = [item[1] for item in data]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(commands)
    Y = labels
    test_svm(X, Y)
    test_knn(X, Y)
    test_svm(X, Y)
    test_decision_tree(X, Y)




def final_svm(X, Y):
    logging.info('report for svm: ')
    clf = svm.SVC(kernel='rbf', gamma=0.22449)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    clf.fit(X_train, Y_train)
    plt.subplot(3, 1, 1)
    plot_learning_curve(clf, "svm", X, Y, cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=0))
    logging.info("svm score: {}".format(clf.score(X_test, Y_test)))
    y_pred = clf.predict(X_test)
    print(classification_report(Y_test, y_pred))

    print(cross_val_score(clf, X, Y, cv=KFold(n_splits=5)))




if __name__ == '__main__':
    test()

