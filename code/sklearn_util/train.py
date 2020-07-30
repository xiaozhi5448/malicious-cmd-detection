#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
import logging
from datetime import datetime
import random
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit, cross_val_score, KFold
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
import settings
from data.util import load_data, get_dataset
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(filename)s %(lineno)d: %(levelname)s  %(message)s')
meta_data_dir = settings.meta_data_dir
dataset_bin = settings.dataset_bin

plt.figure(figsize=(10, 20))


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    t1 = datetime.now()
    logging.info("plot learning curve start at {}".format(t1))
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
    logging.info("plot finished at {}".format(datetime.now()))
    return plt


def test_knn(X, Y):
    clfs = list()
    clfs.append(('KNN', KNeighborsClassifier(n_neighbors=3)))
    clfs.append(('RadiusKNN', RadiusNeighborsClassifier(n_neighbors=3, radius=500)))
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    for index,clf in enumerate(clfs):
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
        print(cross_val_score(model, X, Y, cv=KFold(n_splits=5)))
        print(classification_report(Y_test, Y_pred))
        plt.subplot(2, 1, index+1)
        logging.info("ploting learning curve of {}".format(clf[0]))
        plot_learning_curve(model, clf[0], X, Y, cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=0))
        logging.info("finished")

def test_svm(X, Y):

    logging.info("svm result:")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    # clf = GridSearchCV(svm.SVC(), cv=5, param_grid=[
    #     {
    #         'kernel':['rbf'],
    #         'gamma': np.linspace(0, 1, 50)
    #     },
    #     {
    #         'kernel':['poly'],
    #         'degree': list(range(1, 10))
    #     }
    # ]
    # )
    clf = svm.SVC(C=1.0, kernel='rbf', gamma=0.5)
    clf.fit(X, Y)
    # logging.info("best svm param: {}\nbest score: {}".format(clf.best_params_, clf.best_score_))
    # estimator = clf.best_estimator_
    y_pred = clf.predict(X_test)
    logging.info("svm score: {}".format(clf.score(X_test, Y_test)))
    print(cross_val_score(clf, X, Y, cv=KFold(n_splits=5)))
    print(classification_report(Y_test, y_pred))
    logging.info("ploting learning curve of svm")
    plot_learning_curve(clf, "svm", X, Y, cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=0))
    logging.info("finished")
    return clf


def test_decision_tree(X, Y):
    logging.info("test decision tree:")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    param_grid = {
        'max_depth': range(1, 10, 1),
        'min_samples_leaf': range(1, 10, 2)
    }
    # clf = GridSearchCV(DecisionTreeClassifier(), cv=5, param_grid=param_grid)
    clf = DecisionTreeClassifier(max_depth=9, min_samples_leaf=1)
    clf.fit(X, Y)
    # logging.info("best decision tree param: {}\nbest score: {}".format(clf.best_params_, clf.best_score_))
    # estimator = clf.best_estimator_
    estimator = clf
    y_pred = estimator.predict(X_test)
    logging.info("decision tree score: {}".format(clf.score(X_test, Y_test)))
    print(classification_report(Y_test, y_pred))
    print(cross_val_score(clf, X, Y, cv=KFold(n_splits=5)))
    logging.info('ploting learn curve of decision tree')
    plot_learning_curve(clf, "decision tree", X, Y, cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=0))
    logging.info("finished!")
    return estimator

def test_byes(X, Y):
    logging.info("test byes:")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    clf = MultinomialNB(alpha=0.001)
    clf.fit(X, Y)
    estimator = clf
    y_pred = estimator.predict(X_test)
    logging.info("MultinomialNB score: {}".format(clf.score(X_test, Y_test)))

    print(classification_report(Y_test, y_pred))
    print(cross_val_score(clf, X, Y, cv=KFold(n_splits=5)))
    logging.info('ploting learn curve of MultinomialNB')
    plot_learning_curve(clf, "MultinomialNB", X, Y, cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=0))
    logging.info("finished!")
    return estimator


class DenseTransformer(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()

def test_lda():
    normal_data_set, abnormal_data = load_data()

    normal_data = normal_data_set
    data = []
    data.extend(abnormal_data)
    data.extend(normal_data)
    random.shuffle(data)
    commands = [item[0] for item in data]
    labels = [item[1] for item in data]
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('to_dense', DenseTransformer()),
        ('classifier', LinearDiscriminantAnalysis())
    ])
    X_train, X_test, Y_train, Y_test = train_test_split(commands, labels, test_size=0.2)

    pipeline.fit(X_train, Y_train)
    y_pred = pipeline.predict(X_test)
    logging.info("result of LDA algorithm:")
    print(classification_report(Y_test, y_pred))


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
    # plt.figure(figsize=(10, 15))
    # test_knn(X, Y)
    # plt.figure(figsize=(12, 8))
    # test_svm(X, Y)
    # plt.show()
    # plt.figure(figsize=(12, 8))
    # test_decision_tree(X, Y)
    # test bayes algorithm
    # plt.figure(figsize=(12, 8))
    # test_byes(X, Y)
    # plt.show()

    # test lda algorithm


    test_lda()




# def final_svm(X, Y):
#     logging.info('report for svm: ')
#     clf = svm.SVC(kernel='rbf', gamma=0.22449)
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
#     clf.fit(X_train, Y_train)
#     plt.subplot(3, 1, 1)
#     plot_learning_curve(clf, "svm", X, Y, cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=0))
#     logging.info("svm score: {}".format(clf.score(X_test, Y_test)))
#     y_pred = clf.predict(X_test)
#     print(classification_report(Y_test, y_pred))
#
#     print(cross_val_score(clf, X, Y, cv=KFold(n_splits=5)))




if __name__ == '__main__':
    test()

