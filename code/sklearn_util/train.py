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
from data.clean.clean import load_commands
import os
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

def get_precision_recall(report:str):
    """

    :param report: returned by sklearn function "classification_report"
    :return: [normal_precision, normal_recall, abnormal_precision, abnormal_recall]
    """
    if not report:
        return None
    lines = report.split('\n')
    normal_res = lines[2].strip()
    abnormal_res = lines[3].strip()
    normal_items = normal_res.split()
    abnormal_items = abnormal_res.split()
    return [float(normal_items[1]), float(normal_items[2]), float(abnormal_items[1]), float(abnormal_items[2])]

def test_knn(X, Y, plot=True):
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
        report_res = classification_report(Y_test, Y_pred)
        print(report_res)
        if plot:
            plt.subplot(2, 1, index+1)
            logging.info("ploting learning curve of {}".format(clf[0]))
            plot_learning_curve(model, clf[0], X, Y, cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=0), ylim=[0.5, 1])
            logging.info("finished")


def test_svm(X, Y, plot=True):

    logging.info("svm result:")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    clf = svm.SVC(C=1.0, kernel='rbf', gamma=0.5)
    clf.fit(X, Y)
    y_pred = clf.predict(X_test)
    logging.info("svm score: {}".format(clf.score(X_test, Y_test)))
    # print(cross_val_score(clf, X, Y, cv=KFold(n_splits=5)))
    report_res = classification_report(Y_test, y_pred)
    print(report_res)

    if plot:
        logging.info("ploting learning curve of svm")
        plot_learning_curve(clf, "svm", X, Y, cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=0), ylim=[0.8, 1.0])
        logging.info("finished")
    return clf, report_res


def test_decision_tree(X, Y, plot=True):
    logging.info("test decision tree:")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    clf = DecisionTreeClassifier(max_depth=9, min_samples_leaf=1)
    clf.fit(X, Y)
    estimator = clf
    y_pred = estimator.predict(X_test)
    logging.info("decision tree score: {}".format(clf.score(X_test, Y_test)))
    report_res = classification_report(Y_test, y_pred)
    print(report_res)
    # print(cross_val_score(clf, X, Y, cv=KFold(n_splits=5)))
    if plot:
        logging.info('ploting learn curve of decision tree')
        plot_learning_curve(clf, "decision tree", X, Y, cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=0), ylim=[0.8, 1])
        logging.info("finished!")
    return clf, report_res

def test_bayes(X, Y, plot=True):
    logging.info("test byes:")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    clf = MultinomialNB(alpha=0.001)
    clf.fit(X, Y)
    estimator = clf
    y_pred = estimator.predict(X_test)
    logging.info("MultinomialNB score: {}".format(clf.score(X_test, Y_test)))
    report_res = classification_report(Y_test, y_pred)
    print(report_res)
    # print(cross_val_score(clf, X, Y, cv=KFold(n_splits=5)))
    if plot:
        logging.info('ploting learn curve of MultinomialNB')
        plot_learning_curve(clf, "MultinomialNB", X, Y, cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=0), ylim=[0.8, 1])
        logging.info("finished!")
    return clf, report_res

def test(agent="", injection=False):
    if not agent:
        normal_data_set, abnormal_data = load_data()
        normal_commands = [item[0] for item in normal_data_set]
        abnormal_commands = [item[0] for item in abnormal_data]
    elif agent == 'agent1':
        logging.info("test for agent1")
        original_abnormal_commands = load_commands(os.path.join(meta_data_dir, settings.original_abnormal_dataset))
        addition_abnormal_commands = load_commands(os.path.join(meta_data_dir, settings.addition_abnormal_dataset))
        normal_commands = load_commands(os.path.join(meta_data_dir, settings.agent1_dataset_cleaned_bin))
        abnormal_commands = original_abnormal_commands | addition_abnormal_commands
    else:
        logging.info("test for agent2")
        original_abnormal_commands = load_commands(os.path.join(meta_data_dir, settings.original_abnormal_dataset))
        addition_abnormal_commands = load_commands(os.path.join(meta_data_dir, settings.addition_abnormal_dataset))
        normal_commands = load_commands(os.path.join(meta_data_dir, settings.agent2_dataset_cleaned_bin))
        abnormal_commands = original_abnormal_commands | addition_abnormal_commands
    commands = list(normal_commands)
    commands.extend(list(abnormal_commands))
    labels = [0 for _ in range(len(normal_commands))]
    labels.extend([1 for _ in range(len(abnormal_commands))])
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(commands)
    Y = labels
    # plt.figure(figsize=(10, 15))
    # test_knn(X, Y, plot=False)
    plt.figure(figsize=(12, 8))
    svm_res = test_svm(X, Y, plot=False)
    plt.show()
    plt.figure(figsize=(12, 8))
    dt_res = test_decision_tree(X, Y, plot=False)
    # test bayes algorithm
    plt.figure(figsize=(12, 8))
    bayes_res = test_bayes(X, Y, plot=False)
    plt.show()


    # test inject attack
    abnormal_samples = random.sample(abnormal_commands, 100)
    normal_samples = random.sample(normal_commands, 100)
    concat_tokens = ['|', '||', '&&', ';']
    all_commands = []
    for command in normal_samples:
        for abnormal_command in abnormal_samples:
            token = random.choice(concat_tokens)
            all_commands.append(command + token + abnormal_command)
    X = vectorizer.transform(all_commands)
    Y = [1 for _ in range(len(all_commands))]
    # bayes
    logging.info("injection report for bayes")
    pred_y = bayes_res[0].predict(X)
    print(pred_y)
    print(classification_report(Y, pred_y))

    logging.info("injection report for svm")
    pred_y = svm_res[0].predict(X)
    print(pred_y)
    print(classification_report(Y, pred_y))

    logging.info("injection report for dt")
    pred_y = dt_res[0].predict(X)
    print(pred_y)
    print(classification_report(Y, pred_y))
    return {
        'svm': svm_res,
        'dt': dt_res,
        'bayes': bayes_res
    }

if __name__ == '__main__':
    common_res = test()


