#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
# from sklearn.externals import joblib
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
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(filename)s %(lineno)d: %(levelname)s  %(message)s')
data_dirs = ['../data/normal_all', '../data/abnormal_all']
meta_data_dir = '../data/meta_data/'
dataset_bin = 'dataset_clean.pkl'
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

def get_dataset():
    normal_commands = set()
    abnormal_commands = set()
    labels = []
    # vectorizer = CountVectorizer()
    t1 = datetime.now()
    logging.info("reading data started at : {}".format(t1))
    for data_directory in data_dirs:
        file_list = glob.glob1(data_directory, "*.txt")
        for data_file in file_list:
            original_file_path = os.path.join(data_directory, data_file)
            logging.info("reading data file {}; file size: {}MB".format(original_file_path, round(
                os.path.getsize(original_file_path) / (1024 * 1024), 2)))
            with open(original_file_path, 'r') as infp:
                for line in infp:
                    if not line:
                        continue
                    entry = json.loads(line)
                    cmdline = ' '.join(entry['ccmdline'].split(u'\x00'))
                    if data_directory.endswith('abnormal_all'):
                        abnormal_commands.add(cmdline)
                    else:
                        normal_commands.add(cmdline)

            # vectorizer.fit(commands)
            logging.info("commands len: {}".format(len(normal_commands) + len(abnormal_commands)))
            # print "vectorizer size: {}".format(round(sys.getsizeof(vectorizer) / (1024*1024), 2))
    t2 = datetime.now()
    delta = t2 - t1
    logging.info("reading data cost {} seconds".format(delta.seconds))
    commands = list(normal_commands)
    labels = [0 for _ in range(len(commands))]
    commands.extend(list(abnormal_commands))
    labels.extend([1 for _ in range(len(abnormal_commands))])
    dataset = list(zip(commands, labels))
    random.shuffle(dataset)
    with open('meta_data/dataset.pkl', 'wb') as outfp:
        pickle.dump(dataset, outfp)
        logging.info("data dumped to file meta_data/dataset.pkl")
    logging.info("reading data finished!")
    return dataset


def train_knn(X, Y):
    clfs = []
    clfs.append(('KNN', KNeighborsClassifier(n_neighbors=3)))
    clfs.append(('RadiusKNN', RadiusNeighborsClassifier(n_neighbors=3, radius=500)))
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    for clf in clfs:
        t1 = datetime.now()
        logging.info("training model {} started at {}".format(clf[0], t1))
        model = clf[1]
        model.fit(X_train, Y_train)
        joblib.dump(model, "meta_data/{}.m".format(clf[0]))
        t2 = datetime.now()
        logging.info("training model {} end at {}, cost {} seconds".format(clf[0], t2, (t2 - t1).seconds))
        score = model.score(X_test, Y_test)
        logging.info("{} score: {}".format(clf[0], score))

def generate_model():
    dataset = None
    if os.path.exists('meta_data/dataset.pkl'):
        with open('meta_data/dataset.pkl', 'rb') as infp:
            logging.info("loading data from file: meta_data/dataset.pkl")
            dataset = pickle.load(infp)
    if not dataset:
        dataset = get_dataset()
    commands = [item[0] for item in dataset]
    labels = [item[1] for item in dataset]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(commands)
    Y = labels
    train_knn(X, Y)

def load_data():
    dataset = None
    if os.path.exists(os.path.join(meta_data_dir, dataset_bin)):
        with open(os.path.join(meta_data_dir, dataset_bin), 'rb') as infp:
            logging.info("loading data from file:{}".format(os.path.join(meta_data_dir, dataset_bin)))
            dataset = pickle.load(infp)
    if not dataset:
        dataset = get_dataset()

    abnormal_data = [item for item in dataset if item[1] == 1]
    logging.info("abnormal data item: {}".format(len(abnormal_data)))
    normal_data_set = [item for item in dataset if item[1] == 0 and item[0] not in abnormal_data]
    return normal_data_set, abnormal_data

def test():
    normal_data_set, abnormal_data = load_data()

    repeat_count = len(set(abnormal_data) & set(normal_data_set))
    logging.info("abnormal command has {} items repeat!".format(repeat_count))
    normal_data = random.sample([item for item in normal_data_set], 1000)

    data = []
    data.extend(abnormal_data)
    data.extend(normal_data)
    random.shuffle(data)
    commands = [item[0] for item in data]
    labels = [item[1] for item in data]
    vectorizer = TfidfVectorizer()
    # vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(commands)
    Y = labels
    test_svm(X, Y)
    # test_decision_tree(X, Y)




def test_knn(X, Y):
    logging.info("sklearn result:")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)
    logging.info( "sklearn score: {}".format(clf.score(X_test, Y_test)))
    plt.subplot(3, 1, 1)
    plot_learning_curve(clf, "sklearn", X, Y, cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=0))
    print(classification_report(Y_test, y_pred))
    # print vectorizer.get_feature_names()

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
    y_pred = clf.predict(X_test)
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

def test_kmeans():
    normal_data_set, abnormal_data = load_data()
    normal_data_set = [item[0] for item in normal_data_set]
    abnormal_data = [item[0] for item in abnormal_data]
    vectorizer = TfidfVectorizer()
    vectorizer.fit(normal_data_set)
    # vectorizer.fit(abnormal_data)
    X = vectorizer.transform(normal_data_set)
    Y = vectorizer.transform(abnormal_data)
    # clf = GridSearchCV(KMeans(), param_grid={'n_clusters':list(range(1, 13))}, cv=5)
    clf = KMeans(n_clusters=1)
    clf.fit(X)
    # print("best param: {}; best score: {};".format(clf.best_params_, clf.best_score_))
    # classifier = clf.best_estimator_
    logging.info(clf.cluster_centers_)
    distances1 = clf.transform(X)
    # logging.info(max(distances1))
    distances2 = clf.transform(Y)
    # logging.info(min(distances2))
    # logging.info(max(distances2))
    logging.info(distances1[:10])
    logging.info(distances2[:10])

    boundary = np.linspace(0.8, 1.2, 300)
    results = []
    for bound in boundary:
        # bound = 1.0408026755852844
        TP = len([item[0] for item in distances1 if item[0] < bound])
        FN = len([item[0] for item in distances1 if item[0] >= bound])
        FP = len([item[0] for item in distances2 if item[0] < bound])
        TN = len([item[0] for item in distances2 if item[0] >= bound])
        try:
            PrecisionP = TP /(TP + FP)
            RecallP = TP /(TP + FN)
            PrecisionN = TN / (TN + FN)
            RecallN = TN /(FP + TN)
        except ZeroDivisionError as e:
            continue
        logging.info("precision P: {}; recall p: {}; precision N: {}; recall N: {}; bound: {};".format(PrecisionP, RecallP, PrecisionN, RecallN, bound))
        results.append((PrecisionP, RecallP, PrecisionN, RecallN, bound))
    x_index = [item[4] for item in results]
    precision_P = [item[0] for item in results]
    recal_P = [item[1] for item in results]
    precision_N = [item[2] for item in results]
    recal_N = [item[3] for item in results]
    plt.figure(figsize=(10, 12))
    plt.subplot(2, 1, 1)
    plt.title("kmeans percision")
    plt.plot(x_index, precision_N, color="blue", linewidth=2.0,label="abnormal cmd percision")
    plt.plot(x_index, precision_P, color="red", linewidth=2.0, label="normal cmd percision")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.title("kmeans recall")
    plt.plot(x_index, recal_P, color="red", linewidth=1.0, label="normal cmd recall")
    plt.plot(x_index, recal_N, color="blue", linewidth=1.0, label="abnormal cmd recall")
    plt.legend()

    plt.show()


def final_decision_tree():
    normal_data_set, abnormal_data = load_data()
    repeat = len(set(normal_data_set) & set(abnormal_data))
    print("{} repeat items!".format(repeat))


    repeat_count = len(set(abnormal_data) & set(normal_data_set))
    logging.info("abnormal command has {} items repeat!".format(repeat_count))
    normal_data = random.sample(normal_data_set, 1000)

    data = []
    data.extend(abnormal_data)
    data.extend(normal_data)
    random.shuffle(data)
    commands = [item[0] for item in data]
    labels = [item[1] for item in data]
    vectorizer = TfidfVectorizer()
    # vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(commands)
    Y = labels
    # final_svm(X, Y)


    if os.path.exists('meta_data/decision_tree.m'):
        clf = joblib.load('meta_data/decision_tree.m')

    else:
        clf = test_decision_tree(X, Y)
        joblib.dump(clf, 'meta_data/decision_tree.m')
    commands = [item for item in normal_data_set]
    commands.extend([item for item in abnormal_data])

    random.shuffle(commands)
    cmds = [item[0] for item in commands]
    labels = [item[1] for item in commands]
    X = vectorizer.transform(cmds)
    Y = labels

    y_pred = clf.predict(X)

    logging.info("decision tree score: {}".format(clf.score(X, Y)))
    print(classification_report(Y, y_pred))
    print(cross_val_score(clf, X, Y, cv=KFold(n_splits=5)))




if __name__ == '__main__':
    # test()
    # plt.show()
    test_kmeans()
    # final_decision_tree()

