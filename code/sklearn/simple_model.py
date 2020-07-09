from train import get_dataset, load_data, plot_learning_curve
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
import logging
import warnings
from collections import Counter
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(filename)s %(lineno)d: %(levelname)s  %(message)s')
data_dirs = ['../data/normal_all', '../data/abnormal_all']




def test_knn(X, Y):
    logging.info("knn result:")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)
    logging.info( "knn score: {}".format(clf.score(X_test, Y_test)))
    plt.subplot(3, 1, 1)
    # plot_learning_curve(clf, "sklearn", X, Y, cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=0))
    print(classification_report(Y_test, y_pred))

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
    print(classification_report(Y_test, y_pred))
    # plt.subplot(3, 1, 2)
    # plot_learning_curve(clf, "svm", X, Y, cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=0))

def test_decision_tree(X, Y):
    logging.info("test decision tree:")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    clf = DecisionTreeClassifier()
    clf.fit(X, Y)
    # logging.info("best decision tree param: {}\nbest score: {}".format(clf.best_params_, clf.best_score_))
    # estimator = clf.best_estimator_
    y_pred = clf.predict(X_test)
    logging.info("decision tree score: {}".format(clf.score(X_test, Y_test)))
    print(classification_report(Y_test, y_pred))
    print(cross_val_score(clf, X, Y, cv=KFold(n_splits=5)))
    # plt.subplot(3, 1, 2)
    # plot_learning_curve(clf, "decision tree", X, Y, cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=0))


def multi_decision_predict(classifiers:list, command:str):
    pred_res = []
    for item in classifiers:
        vectorizer = item['vectorizer']
        clf = item['clf']
        X = vectorizer.transform([command])
        pred_y = clf.predict(X)
        pred_res.append(pred_y[0])
    counter = Counter(pred_res)
    counts = counter.most_common(2)
    return counts[0][0]



def test_multi_decision_tree(normal_commands, abnormal_commands):
    classifier = []
    model_filename = os.path.join('meta_data/multi_decision_tree', 'model.m')
    if not os.path.exists(model_filename):
        step = len(normal_commands) // 1000
        classifiers = []
        logging.info("total step:{}".format(step))
        for i in range(step):
            print("epoch {}:".format(i+1))
            some_commands = normal_commands[i * 1000: (i+1) * 1000]
            commands = some_commands + abnormal_commands
            random.shuffle(commands)
            vectorizer = TfidfVectorizer()
            cmds = [item[0] for item in commands]
            labels = [item[1] for item in commands]
            X = vectorizer.fit_transform(cmds)
            Y = labels
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
            clf = DecisionTreeClassifier()
            clf.fit(X, Y)
            y_pred = clf.predict(X_test)
            logging.info("decision tree score: {}".format(clf.score(X_test, Y_test)))
            print(classification_report(Y_test, y_pred))
            classifiers.append({
                'clf': clf,
                'vectorizer': vectorizer
            })
        with open(model_filename, 'wb') as outfp:
            pickle.dump(classifiers, outfp)
    else:
        with open(model_filename, 'rb') as outfp:
            classifier = pickle.load(outfp)

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for index, command_item in enumerate(normal_commands + abnormal_commands):
        cmd = command_item[0]
        label = command_item[1]
        y_pred = multi_decision_predict(classifier, cmd)
        if str(label) == str(y_pred):
            if str(label) == '0':
                TP += 1
            else:
                TN += 1
        else:
            if str(label) == '0':
                FP += 1
            else:
                FN += 1
        if index + 1 % 1000 == 0:
            logging.info("{} commands predicted!".format(index + 1))
    print("percision normal: {}".format(round(TP/ (TP + FP),2)))
    print("percision abnormal: {}".format(round(TN / (TN + FN),2)))
    print("recal normal:{}".format(round(TP/ (TP + FN),2)))
    print("recal abnormal:{}".format(round(TN/(TN + FP),2)))












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
    test_knn(X, Y)
    test_svm(X, Y)
    test_decision_tree(X, Y)
    test_multi_decision_tree(normal_data_set, abnormal_data)

if __name__ == '__main__':
    test()