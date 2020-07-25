from sklearn.svm import OneClassSVM
from data.util import load_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import random
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit, cross_val_score, KFold


def test_ocs():
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



    X_train = normal_vecs
    X_test = list(normal_vecs[:1000])
    # y_true = [1 for _ in range(1000)]
    y_true = [1 for _ in range(normal_vecs.shape[0])]
    X_test= abnormal_vecs
    y_true.extend([-1 for _ in range(len(abnormal_data))])

    param_grid = {
        'kernal': ['linear'],
        'nu': np.linspace(0.001, 0.03, 30),
        'degree': np.linspace(5, 15, 11)
    }

    clf = OneClassSVM(kernel='linear', nu=0.002, degree=2)
    clf.fit(normal_vecs)
    # y_pred_0 = clf.predict(normal_vecs)
    # y_pred_1 = clf.predict(abnormal_vecs)
    # y_pred = list(y_pred_0)
    # y_pred.extend(list(y_pred_1))
    y_pred = clf.predict(X)
    print(classification_report(Y, y_pred))


if __name__ == '__main__':
    test_ocs()

