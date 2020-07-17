from data.util import load_data, get_dataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import logging
import numpy as np
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
        logging.info("precision P: {}; recall p: {}; precision N: {}; recall N: {}; bound:"
                     " {};".format(PrecisionP, RecallP, PrecisionN, RecallN, bound))
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

if __name__ == '__main__':
    test_kmeans()