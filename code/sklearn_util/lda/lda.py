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
from sklearn.decomposition import PCA, TruncatedSVD
import settings
from data.util import load_data, get_dataset
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(filename)s %(lineno)d: %(levelname)s  %(message)s')
meta_data_dir = settings.meta_data_dir
dataset_bin = settings.dataset_bin


def test_dimension():
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

    candidate_dimension = range(500, 5000, 500)
    explained_ratios = []
    for dimension in candidate_dimension:
        model = TruncatedSVD(n_components=dimension)
        model.fit_transform(X)
        explained_ratios.append(np.sum(model.explained_variance_ratio_))
        logging.info("feature: {}, explained ratio: {}".format(dimension, explained_ratios[-1]))

    plt.figure(figsize=(10, 8))
    plt.grid()
    plt.plot(candidate_dimension, explained_ratios)
    plt.xlabel("Number of pca components")
    plt.ylabel('Explained Variance Ratio')
    plt.yticks(np.arange(0.5, 1.05, 0.05))
    plt.show()







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


if __name__ == '__main__':
    test_dimension()