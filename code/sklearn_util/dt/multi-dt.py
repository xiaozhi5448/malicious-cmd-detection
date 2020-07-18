from collections import Counter
import sys
import os, logging, random, pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from data.util import load_data, get_dataset

def multi_decision_predict(classifiers:list, command:str):
    pred_res = []
    for item in classifiers:
        vectorizer = item['vectorizer']
        clf = item['clf']
        X = vectorizer.transform([command])
        pred_y = clf.predict(X)
        pred_res.append(pred_y[0])
    counter = Counter(pred_res)
    # if counter[1] > 5:
    #     return 1
    # else:
    #     return 0
    counts = counter.most_common(2)
    try:

        return counts[0][0]
    except IndexError as e:
        print(command)
        print(counts)
        print(counter)
        print(pred_res)
        sys.exit(1)



def test_multi_decision_tree(normal_commands, abnormal_commands):
    model_filename = os.path.join('data/meta_data/multi_decision_tree', 'model.m')
    # if not os.path.exists(model_filename):
    step = len(normal_commands) // 6000
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
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
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
    # else:
    #     with open(model_filename, 'rb') as outfp:
    #         classifier = pickle.load(outfp)

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    y_preds = []
    for index, command_item in enumerate(normal_commands + abnormal_commands):
        cmd = command_item[0]
        label = command_item[1]
        y_pred = multi_decision_predict(classifiers, cmd)
        y_preds.append(y_pred)
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
        if (index + 1) % 1000 == 0:
            logging.info("{} commands predicted!".format(index + 1))
    real_y = [0 for _ in range(len(normal_commands))]
    real_y.extend([1 for _ in range(len(abnormal_commands))])
    # print("percision normal: {}".format(round(TP/ (TP + FP),2)))
    # print("percision abnormal: {}".format(round(TN / (TN + FN),2)))
    # print("recall normal:{}".format(round(TP/ (TP + FN),2)))
    # print("recall abnormal:{}".format(round(TN/(TN + FP),2)))
    print("report for final test set:")
    print(classification_report(real_y, y_preds))

if __name__ == '__main__':
    normal_data_set, abnormal_data = load_data()
    test_multi_decision_tree(normal_data_set, abnormal_data)