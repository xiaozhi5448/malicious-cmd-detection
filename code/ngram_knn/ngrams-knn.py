import logging,random
from data.util import load_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
logging.basicConfig(level=logging.DEBUG)
import time
from threading import Lock
from datetime import datetime
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import settings
from settings import meta_data_dir
from data.clean.clean import load_commands
random.seed(time.time())

def ngram_distance(items1: list, items2: list, n: int = 1):
    if len(items1) < n or len(items2) < n:
        return None
    # joined_items1 = [' '.join(items1[i:i + n]) for i in range(len(items1) - n + 1)]
    # joined_items2 = [' '.join(items2[i:i + n]) for i in range(len(items2) - n + 1)]
    common_items = set(items1) & set(items2)
    distance = len(items1) + len(items2) - 2 * len(common_items)
    return distance / (len(items1) + len(items2))

def plot_bar(distances:list, color='b'):
    items = Counter(distances)
    data = []
    for item in items.items():
        data.append((item[0], item[1]))
    data.sort(key=lambda item: item[0])
    x_labels = [item[0] for item in data]
    y_values = [item[1] for item in data]
    plt.xlim((0., 1.05))
    plt.xlabel("distance")
    plt.ylabel("count")
    plt.plot(x_labels, y_values, color=color)

def plot_agent_distance(distance1, distance2):
    fig, ax = plt.subplots()
    items = Counter(distance1)
    data = []
    for item in items.items():
        data.append((item[0], item[1]))
    data.sort(key=lambda item: item[0])
    x_labels = [item[0] for item in data]
    y_values = [item[1] for item in data]
    plt.xlim((0., 1.05))
    plt.xlabel("distance")
    plt.ylabel("count")
    line1, = ax.plot(x_labels, y_values, label="normal", color="g")

    items = Counter(distance2)
    data = []
    for item in items.items():
        data.append((item[0], item[1]))
    data.sort(key=lambda item: item[0])
    x_labels = [item[0] for item in data]
    y_values = [item[1] for item in data]
    # plt.xlim((0., 1.05))
    # plt.xlabel("distance")
    # plt.ylabel("count")
    line2, = ax.plot(x_labels, y_values, label="abnormal", color="r")
    ax.legend()


def get_min_distance(command:list, normal_commands:list):
    """

    :param command:
    :param normal_commands:
    :return:
    """
    current_distances = [ngram_distance(command, item) for item in normal_commands]
    min_distance = min(current_distances)
    logging.debug("distance : {}".format(min_distance))
    return min_distance

def predict(commands:list, normal_commands:list) -> list:
    """

    :param commands:
    :param normal_commands:
    :return:
    """
    t1 = datetime.now()
    logging.info("calculating distance (total item: {}), started at: {}".format(len(commands), t1))
    executor = ThreadPoolExecutor(max_workers=32)
    handles = [executor.submit(get_min_distance, command, normal_commands) for command in commands]
    wait(handles, return_when=ALL_COMPLETED)
    min_distances = [item.result() for item in handles]
    # for index, command in enumerate(commands):
    #     if (index + 1) % 100 ==0:
    #         logging.info("processing {}th command...".format(index + 1))
    #     min_distance = get_min_distance(command, normal_commands)
    #     res.append(min_distance)
    t2 = datetime.now()
    logging.info("calculate finished, cost {} seconds".format((t2-t1).seconds))
    return min_distances

def encode(vocabulary, command:list)->list:
    res = []
    for word in command:
        try:
            res.append(vocabulary[word])
        except KeyError as e:
            res.append(-1)
    return res

def build_vocabulary(voca:dict, commands:list):
    vocabulary = voca
    index = max(list(vocabulary.values())) if vocabulary else 0
    for line in commands:
        words = line.strip().split()
        for word in words:
            if word not in vocabulary:
                index+=1
                vocabulary[word] = index
    return vocabulary

def knn(agent=""):
    logging.info("loading data ......")
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

    logging.info('loading finished!')

    vocabulary = build_vocabulary({}, normal_commands)
    vocabulary = build_vocabulary(vocabulary, abnormal_commands)
    logging.info("vocabulaey built finished!")
    abnormal_commands = random.sample(abnormal_commands, 500)
    test_normal_commands = random.sample(normal_commands, 500)
    train_normal_commands = set(normal_commands) - set(test_normal_commands)
    train_normal_commands.add('readlink')
    train_normal_commands.add('grep')
    train_normal_commands.add('top')
    train_normal_commands.add('jstat')
    normal_commands_splited = [command.strip().split(' ') for command in train_normal_commands if command]
    normal_commands_vecs = [encode(vocabulary, line) for line in normal_commands_splited]
    logging.info("computing distance for normal commands! total: {}".format(len(test_normal_commands)))
    test_normal_commands_vecs = [encode(vocabulary, line.split(' ')) for line in test_normal_commands]
    logging.info("convert command to vectors finished!")
    plt.figure(figsize=(10, 15))
    # plt.subplot(2, 1, 1)
    # plt.title('normal test set distance distribution')
    test_normal_distances = predict(test_normal_commands_vecs, normal_commands_vecs)
    logging.info("computing distance for abnormal commands!")
    # plot_bar(test_normal_distances)
    # plt.subplot(2, 1, 2)
    # plt.title('abnormal test set distance distribution')
    abnormal_commands_vecs = [encode(vocabulary, line.split(' ')) for line in abnormal_commands]
    test_abnormal_distances = predict(abnormal_commands_vecs, normal_commands_vecs)
    # plot_bar(test_abnormal_distances)
    # plt.show()
    plot_agent_distance(test_normal_distances, test_abnormal_distances)
    plt.show()

    logging.info("computing finished!")
    logging.info("test for distance bound in [0.2, 0.7]")
    X = []
    normal_precision = []
    normal_recall = []
    abnormal_precision = []
    abnormal_recall = []
    for distance_bound in np.linspace(0.2, 0.7, 21):
        test_normal_result = [1 if item > distance_bound else 0 for item in test_normal_distances]
        test_abnormal_result = [1 if item > distance_bound else 0 for item in test_abnormal_distances]
        pred_y = test_normal_result
        real_y = [0 for _ in range(len(test_normal_result))]
        pred_y.extend(test_abnormal_result)
        real_y.extend([1 for _ in range(len(test_abnormal_result))])
        print("report for distance bound:{}".format(distance_bound))
        report_res = classification_report(real_y, pred_y)
        lines = report_res.split('\n')
        normal_res = lines[2].strip()
        abnormal_res = lines[3].strip()
        normal_items = normal_res.split()
        abnormal_items = abnormal_res.split()
        normal_precision.append(float(normal_items[1]))
        normal_recall.append(float(normal_items[2]))
        abnormal_precision.append(float(abnormal_items[1]))
        abnormal_recall.append(float(abnormal_items[2]))
        X.append(distance_bound)
        print(report_res)

    plt.figure(figsize=(15, 20))
    plt.subplot(2, 1, 1)
    plt.plot(X, normal_recall, color='red', label="normal  recall", linewidth=2)
    plt.plot(X, normal_precision, color="green", label="normal  precision", linewidth=2)
    plt.legend()
    plt.grid("on")
    plt.ylim((0.8, 1.05))
    plt.xlabel("distance")
    plt.ylabel("rate")
    plt.title("normal recall/precision")

    plt.subplot(2, 1, 2)
    plt.plot(X, abnormal_recall, color="red", label="abnormal  recall")
    plt.plot(X, abnormal_precision, color="green", label="abnormal precision")
    plt.legend()
    plt.grid("on")
    plt.ylim((0.8, 1.05))
    plt.xlabel("distance")
    plt.ylabel("rate")
    plt.title("abnormal recall/precision")
    plt.show()



if __name__ == '__main__':
    # print(ngram_distance("readlink /proc/235072/exe ".split(), "command: /opt/java8/jre/bin/java -Dnop -DCATALINA_STARTUP_CHECKER_FLAG=/opt/ecm/app/0010/proc/workspace0/home/FusionGIS/OpenAS_Tomcat7/temp1a0b/HwEoLrLlOd.tcatpid -Djava.util.logging.manager=org.apache.juli.ClassLoaderLogManager -Dorg.apache.catalina.security.SecurityListener.UMASK=0077 -Dopenas.catalina.out.log.file.control=off -Dfile.encoding=UTF-8 -Dopenas.accesslog.control=on -Dopenas.tomcat.flow.control=false -Dopenas.tomcat.flow.control.socket.reuseaddr=true -Dopenas.tomcat.flow.control.reject.timeout=1000 -server -XX:+UseParallelGC -XX:ParallelGCThreads=16 -XX:+UseAdaptiveSizePolicy -Xms3072m -Xmx5760m -XX:NewRatio=4 -XX:PermSize=512m -XX:MaxPermSize=1024m -Xloggc:/opt/ecm/app/0010/proc/workspace0/home/FusionGIS/OpenAS_Tomcat7/logs/tomcatdump/tomcat_gc_200502144203.log -XX:+PrintGCDetails -XX:ErrorFile=/opt/ecm/app/0010/proc/workspace0/home/FusionGIS/OpenAS_Tomcat7/logs/tomcatdump/tomcat_error_200502144203.log -XX:+UseGCLogFileRotation -XX:NumberOfGCLogFiles=10 -XX:GCLogFileSize=10M -Djava.security.egd=file:/dev/./urandom -Dsun.rmi.dgc.server.gcInterval=0x7FFFFFFFFFFFFFE -Dsun.rmi.dgc.client.gcInterval=0x7FFFFFFFFFFFFFE -Dopenas.log.close.interval=3600000 -Dopenas.log.debug.level=error -Djava.endorsed.dirs= -classpath /opt/ecm/app/0010/proc/workspace0/home/FusionGIS/OpenAS_Tomcat7/bin/bootstrap.jar:/opt/ecm/app/0010/proc/workspace0/home/FusionGIS/OpenAS_Tomcat7/bin/tomcat-juli.jar -Dcatalina.base=/opt/ecm/app/0010/proc/workspace0/home/FusionGIS/OpenAS_Tomcat7 -Dcatalina.home=/opt/ecm/app/0010/proc/workspace0/home/FusionGIS/OpenAS_Tomcat7 -Djava.io.tmpdir=/opt/ecm/app/0010/proc/workspace0/home/FusionGIS/OpenAS_Tomcat7/temp org.apache.catalina.startup.Bootstrap start ".split()))
    knn('agent1')
    knn('agent2')

