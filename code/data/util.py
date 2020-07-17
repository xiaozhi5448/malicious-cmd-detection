# -*- coding: utf-8 -*-
# @Time    : 7/16/20 4:03 PM
# @Author  : xiaozhi5448
# @Site    : 
# @File    : util.py
# @Software: PyCharm

import os,pickle,logging
from datetime import datetime
import glob
import json
import random

data_dirs = ['data/normal_all', 'data/abnormal_all']
meta_data_dir = 'data/meta_data/'
dataset_bin = 'dataset_clean.pkl'


def load_data():
    """
    从序列化后的数据中，读取命令文本
    :return: 正常命令数据组成的集合，异常命令数据组成的集合
    """
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

def get_dataset():
    normal_commands = set()
    abnormal_commands = set()
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
            logging.info("commands len: {}".format(len(normal_commands) + len(abnormal_commands)))
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

if __name__ == '__main__':
    load_data()

