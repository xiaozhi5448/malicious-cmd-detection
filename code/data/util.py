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
import settings
from settings import normal_output_dirs, abnormal_output_dirs, agent1_dataset_bin, agent2_dataset_bin
data_dirs = ['data/normal_all', 'data/abnormal_all']
meta_data_dir = settings.meta_data_dir
dataset_bin = settings.dataset_bin
logging.basicConfig(level=logging.INFO)

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
    logging.info("normal data item: {}".format(len(normal_data_set)))
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
    with open('data/meta_data/dataset.pkl', 'wb') as outfp:
        pickle.dump(dataset, outfp)
        logging.info("data dumped to file meta_data/dataset.pkl")
    logging.info("reading data finished!")
    return dataset

def get_host_dataset():
    agent1_commands = set()
    for filename in glob.glob('data/normal_all/Agent1/*.csv'):
        logging.info("processing file {}".format(filename))
        with open(filename, 'r', encoding='utf-8') as infp:
            infp.readline()
            for line in infp:
                if line.strip():
                    items = line.split(',')
                    command = ','.join(items[:-1])
                    if command.strip():
                        agent1_commands.add(command)
    logging.info("agent1 commands count:{}".format(len(agent1_commands)))
    with open(os.path.join(meta_data_dir, agent1_dataset_bin) , 'wb') as outfp:
        pickle.dump(agent1_commands, outfp)
        logging.info('data dumped to file {}'.format(agent1_dataset_bin))

    agent2_commands = set()
    for filename in glob.glob('data/normal_all/Agent2/*.csv'):
        logging.info("process file {}".format(filename))
        with open(filename, 'r', encoding='utf-8') as infp:
            infp.readline()
            for line in infp:
                if line.strip():
                    items = line.split(',')
                    command = ','.join(items[:-1])
                    if command.strip():
                        agent2_commands.add(command)
    logging.info("agent2 commands count:{}".format(len(agent2_commands)))
    with open(os.path.join(meta_data_dir, agent2_dataset_bin), 'wb') as outfp:
        pickle.dump(agent2_commands, outfp)
        logging.info('data dumped to file {}'.format(agent2_dataset_bin))



if __name__ == '__main__':
    get_host_dataset()

