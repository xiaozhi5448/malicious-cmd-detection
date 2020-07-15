from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import pickle as pkl
import os
import logging
from collections import defaultdict
import openpyxl
import re
logging.basicConfig(level=logging.INFO)
data_dir = '../data/meta_data/'

def parse_data():
    commands = []
    with open(os.path.join(data_dir, 'dataset_clean.pkl'), 'rb') as infp:
        commands = pkl.load(infp)
    logging.info("total {} command items".format(len(commands)))
    abnormal_commands = [item[0] for item in commands if item[1] == 1 and item[0]]
    normal_commands = [item[0] for item in commands if item[1] == 0 and item[0] not in abnormal_commands]

    return normal_commands, abnormal_commands

def parse_program(commands):
    programs = defaultdict(int)
    for item in commands:
        items = item.split(' ')
        program_path = ''
        try:
            if items[0].endswith('bash') or 'sh' in items[0]:
                i = 1
                while True:
                    if items[i].startswith('-'):
                        i += 1
                    else:
                        break
                program_path = items[i]
            else:
                program_path = items[0]
            programs[program_path.split('/')[-1]] += 1
        except Exception as e:
            print(e)
            print(item)


    print("total program: {}".format(len(programs)))
    # print(programs.keys())
    return programs

def plot_length(commands:list, title=''):
    result = defaultdict(int)
    meta_length = defaultdict(int)
    for command in commands:
        items = command.split(' ')
        result[len(items)] += 1
        bound = len(items) // 5
        meta_length["{}s".format(bound * 5)] += 1
    print(meta_length)

    x_label = []
    y_value = []
    for key in result:
        x_label.append(key)
        y_value.append(result[key])
    plt.bar(x_label, y_value)
    plt.title(title)
    plt.xlabel('length')
    plt.ylabel('number')
    # plt.show()

def plot_program(programs):
    labels = []
    values = []
    # plt.figure(figsize=(20, 20))
    for key in programs:
        labels.append(key)
        values.append(programs[key])
    plt.barh(labels, values)
    # plt.show()

if __name__ == '__main__':
    normal_commands, abnormal_commands = parse_data()

    logging.info("total normal commands: {}".format(len(normal_commands)))
    logging.info("total abnormal commands:{}".format(len(abnormal_commands)))
    print(abnormal_commands)
    for command in abnormal_commands:
        if 'grep' in abnormal_commands:
            print(command)
        if len(command) > 1000:
            print(command)
    p1s = parse_program(normal_commands)
    p2s = parse_program(abnormal_commands)
    plt.figure(figsize=(30, 20))
    plot_program(p1s)
    plt.show()
    plt.figure(figsize=(40, 20))
    plt.subplot(2, 1, 1)
    plot_length(normal_commands,'normal')
    plt.subplot(2, 1, 2)
    plot_length(abnormal_commands, 'abnormal')
    plt.tick_params(labelsize=23)
    plt.show()


