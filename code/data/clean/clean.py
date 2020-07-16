"""
恶意数据集中包含有正常数据
脚本目的是去除恶意数据集中的正常指令
"""
import openpyxl
import matplotlib.pyplot as plt
import pickle as pkl
import os
import logging
import re
logging.basicConfig(level=logging.INFO)

data_dir = '../'
meta_data_dir = '../meta_data/'
def parse_data():
    commands = []
    with open(os.path.join(meta_data_dir, 'dataset.pkl'), 'rb') as infp:
        commands = pkl.load(infp)
    logging.info("total {} command items".format(len(commands)))
    abnormal_commands = [item[0] for item in commands if item[1] == 1 and item[0]]
    normal_commands = [item[0] for item in commands if item[1] == 0 and item[0] not in abnormal_commands]

    return normal_commands, abnormal_commands
def clean():
    normal_commands, abnormal_commands = parse_data()
    abnormal_workbook = openpyxl.load_workbook(os.path.join(data_dir, 'cheat sheet.xlsx'))
    abnormal_sheet = abnormal_workbook.get_sheet_by_name("cmd")
    abnormal_command_set = set()

    for row in list(abnormal_sheet.rows)[1:92]:
        abnormal_command_set.add(row[0].value)
    for row in list(abnormal_sheet.rows)[116:]:
        abnormal_command_set.add(row[0].value)

    logging.info("{} abnormal items in abnormal cheat sheet!".format(len(abnormal_command_set)))
    logging.info("{} abnormal items in log file".format(len(abnormal_commands)))

    logging.info("{} normal items in log file".format(len(normal_commands)))
    normal_command_set = set(normal_commands) - abnormal_command_set

    grep_commands = [command for command in normal_command_set if re.match("grep -w \d+", command)]
    normal_command_set = normal_command_set - set(grep_commands)
    normal_command_set.add(grep_commands[0])
    logging.info("{} grep items!".format(len(grep_commands)))
    readlink_commands = [command for command in normal_command_set if re.match("readlink /proc/\d+/exe", command)]
    normal_command_set = normal_command_set - set(readlink_commands)
    normal_command_set.add(readlink_commands[0])
    logging.info("{} readlink items".format(len(readlink_commands)))

    top_commands = [command for command in normal_command_set if re.match("top -p \d+ -bn 1", command)]
    normal_command_set = normal_command_set - set(top_commands)
    normal_command_set.add(top_commands[0])
    logging.info("{} top items".format(len(top_commands)))

    jstat_commands = [command for command in normal_command_set if re.match('(/\w+){1,}/jstat -gc \d+', command)]
    normal_command_set -= set(jstat_commands)
    normal_command_set.add(jstat_commands[0])
    logging.info("{} jstat items".format(len(jstat_commands)))

    cp_commands = [command for command in normal_command_set if command.startswith("cp -rf /opt/webserver/")]
    normal_command_set -= set(cp_commands)
    normal_command_set.add(cp_commands[0])
    logging.info("{} cp items".format(len(cp_commands)))

    java_commands = []
    for command in normal_command_set:
        items = command.split(" ")
        if items[0].split('/')[-1] == 'java':
            java_commands.append('java ' + ' '.join(items[1:]))
    logging.info("{} java items".format(len(java_commands)))
    java_command_set = set(java_commands)
    logging.info("{} java items after remove duplicated!".format(len(java_command_set)))
    print(java_commands[0])



    logging.info("{} normal items after clean abnormal command".format(len(normal_command_set)))
    commands = [(command, 0) for command in normal_command_set]
    for command in abnormal_command_set:
        commands.append((command, 1))
    with open(os.path.join(meta_data_dir, 'dataset_clean.pkl'), 'wb') as outfp:
        pkl.dump(commands, outfp)
    logging.info("commands cleaned, dumped to meta_data/dataset_clean.pkl")

if __name__ == '__main__':
    clean()
