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
from data.DataAnalysis import parse_program
logging.basicConfig(level=logging.INFO)

data_dir = 'data'
meta_data_dir = 'data/meta_data/'

def ngram_distance(items1: list, items2: list, n: int = 1):
    if len(items1) < n or len(items2) < n:
        return None
    joined_items1 = [' '.join(items1[i:i + n]) for i in range(len(items1) - n + 1)]
    joined_items2 = [' '.join(items2[i:i + n]) for i in range(len(items2) - n + 1)]
    common_items = set(joined_items1) & set(joined_items2)
    return len(joined_items1) + len(joined_items2) - 2 * len(common_items)

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
    normal_command_set |= set(grep_commands[:500])
    logging.info("{} grep items!".format(len(grep_commands)))
    readlink_commands = [command for command in normal_command_set if re.match("readlink /proc/\d+/exe", command)]
    normal_command_set = normal_command_set - set(readlink_commands)
    normal_command_set |= set(readlink_commands[:500])
    logging.info("{} readlink items".format(len(readlink_commands)))

    top_commands = [command for command in normal_command_set if re.match("top -p \d+ -bn 1", command)]
    normal_command_set = normal_command_set - set(top_commands)
    normal_command_set|= set(top_commands[:500])
    logging.info("{} top items".format(len(top_commands)))

    jstat_commands = [command for command in normal_command_set if re.match('(/\w+){1,}/jstat -gc \d+', command)]
    normal_command_set -= set(jstat_commands)
    normal_command_set |= set(jstat_commands[0:500])
    logging.info("{} jstat items".format(len(jstat_commands)))

    cp_commands = [command for command in normal_command_set if command.startswith("cp -rf /opt/webserver/")]
    normal_command_set -= set(cp_commands)
    normal_command_set|= set(cp_commands[0:500])
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

def find_program_from_command(command:str):
    if not command:
        return None
    items = command.split(' ')
    program_path = ''
    try:
        if items[0].endswith('bash') or items[0] == 'sh' or items[0].endswith('/sh'):
            i = 1
            while True:
                if items[i].startswith('-'):
                    i += 1
                else:
                    break
            program_path = items[i]
        else:
            program_path = items[0]
        return program_path
    except Exception as e:
        print(e)
        return None


def add_abnormal_command():
    abnormal_commands = []
    linuxde_filepath = 'data/command_linuxde/total.txt'
    runoob_filepath = 'data/command_runoob/total.txt'
    for filepath in [linuxde_filepath, runoob_filepath]:
        with open(filepath, 'r', encoding='utf-8') as infp:
            for line in infp:
                abnormal_commands.append(line.strip())
    sudo_commands = []
    for command in abnormal_commands:
        items = command.split(' ')
        if items[0] != 'sudo':
            items.insert(0, 'sudo')
            sudo_commands.append(' '.join(items))
    abnormal_commands = [item for item in abnormal_commands if item]
    abnormal_commands.extend(sudo_commands)

    logging.info("{} abnormal command add to dataset!".format(len(abnormal_commands)))
    with open('data/meta_data/dataset_clean.pkl', 'rb') as infp:
        commands_list_with_label = pkl.load(infp)
    original_normal_commands = [item[0] for item in commands_list_with_label if item[1] == 0]
    original_abnormal_commands = [item[0] for item in commands_list_with_label if item[1] == 1]
    normal_programs_dict = parse_program(original_normal_commands)
    normal_programs = set(normal_programs_dict.keys())
    results_abnormal = []
    # for command in abnormal_commands:
    #     distances = [ngram_distance(command.split(' '), normal_command.split(' ')) for normal_command in original_normal_commands]
    #     if min(distances) > 4:
    #         results_abnormal.append(command)
    #     else:
    #         logging.info("command {} dropped".format(command))
    # for command in abnormal_commands:
    #     program = find_program_from_command(command)
    #     if program in normal_programs:
    #         print("command {} dropped".format(command))
    # abnormal_commands = [command for command in abnormal_commands if find_program_from_command(command) not in normal_programs]

    logging.info("{} abnormal saved after removal of normal program".format(len(abnormal_commands)))
    results_abnormal.extend(original_abnormal_commands)
    results_abnormal.extend(abnormal_commands)

    result_dataset = []
    for command in results_abnormal:
        result_dataset.append((command, 1))
    for command in original_normal_commands:
        result_dataset.append((command, 0))
    with open('data/meta_data/dataset_clean_add_linuxde.pkl', 'wb') as outfp:
        pkl.dump(result_dataset, outfp)
    logging.info("finished")




if __name__ == '__main__':
    add_abnormal_command()
