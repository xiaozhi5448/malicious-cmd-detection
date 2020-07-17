#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
将在压缩包中的日志文件解压出来
"""
import tarfile
import rarfile
import os
import gzip
import json
normal_data_dir = 'normal'
normal_data_output = 'normal_all'
import fnmatch
import numpy
from sklearn.feature_extraction.text import CountVectorizer

# 判断某文件是否符合某模式
def is_file_match(filename, partterns):
    for parttern in partterns:
        if fnmatch.fnmatch(filename, parttern):
            return True

    return False

# 查找root目录下除exculde_dir中符合parttern的文件
def find_specific_file(root, partterns=['*'], exclude_dir=[]):
    for rootdir, dirnames, filenames in os.walk(root):
        for filename in filenames:
            if is_file_match(filename, partterns):
                yield os.path.join(rootdir, filename)

        for dirname in dirnames:
            if dirname in exclude_dir:
                dirnames.remove(dirname)

def uncompress_data():
    id = 0
    for file in find_specific_file(normal_data_dir, ['*.rar', "*.gz"], []):
        id += 1
        print('process {}'.format(file))
        dest_file = os.path.join(normal_data_output, 'data-{}.txt'.format(id))
        if file.endswith('gz'):

            fp = gzip.open(file)
            content = fp.read()
            fp.close()
            with open(dest_file, 'w') as out_fp:
                out_fp.write(content)
        elif file.endswith('rar'):
            fp = rarfile.RarFile(file)
            filename = fp.namelist()[0]
            fp.extractall(normal_data_output)
            os.rename(os.path.join(normal_data_output, filename), dest_file)
            fp.close()

def test_data():
    data_file = 'abnormal_all/data_1.txt'
    with open(data_file, 'r') as infp:
        line = infp.readline()
        entry = json.loads(line)
        print(entry['cexe'])
        print(entry['ccmdline'].split('\x00'))

        print(entry)

def get_cmdline():
    total_size = 0
    for file in os.listdir('normal_all'):
        print("processing {}".format(file))
        dest_file = os.path.join('normal_all',"{}.csv".format(file.split('.')[0]))
        with open(os.path.join('normal_all',file), 'r') as infp, open(dest_file, 'w') as outfp:
            outfp.write("cmdline,label{}".format(os.linesep))
            for line in infp:
                entry = json.loads(line)
                outfp.write("{},{}{}".format(entry['ccmdline'], 1, os.linesep))
        total_size += os.path.getsize(dest_file)
    for file in os.listdir('abnormal_all'):
        print("processing {}...".format(file))
        dest_file = os.path.join('abnormal_all', "{}-abnormal.csv".format(file.split('.')[0]))
        with open(os.path.join('abnormal_all',file), 'r') as infp, open(dest_file, 'w') as outfp:
            outfp.write("cmdline,label{}".format(os.linesep))
            for line in infp:
                entry = json.loads(line)
                outfp.write("{},{}{}".format(entry['ccmdline'], 0, os.linesep))
        total_size += os.path.getsize(dest_file)
    print("total data size: {}MB".format(round(total_size / (1024*1024), 2)))

if __name__ == '__main__':
    pass