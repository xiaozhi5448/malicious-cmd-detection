import logging,random
from data.util import load_data
from sklearn.model_selection import train_test_split

def ngram_distance(items1: list, items2: list, n: int = 1):
    if len(items1) < n or len(items2) < n:
        return None
    joined_items1 = [' '.join(items1[i:i + n]) for i in range(len(items1) - n + 1)]
    joined_items2 = [' '.join(items2[i:i + n]) for i in range(len(items2) - n + 1)]
    common_items = set(joined_items1) & set(joined_items2)
    return len(joined_items1) + len(joined_items2) - 2 * len(common_items)

def knn():
    normal_commands, abnormal_commands = load_data()

    normal_commands = [item[0] for item in normal_commands]
    abnormal_commands = [item[0] for item in abnormal_commands]
    test_normal_commands = random.sample(normal_commands, 1000)
    train_normal_commands = set(normal_commands) - set(test_normal_commands)
    normal_commands_splited = [command.split(' ') for command in train_normal_commands]
    abnormal_commands_splited = [command.split(' ') for command in abnormal_commands]
    test_normal_distances = []
    test_abnormal_distances = []
    for command in test_normal_commands:
        current_distances = [ngram_distance(command.split(' '), item) for item in normal_commands_splited ]
        min_distance = min(current_distances)
        test_normal_distances.append(min_distance)
    for command in abnormal_commands:
        current_distances = [ngram_distance(command.split(' '), item) for item in normal_commands_splited]
        min_distance = min(current_distances)
        test_abnormal_distances.append(min_distance)




    pass


if __name__ == '__main__':
    command1 = 'cat testfile > output.txt'
    command2 = 'cat testfile2 > output.txt'
    print(ngram_distance(command1.split(' '), command2.split(), 3))
    knn()
