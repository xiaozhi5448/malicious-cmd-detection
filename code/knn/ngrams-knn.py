import logging,random
from data.util import load_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
logging.basicConfig(level=logging.INFO)



def ngram_distance(items1: list, items2: list, n: int = 1):
    if len(items1) < n or len(items2) < n:
        return None
    joined_items1 = [' '.join(items1[i:i + n]) for i in range(len(items1) - n + 1)]
    joined_items2 = [' '.join(items2[i:i + n]) for i in range(len(items2) - n + 1)]
    common_items = set(joined_items1) & set(joined_items2)
    return len(joined_items1) + len(joined_items2) - 2 * len(common_items)

def knn():
    logging.info("loading data from bin file:{}".format(dataset_bin))
    normal_commands, abnormal_commands = load_data()
    logging.info("finished!")

    normal_commands = [item[0] for item in normal_commands]
    abnormal_commands = [item[0] for item in abnormal_commands]
    test_normal_commands = random.sample(normal_commands, 1000)
    train_normal_commands = set(normal_commands) - set(test_normal_commands)
    normal_commands_splited = [command.split(' ') for command in train_normal_commands]
    test_normal_distances = []
    test_abnormal_distances = []
    logging.info("computing distance for normal commands! total: {}".format(len(test_normal_commands)))
    for index, command in enumerate(test_normal_commands):
        if (index + 1) % 100 ==0:
            logging.info("{} commands finished!".format(index + 1))
        current_distances = [ngram_distance(command.split(' '), item) for item in normal_commands_splited ]
        min_distance = min(current_distances)
        test_normal_distances.append(min_distance)
    logging.info("computing distance for abnormal commands!")
    for command in abnormal_commands:
        current_distances = [ngram_distance(command.split(' '), item) for item in normal_commands_splited]
        min_distance = min(current_distances)
        test_abnormal_distances.append(min_distance)
    logging.info("computing finished!")
    logging.info("test for distance bound in [1, 5]")
    for distance_bound in range(2, 6):
        test_normal_result = [1 if item > distance_bound else 0 for item in test_normal_distances]
        test_abnormal_result = [1 if item > distance_bound else 0 for item in test_abnormal_distances]
        pred_y = test_normal_result
        real_y = [0 for _ in range(len(test_normal_result))]
        pred_y.extend(test_abnormal_result)
        real_y.extend([1 for _ in range(len(test_abnormal_result))])
        print("report for distance bound:{}".format(distance_bound))
        print(classification_report(real_y, pred_y))


if __name__ == '__main__':
    knn()
