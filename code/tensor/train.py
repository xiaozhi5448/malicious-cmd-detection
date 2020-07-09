
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ' '
import logging
import pickle
from datetime import datetime
import warnings
import glob
import json
import random
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(filename)s %(lineno)d: %(levelname)s  %(message)s')
data_dirs = ['../data/normal_all', '../data/abnormal_all']
meta_data_dir = '../data/meta_data'
def get_dataset():
    normal_commands = set()
    abnormal_commands = set()
    labels = []
    # vectorizer = CountVectorizer()
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

            # vectorizer.fit(commands)
            logging.info("commands len: {}".format(len(normal_commands) + len(abnormal_commands)))
            # print "vectorizer size: {}".format(round(sys.getsizeof(vectorizer) / (1024*1024), 2))
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

def load_data():
    dataset = None
    if os.path.exists(os.path.join(meta_data_dir, 'dataset_clean.pkl')):
        with open(os.path.join(meta_data_dir, 'dataset_clean.pkl'), 'rb') as infp:
            logging.info("loading data from file: meta_data/dataset_clean.pkl")
            dataset = pickle.load(infp)
    if not dataset:
        dataset = get_dataset()

    abnormal_data = [item for item in dataset if item[1] == 1]
    logging.info("abnormal data item: {}".format(len(abnormal_data)))
    normal_data_set = [item for item in dataset if item[1] == 0]
    return normal_data_set, abnormal_data

def train():
    BUFFER_SIZE = 50000
    BATCH_SIZE = 32
    TAKE_SIZE = 300
    def labeler(example, index):
        return example, tf.cast(index, tf.int64)
    normal_commands, abnormal_commands = load_data()
    with open('meta_data/abnormal_commands.txt', 'w', encoding='utf-8') as outfp:
        outfp.write(os.linesep.join([item[0].strip() for item in abnormal_commands]))
    with open('meta_data/normal_commands.txt', 'w', encoding='utf-8') as outfp:
        outfp.write(os.linesep.join([item[0].strip() for item in normal_commands[:1000]]))

    normal_data = tf.data.TextLineDataset('meta_data/normal_commands.txt')
    abnormal_data = tf.data.TextLineDataset('meta_data/abnormal_commands.txt')
    labeled_data = normal_data.map(lambda ex: labeler(ex, 0))
    labeled_data = labeled_data.concatenate(abnormal_data.map(lambda  ex: labeler(ex, 1)))

    labeled_data = labeled_data.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)
    for ex in labeled_data.take(5):
        print(ex)
    tokenizer = tfds.features.text.Tokenizer()

    vocabulary_set = set()
    for text_tensor, _ in labeled_data:
        some_tokens = tokenizer.tokenize(text_tensor.numpy())
        vocabulary_set.update(some_tokens)

    vocab_size = len(vocabulary_set)
    print(vocab_size)

    encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)

    def encode(text_tensor, label):
        encoded_text = encoder.encode(text_tensor.numpy())
        return encoded_text, label

    def encode_map_fn(text, label):
        # py_func doesn't set the shape of the returned tensors.
        encoded_text, label = tf.py_function(encode,
                                             inp=[text, label],
                                             Tout=(tf.int64, tf.int64))

        # `tf.data.Datasets` work best if all components have a shape set
        #  so set the shapes manually:
        encoded_text.set_shape([None])
        label.set_shape([])

        return encoded_text, label
    encoded_data = labeled_data.map(encode_map_fn)
    for line in encoded_data.take(5):
        print(line)
    train_data = encoded_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
    train_data = train_data.padded_batch(BATCH_SIZE,padded_shapes=((-1,), ()))

    test_data = encoded_data.take(TAKE_SIZE)
    test_data = test_data.padded_batch(BATCH_SIZE,padded_shapes=((-1,), ( )))

    # sample_text, sample_labels = next(iter(test_data))
    print(list(test_data.take(5).as_numpy_iterator()))

    vocab_size += 1

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Embedding(vocab_size, 64))

    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))

    # 一个或多个紧密连接的层
    # 编辑 `for` 行的列表去检测层的大小
    for units in [64, 64]:
        model.add(tf.keras.layers.Dense(units, activation='relu'))

    # 输出层。第一个参数是标签个数。
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_data, epochs=3, validation_data=test_data)

    eval_loss, eval_acc = model.evaluate(test_data)
    y_pred = model.predict(test_data)
    pred_labels = [np.argmax(res) for res in y_pred]
    print(y_pred)
    print('\nEval loss: {}, Eval accuracy: {}'.format(eval_loss, eval_acc))
if __name__ == '__main__':
    train()