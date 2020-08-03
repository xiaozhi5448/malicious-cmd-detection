import os,sys
sys.path.append(os.getcwd())
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
from sklearn.metrics import classification_report
from data.util import load_data
from data.clean.clean import load_commands
import settings
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(filename)s %(lineno)d: %(levelname)s  %(message)s')

from settings import meta_data_dir

def train(host:str=None):
    BUFFER_SIZE = 50000
    BATCH_SIZE = 128
    TAKE_SIZE = 3000
    def labeler(example, index):
        return example, tf.cast(index, tf.int64)
    logging.info("reading data from disk.......")
    if not host:
        normal_commands, abnormal_commands = load_data()

    elif host == 'agent1':
        normal_commands = load_commands(os.path.join(settings.meta_data_dir, settings.agent1_dataset_cleaned_bin))
        abnormal_commands = load_commands(os.path.join(settings.meta_data_dir, settings.original_abnormal_dataset))
        abnormal_commands |= load_commands(os.path.join(settings.meta_data_dir, settings.addition_abnormal_dataset))
    else:
        normal_commands = load_commands(os.path.join(settings.meta_data_dir, settings.agent2_dataset_cleaned_bin))
        abnormal_commands = load_commands(os.path.join(settings.meta_data_dir, settings.original_abnormal_dataset))
        abnormal_commands |= load_commands(os.path.join(settings.meta_data_dir, settings.addition_abnormal_dataset))
    logging.info("{} normal commands loaded from disk!".format(len(normal_commands)))
    logging.info("{} abnormal commands loaded from disk!".format(len(abnormal_commands)))
    with open(os.path.join(meta_data_dir, 'abnormal_commands.txt'), 'w', encoding='utf-8') as outfp:
        outfp.write(os.linesep.join([item[0].strip() for item in abnormal_commands]))
    with open(os.path.join(meta_data_dir, 'normal_commands.txt'), 'w', encoding='utf-8') as outfp:
        outfp.write(os.linesep.join([item[0].strip() for item in normal_commands]))
    logging.info("loading finished!")
    normal_data = tf.data.TextLineDataset(os.path.join(meta_data_dir, 'normal_commands.txt'))
    abnormal_data = tf.data.TextLineDataset(os.path.join(meta_data_dir, 'abnormal_commands.txt'))
    labeled_data = normal_data.map(lambda ex: labeler(ex, 0))
    labeled_data = labeled_data.concatenate(abnormal_data.map(lambda  ex: labeler(ex, 1)))

    labeled_data = labeled_data.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)
    # for ex in labeled_data.take(5):
    #     print(ex)
    logging.info("add label to dataset")
    tokenizer = tfds.features.text.Tokenizer()

    vocabulary_set = set()
    for text_tensor, _ in labeled_data:
        some_tokens = tokenizer.tokenize(text_tensor.numpy())
        vocabulary_set.update(some_tokens)

    vocab_size = len(vocabulary_set)
    logging.info("vocabulary built")
    # print(vocab_size)

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
    # for line in encoded_data.take(5):
    #     print(line)
    train_data = encoded_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
    train_data = train_data.padded_batch(BATCH_SIZE,padded_shapes=((-1,), ()))

    test_data = encoded_data.take(TAKE_SIZE)
    test_data = test_data.padded_batch(BATCH_SIZE,padded_shapes=((-1,), ( )))

    # sample_text, sample_labels = next(iter(test_data))
    # for item in test_data.take(5).as_numpy_iterator():
    #     print(item[1])
    logging.info("split train and test data set")
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
    logging.info("lstm model built")
    logging.info("training...")
    model.fit(train_data, epochs=5, validation_data=test_data)
    logging.info("finished")
    eval_loss, eval_acc = model.evaluate(test_data)
    y_pred = model.predict(test_data)
    y_true = []
    for batch in test_data.as_numpy_iterator():
        y_true.extend(batch[1])
    pred_labels = [np.argmax(res) for res in y_pred]
    # print(pred_labels)

    print('\nEval loss: {}, Eval accuracy: {}'.format(eval_loss, eval_acc))
    print(classification_report(y_true, pred_labels))
if __name__ == '__main__':
    train()