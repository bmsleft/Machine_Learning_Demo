# -*- coding: UTF-8 -*-
"""
Date: 2018-10-17
Note: self-define iamge recognize
Ref:  《21个项目玩转深度学习 --基于TensorFlow的实战详解》
    第 3 章 打造自己的图像识别模型
"""

import argparse
import os
import logging
import sys
import threading
import random
import numpy as np
import tensorflow as tf

from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tensorflow-data-dir', default='pic/')
    parser.add_argument('--train-shards', default=2, type=int)
    parser.add_argument('--validation-shards', default=2, type=int)
    parser.add_argument('--num-threads', default=2, type=int)
    parser.add_argument('--dataset-name', default='satellite', type=str)
    return parser.parse_args()



def check_and_set_default_args(command_args):
    if not(hasattr(command_args, 'train_shards')) or command_args.train_shards is None:
        command_args.train_shards = 5
    if not(hasattr(command_args, 'validation_shards')) or command_args.validation_shards is None:
        command_args.validation_shards = 5
    if not(hasattr(command_args, 'num_threads')) or command_args.num_threads is None:
        command_args.num_threads = 5
    if not(hasattr(command_args, 'class_label_base')) or command_args.class_label_base is None:
        command_args.class_label_base = 0
    if not(hasattr(command_args, 'dataset_name')) or command_args.dataset_name is None:
        command_args.dataset_name = ''
    assert not command_args.train_shards % command_args.num_threads, (
        'Please make the command_args.num_threads commensurate with command_args.train_shards')
    assert not command_args.validation_shards % command_args.num_threads, (
        'Please make the command_args.num_threads commensurate with '
        'command_args.validation_shards')
    assert command_args.train_directory is not None
    assert command_args.validation_directory is not None
    assert command_args.labels_file is not None
    assert command_args.output_directory is not None



def _find_image_files(data_dir, labels_file, command_args):
    '''
    :param data_dir:
    :param labels_file:
    :param command_args:
    :return:
      filenames: list of strings; each string is a path to an image file.
      texts: list of strings; each string is the class, e.g. 'dog'
      labels: list of integer; each integer identifies the ground truth.
    '''
    logging.info('Determining list of input files and labels from %s.' % data_dir)
    unique_labels = [l.strip() for l in tf.gfile.FastGFile(labels_file, 'r').readlines()]

    filenames= []
    texts = []
    labels = []
    label_index = command_args.class_label_base

    for text in unique_labels:
        jpeg_file_path = os.path.join(data_dir, text)
        matching_files = tf.gfile.Glob(jpeg_file_path)
        filenames.extend(matching_files)
        texts.extend([text] * len(matching_files))
        labels.extend([label_index] * len(matching_files))
        label_index += 1

    shuffled_index = range(len(filenames))
    random.seed(42)
    random.shuffle(shuffled_index)
    filenames = [filenames[i] for i in shuffled_index]
    texts = [texts[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    return filenames, texts, labels


class ImageCoder(object):
    def __init__(self):
        self._sess = tf.Session()
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        self._decode_jpg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpg = tf.image.decode_jpeg(self._decode_jpg_data, channels=3)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpg, feed_dict={self._png_data: image_data})

    def decode_jpeg(self, image_date):
        jpg_image = self._sess.run(self._decode_jpg, feed_dict={self._decode_jpg_data: image_date})
        assert len(jpg_image.shape) == 3
        assert jpg_image.shape[2] == 3
        return jpg_image
    

def _process_image_files(name, filenames, texts, labels, num_shards, command_args):
    assert len(filenames) == len(texts)
    assert len(filenames) == len(labels)

    #sperate files into threads nums parts
    spacing = np.linspace(0, len(filenames), command_args.num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i+1]])

    coord = tf.train.Coordinator()








def _process_dataset(name, directory, num_shards, labels_file, command_args):
    filenames, texts, labels = _find_image_files(directory, labels_file, command_args)
    _process_image_files(name, filenames, texts, labels, num_shards, command_args)


def main(command_args):
    check_and_set_default_args(command_args)
    logging.info('will save results into %s' % command_args.output_directory)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    args.tensorflow_dir = args.tensorflow_data_dir
    args.train_directory = os.path.join(args.tensorflow_dir, 'train')
    args.validation_directory = os.path.join(args.tensorflow_dir, 'validation')
    args.output_directory = args.tensorflow_dir
    args.labels_file = os.path.join(args.tensorflow_dir, 'label.txt')

    if os.path.exists(args.labels_file) is False:
        logging.warning('create new label.txt cause can not find it')
        all_entries = os.listdir(args.train_directory)
        dirnames = []
        for entry in all_entries:
            if os.path.isdir(os.path.join(args.train_directory, entry)):
                dirnames.append(entry)
        with open(args.labels_file, 'w') as f:
            for dirname in dirnames:
                f.write(dirname + '\n')

    main(args)
