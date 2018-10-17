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
        jpeg_file_path = '%s/%s/*' % (data_dir, text)
        matching_files = tf.gfile.Glob(jpeg_file_path)
        filenames.extend(matching_files)
        texts.extend([text] * len(matching_files))
        labels.extend([label_index] * len(matching_files))
        label_index += 1

    shuffled_index = list(range(len(filenames)))
    random.seed(42)
    random.shuffle(shuffled_index)
    filenames = [filenames[i] for i in shuffled_index]
    texts = [texts[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    logging.info('Found %d JPEG files across %d labels inside %s.' %
                 (len(filenames), len(unique_labels), data_dir))

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


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def _convert_to_example(filename, image_buffer, label, text, height, width):
    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/colorspace': _bytes_feature(colorspace.encode()),
        'image/channels': _int64_feature(channels),
        'image/class/label': _int64_feature(label),
        'image/class/text': _bytes_feature(text.encode()),
        'image/format': _bytes_feature(image_format.encode()),
        'image/filename': _bytes_feature(os.path.basename(filename).encode()),
        'image/encoded': _bytes_feature(image_buffer)}))
    return example



def _is_png(filename):
    return '.png' in filename


def _process_image(filename, coder):
    """Process a single image file.
    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    with open(filename, 'rb') as f:
        image_data = f.read()

    # Convert any PNG to JPEG's for consistency.
    if _is_png(filename):
        logging.info('Converting PNG to JPEG for %s' % filename)
        image_data = coder.png_to_jpeg(image_data)

    # Decode the RGB JPEG.
    image = coder.decode_jpeg(image_data)

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    return image_data, height, width


def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
                               texts, labels, num_shards, command_args):
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s_%s_%.5d-of-%.5d.tfrecord' % (command_args.dataset_name, name, shard, num_shards)
        output_file = os.path.join(command_args.output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            label = labels[i]
            text = texts[i]

            image_buffer, height, width = _process_image(filename, coder)

            example = _convert_to_example(filename, image_buffer, label,
                                          text, height, width)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 1000:
                logging.info('%s [thread %d]: Processed %d of %d images in thread batch.' %
                             (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()

        writer.close()
        logging.info('%s [thread %d]: Wrote %d images to %s' %
                     (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    logging.info('%s [thread %d]: Wrote %d images to %d shards.' %
                 (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()



def _process_image_files(name, filenames, texts, labels, num_shards, command_args):
    assert len(filenames) == len(texts)
    assert len(filenames) == len(labels)

    #sperate files into threads nums parts
    spacing = np.linspace(0, len(filenames), command_args.num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i+1]])

    coord = tf.train.Coordinator()
    coder = ImageCoder()
    threads = []
    for thread_index in range(len(ranges)):
        args = (coder, thread_index, ranges, name, filenames,
                texts, labels, num_shards, command_args)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    coord.join(threads)
    logging.info('%s: Finished writing all %d images in data set.' %
                 (datetime.now(), len(filenames)))
    sys.stdout.flush()



def _process_dataset(name, directory, num_shards, labels_file, command_args):
    filenames, texts, labels = _find_image_files(directory, labels_file, command_args)
    _process_image_files(name, filenames, texts, labels, num_shards, command_args)


def main(command_args):
    check_and_set_default_args(command_args)
    logging.info('will save results into %s' % command_args.output_directory)

    _process_dataset('validation', command_args.validation_directory,
                     command_args.validation_shards, command_args.labels_file, command_args)
    _process_dataset('train', command_args.train_directory,
                     command_args.train_shards, command_args.labels_file, command_args)


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
