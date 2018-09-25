# -*- coding: UTF-8 -*-
"""
Date: 2018-9-25
Note: CIFAR-10 iamge recognize
Ref:  《21个项目玩转深度学习 --基于TensorFlow的实战详解》
    第 2 章 CIFAR-10 与ImageNet图像识别
"""

import os
import tarfile
import urllib
import sys
from PIL import Image
import tensorflow as tf
from six.moves import xrange

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
DATA_PATH = 'cifar10_data'


def maybe_download_and_extract(data_dir=DATA_PATH, data_url=DATA_URL):
    """
    Download and extract the tarball from Alex's website.
    """
    dest_directory = data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)

    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
          sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
              float(count * block_size) / float(total_size) * 100.0))
          sys.stdout.flush()
        
        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
        statinfo = os.stat(filepath)
        print('\nSuccessfully downloaded', filename, statinfo.st_size, 'bytes.')

    extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def read_cifar10(filename_queue):
    class CIFAR10Record(object):
        pass

    result = CIFAR10Record()

    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    label_bytes = 1  # 2 for CIFAR-100
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes

    # Read a record, getting filenames from the filename_queue.  No
    # header or footer in the CIFAR-10 format, so we leave header_bytes
    # and footer_bytes at their default of 0.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)

    # The first bytes represent the label, which we convert from uint8->int32.
    result.label = tf.cast(
        tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(
        tf.strided_slice(record_bytes, [label_bytes],
                         [label_bytes + image_bytes]),
        [result.depth, result.height, result.width])
    # Convert from [depth, height, width] to [height, width, depth].
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result


def inputs_origin(data_dir):
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in xrange(1, 6)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    filename_queue = tf.train.string_input_producer(filenames)
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    return reshaped_image



def extract_jpg():
    with tf.Session() as sess:
        reshaped_image = inputs_origin(DATA_PATH + '/cifar-10-batches-bin')
        threads = tf.train.start_queue_runners(sess=sess)
        sess.run(tf.global_variables_initializer())
        if not os.path.exists(DATA_PATH + '/raw/'):
            os.makedirs(DATA_PATH + '/raw/')
        for i in range(30):
            image_array = sess.run(reshaped_image)
            Image.fromarray(image_array.astype('uint8')).save(DATA_PATH + '/raw/%d.jpg' % i)



if __name__ == '__main__':
    maybe_download_and_extract()
    extract_jpg()


