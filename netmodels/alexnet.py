'''
图像分类模型的tensorflow实现之--AlexNet

Tensorflow Version: 1.11
Python Version: 3.6

Refs: https://blog.csdn.net/zyqdragon/article/details/72353420 
bms
2018-10-25
'''


import tensorflow as tf
import numpy as np

class AlexNet(object):
    '''
    #use like this:
    model = AlexNet(input, num_classes, keep_prob, is_training)
    score = model.fc8
    # then you can get loss op using score
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = score, labels = y))
    '''
    def __init__(self, input, num_classes, keep_prob=0.5, is_training=True):
        self.INPUT = input
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.IS_TRAINING = is_training
        self.default_image_size = 224

        self.create()

    def create(self):
        # 1st Layer : conv -> pool -> lrn
        conv1 = conv(self.INPUT, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        pool1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
        norm1 = lrn(pool1, 2, 2e-05, 0.75, name='norm1')

        # 2nd Layer: Conv -> Pool -> Lrn
        conv2 = conv(norm1, 5, 5, 256, 1, 1, name='conv2')
        pool2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
        norm2 = lrn(pool2, 2, 2e-05, 0.75, name='norm2')

        # 3rd Layer: Conv
        conv3 = conv(norm2, 3, 3, 384, 1, 1, name='conv3')

        # 4th Layer: Conv
        conv4 = conv(conv3, 3, 3, 384, 1, 1, name='conv4')

        # 5th Layer: Conv  -> Pool
        conv5 = conv(conv4, 3, 3, 256, 1, 1, name='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        # 6th Layer: Flatten -> FC -> Dropout
        flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
        fc6 = fc(flattened, 6 * 6 * 256, 4096, name='fc6')
        dropout6 = dropout(fc6, self.KEEP_PROB, is_training=self.IS_TRAINING)

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fc(dropout6, 4096, 4096, name='fc7')
        dropout7 = dropout(fc7, self.KEEP_PROB, is_training=self.IS_TRAINING)

        # 8th Layer: FC and return unscaled activations (for tf.nn.softmax_cross_entropy_with_logits)
        self.fc8 = fc(dropout7, 4096, self.NUM_CLASSES, name='fc8', is_relu=False)



def conv(input, filter_height, filter_width, num_filters, stride_x, stride_y, name, padding='SAME' ):
    '''
     先定义conv的通用模式
    '''
    input_channels = int(input.get_shape()[-1])
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels, num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])
        conv = tf.nn.conv2d(input, weights,
                            strides=[1, stride_y, stride_x, 1],
                            padding=padding)
        bias = tf.nn.bias_add(conv, biases)
        relu = tf.nn.relu(bias, name=scope.name)

        return relu


def fc(input, num_input, num_output, name, is_relu=True):
    '''定义全连接层'''
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[num_input, num_output], trainable=True)
        biases = tf.get_variable('biases', [num_output], trainable=True)

        act = tf.nn.xw_plus_b(input, weights, biases, name=scope.name)
        if is_relu:
            return tf.nn.relu(act, name=scope.name)
        else:
            return act


def max_pool(input, filter_height, filter_width, stride_x, stride_y, name, padding='SAME'):
    return tf.nn.max_pool(input,
                          ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_x, stride_y, 1],
                          padding=padding,
                          name=name)


def lrn(input, radius=2, alpha=2e-05, beta=0.75, bias=1.0, name=''):
    return tf.nn.local_response_normalization(input,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias,
                                              name=name)


def dropout(input, keep_prob=0.5, is_training=True):
    if is_training:
        return tf.nn.dropout(input, keep_prob)
    else:
        return input