'''
图像分类模型的tensorflow实现之--VGG-16

Tensorflow Version: 1.11
Python Version: 3.6

Refs: https://blog.csdn.net/qq_40027052/article/details/79015827
bms
2018-10-28
'''


import tensorflow as tf

class VGG16(object):
    '''
    #use like this:
    model = VGG16(input, num_classes, keep_prob, is_training)
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
        with tf.name_scope('vgg16'):
            x = conv('conv1_1', self.INPUT, 64)
            x = conv('conv1_2', x, 64)
            x = max_pool('pool1', x)

            x = conv('conv2_1', x, 128)
            x = conv('conv2_2', x, 128)
            x = max_pool('pool2', x)

            x = conv('conv3_1', x, 256)
            x = conv('conv3_2', x, 256)
            x = conv('conv3_3', x, 256)
            x = max_pool('pool3', x)

            x = conv('conv4_1', x, 512)
            x = conv('conv4_2', x, 512)
            x = conv('conv4_3', x, 512)
            x = max_pool('pool4', x)

            x = conv('conv5_1', x, 512)
            x = conv('conv5_2', x, 512)
            x = conv('conv5_3', x, 512)
            x = max_pool('pool5', x)

            x = fc('fc6', x, 4096)
            x = dropout(x, self.KEEP_PROB)

            x = fc('fc7', x, 4096)
            x = dropout(x, self.KEEP_PROB)

            self.fc8 = fc('fc8', x, self.NUM_CLASSES, is_relu=False)


def conv(name, input, num_filters, filter_height=3, filter_width=3, stride_x=1, stride_y=1, padding='SAME' ):
    '''
     先定义conv的通用模式
    '''
    input_channels = int(input.get_shape()[-1])

    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels, num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])
        conv = tf.nn.conv2d(input, weights,
                            strides=[1, stride_y, stride_x, 1],
                            padding=padding)
        bias = tf.nn.bias_add(conv, biases)
        relu = tf.nn.relu(bias, name=scope.name)

        return relu


def fc(name, input, num_output, is_relu=True, is_trainable=True):
    '''定义全连接层'''
    shape = input.get_shape()
    if len(shape) == 4:
        num_input = shape[1].value * shape[2].value * shape[3].value
    else:
        num_input = shape[-1].value

    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[num_input, num_output], trainable=is_trainable)
        biases = tf.get_variable('biases', [num_output], trainable=is_trainable)

        flat_x = tf.reshape(input, [-1, num_input])
        act = tf.nn.xw_plus_b(flat_x, weights, biases, name=scope.name)
        if is_relu:
            return tf.nn.relu(act, name=scope.name)
        else:
            return act


def max_pool(name, input, filter_height=2, filter_width=2, stride_x=2, stride_y=2, padding='SAME'):
    return tf.nn.max_pool(input,
                          ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_x, stride_y, 1],
                          padding=padding,
                          name=name)


def dropout(input, keep_prob=0.5, is_training=True):
    if is_training:
        return tf.nn.dropout(input, keep_prob)
    else:
        return input













