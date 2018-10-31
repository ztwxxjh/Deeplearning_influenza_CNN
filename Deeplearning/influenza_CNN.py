import tensorflow as tf


NUM_CHANNELS = 1

CONV1_DEEP = 16
CONV1_SIZE = 5

CONV2_DEEP = 32
CONV2_SIZE = 3

FC1_SIZE = 128
FC2_SIZE = 25
NUM_LABELS = 1


def interface(input_tensor):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        conv1_biases = tf.get_variable(
            "bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0)
        )
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    with tf.variable_scope('layer2_pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1,  2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    with tf.variable_scope('layer3_conv2'):
        conv2_weights = tf.get_variable(
            "weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        conv2_biases = tf.get_variable(
            "bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0)
        )
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    with tf.variable_scope('layer4_pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [nodes, -1])
    with tf.variable_scope('layer5_fc1'):
        fc1_weight = tf.get_variable(
            "weights", [FC1_SIZE, nodes],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        # if regularizer != None:
        #     tf.add_to_collection('losses', regularizer(fc1_weight))
        fc1_biases = tf.get_variable(
            "biase", [FC1_SIZE, 1], initializer=tf.constant_initializer(0.0)
        )
        fc1 = tf.nn.relu(tf.matmul(fc1_weight, reshaped) + fc1_biases)
        # if train: fc1 = tf.nn.dropout(fc1, 0.5)
    with tf.variable_scope('layer6_fc2'):
        fc2_weight = tf.get_variable(
            "weight", [FC2_SIZE, FC1_SIZE],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        # if regularizer != None:
        #     tf.add_to_collection('losses', regularizer(fc2_weight))
        fc2_biases = tf.get_variable(
            "biase", [FC2_SIZE, 1], initializer=tf.constant_initializer(0.0)
        )
        fc2 = tf.nn.relu(tf.matmul(fc2_weight, fc1)+fc2_biases)
        # if train: fc2 = tf.nn.dropout(fc2, 0.5)
    with tf.variable_scope('layer7_fc3'):
        fc3_weight = tf.get_variable(
            "weight", [NUM_LABELS, FC2_SIZE],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        fc3_biases = tf.get_variable(
            "biase", [NUM_LABELS, 1],
            initializer=tf.constant_initializer(0.0)
        )

        logit = tf.matmul(fc3_weight, fc2)+fc3_biases

    return logit