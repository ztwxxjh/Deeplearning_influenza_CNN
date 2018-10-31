import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import CNN_utils
import influenza_CNN
import os
BATCH_SIZE = 64
BATCH_X_SIEZ1 = 329
BATCH_X_SIZE2 = 21
BATCH_SIZE_Y = 1
LEARNING_RATE = 0.0001
TRAINING_STEPS = 2000
costs = []

MODEL_SVAE_PATH = 'MODEL/'
MODEL_NAME = 'model.ckpt'

X_train_orig, Y_train_orig, X_verify_orig, Y_verify_orig = CNN_utils.load_dataset()
X_train = X_train_orig
Y_train = Y_train_orig
X_verify = X_verify_orig
Y_verify = Y_verify_orig
# print(X_train.shape)
# print(Y_train.shape)
# print(X_verify.shape)
# print(Y_verify.shape)


def creat_placeholders(n_y):
    X = tf.placeholder(tf.float32, [None, BATCH_X_SIEZ1, BATCH_X_SIZE2, 1], name='X_input')
    Y = tf.placeholder(tf.float32, [n_y, None], name='Y_input')
    return X, Y


def compute_cost(Y, Y_):
    logits = tf.transpose(Y)
    labels = tf.transpose(Y_)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits, labels=labels))
    return cost


def model(X_train, Y_train, X_verify, Y_verify):
    ops.reset_default_graph()
    m = X_train.shape[0]
    seed = 3
    n_y = Y_train.shape[0]
    X, Y_ = creat_placeholders(n_y)
    Y = influenza_CNN.interface(X)
    cost = compute_cost(Y, Y_)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)
    with tf.control_dependencies([train_step]):
        train_op = tf.no_op(name='train')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for epoch in range(TRAINING_STEPS):
            epoch_cost = 0
            num_minibatches = int(m / BATCH_SIZE)
            seed = seed + 1
            minibatches = CNN_utils.random_mini_batches(X_train, Y_train, BATCH_SIZE, seed)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, minibatch_cost = sess.run([train_op, cost], feed_dict={X: minibatch_X, Y_: minibatch_Y})
                epoch_cost = epoch_cost + minibatch_cost / num_minibatches
            if epoch % 5 == 0:
                costs.append(epoch_cost)
                print("epoch = " + str(epoch) + "      epoch_cost:" + str(epoch_cost))
                if epoch % 100 == 0:
                    saver.save(
                        sess, os.path.join(MODEL_SVAE_PATH, MODEL_NAME)
                    )

        # 计算当前的预测结果
        # correct_prediction = tf.equal(tf.sigmoid(Y), Y_)

        # 计算准确率
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # print(Y_train.shape)
        # print(Y_verify.shape)
        # print("训练集的准确率：", accuracy.eval({X: X_train, Y_: Y_train}))
        # print("测试集的准确率:", accuracy.eval({X: X_verify, Y_: Y_verify}))


def main(argv=None):
    model(X_train, Y_train, X_verify, Y_verify)


if __name__ == '__main__':
    tf.app.run()