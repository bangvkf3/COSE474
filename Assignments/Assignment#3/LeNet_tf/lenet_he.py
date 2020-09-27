import tensorflow as tf
import math
import numpy as np

class LeNet:
    def __init__(self, config):
        self._num_classes = config.num_classes # label 개수 (10개-airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
        self._l2_reg_lambda = config.l2_reg_lambda #weight decay를 위한 lamda 값



        self.X = tf.placeholder(tf.float32, [None, 32, 32, 3], name="X") # 가로: 32, 세로:32, 채널: RGB
        self.Y = tf.placeholder(tf.float32, [None, self._num_classes], name="Y") # 정답이 들어올 자리, [0 0 0 0 0 0 0 0 0 1] one-hot encoding 형태
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob") # dropout 살릴 확률

        ##############################################################################################################
        #                         TODO : LeNet5 모델 생성                                                             #
        ##############################################################################################################


        # (32, 32, 3) image
        # * hint he initialization: stddev = sqrt(2/n), filter에서 n 값은?
        self.W1 = tf.Variable(tf.random_normal([5, 5, 3, 6], stddev=math.sqrt(2/5*5*6))) # HE: stddev=sqrt(2/5*5*6)
        # filter1 적용 -> (28, 28, 6) * filter1: 5*5, input_channel: 3, output_channel(# of filters): 6
        self.L1 = tf.nn.conv2d(self.X, self.W1, strides=[1, 1, 1, 1], padding="VALID")
        # relu -> (28, 28, 6)
        self.L1 = tf.nn.relu(self.L1)
        # max_pooling 적용 -> (14, 14, 6)
        self.L1 = tf.nn.max_pool(self.L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

        # (14, 14, 6) feature map
        self.W2 = tf.Variable(tf.random_normal([5, 5, 6, 16], stddev=math.sqrt(2/5*5*16))) # HE: stddev=sqrt(2/5*5*16)
        # filter2 적용 -> (10, 10, 16) * filter1: 5*5, input_channel: 6, output_channel(# of filters): 16
        self.L2 = tf.nn.conv2d(self.L1, self.W2, strides=[1, 1, 1, 1], padding="VALID")
        # relu -> (10, 10, 16)
        self.L2 = tf.nn.relu(self.L2)
        # max_pooling 적용 -> (5, 5, 16)
        self.L2 = tf.nn.max_pool(self.L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

        # (5, 5, 16) feature map
        # 평탄화 -> (5 * 5 *16)
        self.L2_flat = tf.reshape(self.L2, [-1, 5*5*16])
        
        # FC1 추가 (5 * 5 * 16, 120) -> (120)
        self.W3 = tf.get_variable("W3", shape=[5 * 5 * 16, 120], initializer=tf.contrib.layers.variance_scaling_initializer())
        self.b3 = tf.Variable(tf.random_normal([120]))
        self.L3 = tf.nn.relu(tf.matmul(self.L2_flat, self.W3) + self.b3)
        self.L3 = tf.nn.dropout(self.L3, keep_prob=self.keep_prob)

        # (120) features
        # FC2 추가 (120, 84) -> (84)
        self.W4 = tf.get_variable("W4", shape=[120, 84], initializer=tf.contrib.layers.variance_scaling_initializer())
        self.b4 = tf.Variable(tf.random_normal([84]))
        self.L4 = tf.nn.relu(tf.matmul(self.L3, self.W4) + self.b4)
        self.L4 = tf.nn.dropout(self.L4, keep_prob=self.keep_prob)

        # (84) features
        # Softmax layer 추가 (84) -> (10)
        self.W5 = tf.get_variable("W5", shape=[84, 10], initializer=tf.contrib.layers.variance_scaling_initializer())
        self.b5 = tf.Variable(tf.random_normal([10]))
        self.hypothesis = tf.nn.xw_plus_b(self.L4, self.W5, self.b5, name="hypothesis")

        with tf.variable_scope('logit'):
            self.predictions = tf.argmax(self.hypothesis, 1, name="predictions")

        with tf.variable_scope('loss'):
            self.costs = []
            for var in tf.trainable_variables():
                self.costs.append(tf.nn.l2_loss(var))  # 모든 가중치들의 l2_loss 누적
            self.l2_loss = tf.add_n(self.costs)
            self.xent = tf.nn.softmax_cross_entropy_with_logits(logits=self.hypothesis, labels=self.Y)
            self.loss = tf.reduce_mean(self.xent, name='xent') + self._l2_reg_lambda * self.l2_loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
