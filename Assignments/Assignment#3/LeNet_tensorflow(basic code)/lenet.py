import tensorflow as tf
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
        # filter1 적용 -> (28, 28, 6) * filter1: 5*5, input_channel: 3, output_channel(# of filters): 6

        # * hint he initialization: stddev = sqrt(2/n), filter에서 n 값은?

        # relu -> (28, 28, 6)
        # max_pooling 적용 -> (14, 14, 6)

        # (14, 14, 6) feature map
        # filter2 적용 -> (10, 10, 16) * filter1: 5*5, input_channel: 6, output_channel(# of filters): 16
        # relu -> (10, 10, 16)
        # max_pooling 적용 -> (5, 5, 16)

        # (5, 5, 16) feature map
        # 평탄화 -> (5 * 5 *16)
        # FC1 추가 (5 * 5 * 16, 120) -> (120)

        # (120) features
        # FC2 추가 (120, 84) -> (84)

        # (84) features
        # Softmax layer 추가 (84) -> (10)

        with tf.variable_scope('logit'):
          self.predictions = tf.argmax(hypothesis, 1, name="predictions")

        with tf.variable_scope('loss'):
          costs = []
          for var in tf.trainable_variables():
              costs.append(tf.nn.l2_loss(var)) # 모든 가중치들의 l2_loss 누적
          l2_loss = tf.add_n(costs)
          xent = tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=self.Y)
          self.loss = tf.reduce_mean(xent, name='xent') + self._l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
