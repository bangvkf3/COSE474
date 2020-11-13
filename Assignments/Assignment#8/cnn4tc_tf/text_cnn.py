import tensorflow as tf
import numpy as np


class TextCNN(object):

    def __init__(self, config):
        self.num_classes = config["num_classes"]
        self.vocab_size = config["vocab_size"]
        self.embedding_size = config["embedding_dim"] # word vector size
        self.filter_sizes = list(map(int, config["filter_sizes"].split(","))) # filter의 세로 사이즈, 즉 n-gram의 n 사이즈
        self.num_filters = config["num_filters"] # 각각의 filter size별 개수, 즉 output channel
        self.l2_reg_lambda = config["l2_reg_lambda"]
        self.max_length = config["max_length"] # training data에서 문장의 최대 길이

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, self.max_length], name="input_x") # (batch, max_length) 단어의 index 형태로 문장 표현, input
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y") # (batch, num_classes)
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Embedding layer
        with tf.device('/gpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0), trainable=True,
                name="W") # (|V|, word vector size)
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x) # look up, 즉 단어 index를 바탕으로 word vector를 가져와서 이어 붙임
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1) # batch, max_length, embed_dim, 1

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes): #filter size 별로 convolution 수행
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                n = filter_size * self.embedding_size * self.num_filters
                W = tf.Variable(tf.random_normal(filter_shape, stddev=np.sqrt(2.0/n)), name="W") # He initialization
                # W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                # convolution
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.max_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3) # 각각의 filter에서 나온 feature들을 concatenation
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total]) # (batch_size, 300)

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"): # fully-connected layer
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, self.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        costs = []
        for var in tf.trainable_variables():
            costs.append(tf.nn.l2_loss(var))
        l2_loss = tf.add_n(costs)

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")