import tensorflow as tf
import numpy as np

##############################################################################################################
#                    TODO : X1 ~ X7에 올바른 숫자 또는 변수를 채워넣어 ResNet32 코드를 완성할 것                 #
##############################################################################################################

class CGAN(object):

    def __init__(self, config):
        self.D_hidden_dim = config["D_hidden_dim"] # discriminator의 hidden size
        self.G_hidden_dim = config["G_hidden_dim"] # generator의 hidden size
        self.Z_dim = config["Z_dim"] # noise vector size
        self.Y_dim = config["Y_dim"] # output layer size

        # Placeholders for input
        self.input_x = tf.placeholder(tf.float32, shape=[None, 784])
        self.input_y = tf.placeholder(tf.float32, shape=[None, self.Y_dim])
        self.input_z = tf.placeholder(tf.float32, shape=[None, self.Z_dim])

        # discriminator의 weight matrix들을 초기화
        self.D_W1 = self.kernel('D_W1', 3, 1, 16)
        self.D_W2_0 = self.kernel('D_W2_0', 3, 16, 16)
        self.D_W2_1 = self.kernel('D_W2_1', 3, 16, 16)
        self.D_W2_2 = self.kernel('D_W2_2', 3, 16, 16)
        self.D_W2_3 = self.kernel('D_W2_3', 3, 16, 16)

        self.D_W3_0 = self.kernel('D_W3_0', 3, 16, 32)
        self.D_W3_1 = self.kernel('D_W3_1', 3, 32, 32)
        self.D_W3_2 = self.kernel('D_W3_2', 3, 32, 32)
        self.D_W3_3 = self.kernel('D_W3_3', 3, 32, 32)

        self.D_FC_W1 = tf.get_variable("D_FC_W1", shape=[32 + self.Y_dim, self.D_hidden_dim], initializer=tf.contrib.layers.xavier_initializer())
        self.D_FC_b1 = tf.Variable(tf.zeros(shape=[self.D_hidden_dim]))

        self.D_FC_W2 = tf.get_variable("D_FC_W2", shape=[self.D_hidden_dim, 1], initializer=tf.contrib.layers.xavier_initializer())
        self.D_FC_b2 = tf.Variable(tf.zeros(shape=[1]))

        # discriminator의 파라미터 집합 (generator의 파라미터는 freeze 하기 위함)
        self.theta_D = [self.D_W1, self.D_W2_0, self.D_W2_1, self.D_W2_2, self.D_W2_3, self.D_W3_0, self.D_W3_1, self.D_W3_2, self.D_W3_3, self.D_FC_W1, self.D_FC_b1, self.D_FC_W2, self.D_FC_b2]

        # generator의 weight matrix들을 초기화
        self.G_W1 = self.kernel('G_W1', 3, 1, 16)
        self.G_W2_0 = self.kernel('D_G2_0', 3, 16, 16)
        self.G_W2_1 = self.kernel('D_G2_1', 3, 16, 16)
        self.G_W2_2 = self.kernel('D_G2_2', 3, 16, 16)
        self.G_W2_3 = self.kernel('D_G2_3', 3, 16, 16)

        self.G_W3_0 = self.kernel('G_W3_0', 3, 16, 32)
        self.G_W3_1 = self.kernel('G_W3_1', 3, 32, 32)
        self.G_W3_2 = self.kernel('G_W3_2', 3, 32, 32)
        self.G_W3_3 = self.kernel('G_W3_3', 3, 32, 32)

        self.G_FC_W1 = tf.get_variable("G_FC_W1", shape=[32 + self.Y_dim, self.D_hidden_dim], initializer=tf.contrib.layers.xavier_initializer())
        self.G_FC_b1 = tf.Variable(tf.zeros(shape=[self.G_hidden_dim]))

        self.G_FC_W2 = tf.get_variable("D_GC_W2", shape=[self.G_hidden_dim, 784], initializer=tf.contrib.layers.xavier_initializer())
        self.G_FC_b2 = tf.Variable(tf.zeros(shape=[784]))

        # generator의 파라미터 집합 (discriminator 의 파라미터는 freeze 하기 위함)
        self.theta_G = [self.G_W1, self.G_W2_0, self.G_W2_1, self.G_W2_2, self.G_W2_3, self.G_W3_0, self.G_W3_1, self.G_W3_2, self.G_W3_3, self.G_FC_W1, self.G_FC_b1, self.G_FC_W2, self.G_FC_b2]

        self.G_sample = self.generator(self.input_z, self.input_y) # generator가 생성한 image
        D_real, D_logit_real = self.discriminator(self.input_x, self.input_y) # discriminator의 실제이미지에 대한 예측
        D_fake, D_logit_fake = self.discriminator(self.G_sample, self.input_y) # discriminator의 가짜이미지에 대한 예측

        # discriminator의 실제이미지 예측에 대한 loss
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
        # discriminator의 가짜이미지 예측에 대한 loss
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
        # discriminator의 loss
        self.D_loss = D_loss_real + D_loss_fake
        # generator의 loss
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

    def sample_Z(self, m, n):
        return np.random.uniform(-1., 1., size=[m, n]) # noise 생성

    def generator(self, z, y): # generator에서 affine 연산 및 activation
        z = tf.reshape(z, [-1, 10, 10, 1])
        activate_before_residual = [True, False]

        L1 = tf.nn.conv2d(z, self.G_W1, [1, 1, 1, 1], padding="SAME")
        L2 = self.residual(L1, self.G_W2_0, activate_before_residual[0], strides=[1, 1, 1, 1])
        L2 = self.residual(L2, self.G_W2_1, strides=[1, 1, 1, 1])
        L2 = self.residual(L2, self.G_W2_2, strides=[1, 1, 1, 1])
        L2 = self.residual(L2, self.G_W2_3, strides=[1, 1, 1, 1])
        L3 = self.residual(L2, self.G_W3_0, activate_before_residual[1], strides=[1, 2, 2, 1])
        L3 = self.residual(L3, self.G_W3_1, strides=[1, 1, 1, 1])
        L3 = self.residual(L3, self.G_W3_2, strides=[1, 1, 1, 1])
        L3 = self.residual(L3, self.G_W3_3, strides=[1, 1, 1, 1])
        L3 = self.global_avg_pool(L3)
        inputs = tf.concat(axis=1, values=[L3, y])
        G_h1 = tf.nn.relu(tf.matmul(inputs, self.G_FC_W1) + self.G_FC_b1)

        G_log_prob = tf.matmul(G_h1, self.G_FC_W2) + self.G_FC_b2
        G_prob = tf.nn.sigmoid(G_log_prob)

        return G_prob

    def discriminator(self, x, y): # discriminator에서 affine 연산 및 activation
        x = tf.reshape(x, [-1, 28, 28, 1])

        activate_before_residual = [True, False]

        L1 = tf.nn.conv2d(x, self.D_W1, [1, 1, 1, 1], padding="SAME")
        L2 = self.residual(L1, self.D_W2_0, activate_before_residual[0], strides=[1, 1, 1, 1])
        L2 = self.residual(L2, self.D_W2_1, strides=[1, 1, 1, 1])
        L2 = self.residual(L2, self.D_W2_2, strides=[1, 1, 1, 1])
        L2 = self.residual(L2, self.D_W2_3, strides=[1, 1, 1, 1])
        L3 = self.residual(L2, self.D_W3_0, activate_before_residual[1], strides=[1, 2, 2, 1])
        L3 = self.residual(L3, self.D_W3_1, strides=[1, 1, 1, 1])
        L3 = self.residual(L3, self.D_W3_2, strides=[1, 1, 1, 1])
        L3 = self.residual(L3, self.D_W3_3, strides=[1, 1, 1, 1])
        L3 = self.global_avg_pool(L3)
        inputs = tf.concat(axis=1, values=[L3, y])
        D_h1 = tf.nn.relu(tf.matmul(inputs, self.D_FC_W1) + self.D_FC_b1)
        D_logit = tf.matmul(D_h1, self.D_FC_W2) + self.D_FC_b2
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob, D_logit

    def kernel(self, name, filter_size, in_filters, out_filters):
        n = filter_size * filter_size * out_filters
        kernel = tf.get_variable(
            name, [filter_size, filter_size, in_filters, out_filters],
            tf.float32, initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
        return kernel

    def residual(self, x, kernel, activate_before_residual=False, strides=[1, 1, 1, 1]):

        in_filter = kernel.get_shape()[2]
        out_filter = kernel.get_shape()[3]


        if activate_before_residual:
            x = tf.nn.relu(x)
            orig_x = x
        else:
            orig_x = x
            x = tf.nn.relu(x)

        x = tf.nn.conv2d(x, kernel, strides, padding="SAME")

        x = tf.nn.relu(x)
        # x = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding="SAME")

        if in_filter != out_filter:
            orig_x = tf.nn.avg_pool(orig_x, strides, strides, 'VALID')
            orig_x = tf.pad(
                orig_x, [[0, 0], [0, 0], [0, 0],
                         [(out_filter - in_filter) // 2, (out_filter - in_filter) // 2]])
        x += orig_x
        return x

    def global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])

