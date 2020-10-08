import tensorflow as tf
import os
import time
import gzip
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.keras.datasets.fashion_mnist import load_data
import numpy as np
from GAN.cgan import CGAN

# Model Hyperparameters
tf.flags.DEFINE_integer("Z_dim", 100, "Dimensionality of noise vector (default: 100)")
tf.flags.DEFINE_integer("Y_dim", 10, "The number of labels (default: 10)")
tf.flags.DEFINE_integer("G_hidden_dim", 128, "Dimensionality of hidden layer (default: 100)")
tf.flags.DEFINE_integer("D_hidden_dim", 128, "Dimensionality of hidden layer (default: 100)")

tf.flags.DEFINE_float("lr", 1e-3, "Learning rate(default: 0.01)")
tf.flags.DEFINE_float("lr_decay", 0.99, "Learning rate decay rate (default: 0.98)")
# Training parameters
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 3, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

def batch_iter(x, y, batch_size, num_epochs):

    num_batches_per_epoch = int((len(x) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        shuffle_indices = np.random.permutation(np.arange(len(x)))  # epoch 마다 shuffling
        shuffled_x = x[shuffle_indices]
        shuffled_y = y[shuffle_indices]
        # data 에서 batch 크기만큼 데이터 선별
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(x))
            yield list(zip(shuffled_x[start_index:end_index], shuffled_y[start_index:end_index]))

def preprocess():
    (images, labels), (_, _) = load_data() # fashion mnist data 불러오기
    images = images / 255 # 픽셀값을 0~1 사이로 조정
    images = np.reshape(images, [-1, 28 * 28]) # image shape를 (28,28)에서 (784)로
    labels = np.eye(10)[labels] # label을 one-hot encoding으로 표현
    return [images, labels]

def train(mnist):
    # Training
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            gan = CGAN(FLAGS.flag_values_dict())

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            decayed_lr = tf.train.exponential_decay(FLAGS.lr, global_step, 1000, FLAGS.lr_decay, staircase=True) # lr decay
            D_solver = tf.train.AdamOptimizer(decayed_lr).minimize(gan.D_loss, var_list=gan.theta_D) # Discriminator 내 파라미터를 update하기 위한 operation
            G_solver = tf.train.AdamOptimizer(decayed_lr).minimize(gan.G_loss, var_list=gan.theta_G, global_step=global_step) # Generator 내 파라미터를 update 하기 위한 operation

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Checkpoint directory
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            if not os.path.exists('out/'):
                os.makedirs('out/')

            batches = batch_iter(mnist[0], mnist[1], FLAGS.batch_size, FLAGS.num_epochs) # batch 생성
            for batch in batches:
                batch_xs, batch_ys = zip(*batch)

                # discriminator 학습
                _, D_loss_curr = sess.run([D_solver, gan.D_loss], feed_dict={gan.input_x: batch_xs, gan.input_y: batch_ys, gan.input_z: gan.sample_Z(FLAGS.batch_size, FLAGS.Z_dim)})
                # Generator 학습
                _, step, G_loss_curr = sess.run([G_solver, global_step, gan.G_loss], feed_dict={gan.input_z: gan.sample_Z(FLAGS.batch_size, FLAGS.Z_dim), gan.input_y:batch_ys})
                if step % 100 == 0: # 100 iteration 마다
                    y_sample = np.zeros(shape=[16, FLAGS.Y_dim])
                    for i in range(16): # 0~9 사이의 label 16개
                        y_sample[i, i % 10] = 1
                    # generator를 이용해 image 생성
                    samples, step = sess.run([gan.G_sample, global_step],
                                             feed_dict={gan.input_z: gan.sample_Z(16, FLAGS.Z_dim), gan.input_y: y_sample})

                    fig = plot(samples)
                    plt.savefig('out/{}.png'.format(str(step).zfill(3)), bbox_inches='tight')
                    plt.close(fig)
                    saver.save(sess, checkpoint_prefix, global_step=step)
                    print('Iter: {}'.format(step))
                    print('D loss: {:.4}'.format(D_loss_curr))
                    print('G_loss: {:.4}'.format(G_loss_curr))
                    print()

            if not os.path.exists('final_out/'):
                os.makedirs('final_out/')
            # 각각의 label에 대한 image를 16장 씩 생성
            for i in range(10):
                y_sample = np.zeros(shape=[16, FLAGS.Y_dim])
                y_sample[:, i] = 1
                samples, step = sess.run([gan.G_sample, global_step],
                                         feed_dict={gan.input_z: gan.sample_Z(16, FLAGS.Z_dim), gan.input_y: y_sample})

                fig = plot(samples)
                plt.savefig('final_out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                plt.close(fig)

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

def main(argv=None):
    mnist = preprocess()
    train(mnist)

if __name__ == '__main__':
    tf.app.run()