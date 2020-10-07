import tensorflow as tf
import os
import time
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from GAN.vanilla import GAN

# Model Hyperparameters
tf.flags.DEFINE_integer("Z_dim", 100, "Dimensionality of noise vector (default: 100)")
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

def preprocess():
    # Load data
    mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
    print("Train: {:d}".format(mnist.train.num_examples))
    return mnist

def train(mnist):
    # Training
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            gan = GAN(FLAGS.flag_values_dict())

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            decayed_lr = tf.train.exponential_decay(FLAGS.lr, global_step, 1000, FLAGS.lr_decay, staircase=True)
            D_solver = tf.train.AdamOptimizer(decayed_lr).minimize(gan.D_loss, var_list=gan.theta_D)
            G_solver = tf.train.AdamOptimizer(decayed_lr).minimize(gan.G_loss, var_list=gan.theta_G, global_step=global_step)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            if not os.path.exists('out/'):
                os.makedirs('out/')

            for epoch in range(FLAGS.num_epochs):
                total_batch = int(mnist.train.num_examples / FLAGS.batch_size)

                for _ in range(total_batch):
                    batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)

                    _, D_loss_curr = sess.run([D_solver, gan.D_loss], feed_dict={gan.input_x: batch_xs, gan.input_z: gan.sample_Z(FLAGS.batch_size, FLAGS.Z_dim)})
                    _, step, G_loss_curr = sess.run([G_solver, global_step, gan.G_loss], feed_dict={gan.input_z: gan.sample_Z(FLAGS.batch_size, FLAGS.Z_dim)})
                    if step % 100 == 0:

                        samples, step = sess.run([gan.G_sample, global_step],
                                                 feed_dict={gan.input_z: gan.sample_Z(16, FLAGS.Z_dim)})

                        fig = plot(samples)
                        plt.savefig('out/{}.png'.format(str(step).zfill(3)), bbox_inches='tight')
                        plt.close(fig)
                        saver.save(sess, checkpoint_prefix, global_step=step)
                        print('Iter: {}'.format(step))
                        print('D loss: {:.4}'.format(D_loss_curr))
                        print('G_loss: {:.4}'.format(G_loss_curr))
                        print()

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