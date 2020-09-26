import tensorflow as tf
import numpy as np

from tensorflow.keras.datasets.cifar10 import load_data

(x_train_val, y_train_val), (x_test, y_test) = load_data()

# Eval Parameters
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1600681309/checkpoints", "Checkpoint directory from training run")

FLAGS = tf.flags.FLAGS
# ==================================================
y_test_one_hot = np.eye(10)[y_test] #one-hot encoding 만들기
y_test_one_hot = np.squeeze(y_test_one_hot, axis=1)
x_test = (x_test/127.5) - 1

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)#가장 validation accuracy가 높은 시점 load
graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name, name을 통해 operation 가져오기
        X = graph.get_operation_by_name("X").outputs[0]
        Y = graph.get_operation_by_name("Y").outputs[0]
        keep_prob = graph.get_operation_by_name("keep_prob").outputs[0]
        accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]

        test_accuracy = sess.run(accuracy, feed_dict={X: x_test, Y: y_test_one_hot, keep_prob:1.0})
        print('Test Max Accuracy:', test_accuracy)
