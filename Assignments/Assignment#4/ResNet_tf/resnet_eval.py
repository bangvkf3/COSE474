import tensorflow as tf
import os
import numpy as np

from tensorflow.keras.datasets.cifar10 import load_data

# 에러 무시 코드
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TimeStamp = {
    'Setting#1': "1601189007",
    'Setting#2': "1601196973",
    'Setting#3': "1601191502",
}

(x_train_val, y_train_val), (x_test, y_test) = load_data()
y_test_one_hot = np.eye(10)[y_test]  # one-hot encoding 만들기
y_test_one_hot = np.squeeze(y_test_one_hot, axis=1)
x_test = (x_test / 127.5) - 1


def del_all_flags(FLAGS):  # flags 초기화 함수
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)

for setting, stamp in TimeStamp.items():

    # flag 초기화
    del_all_flags(tf.flags.FLAGS)

    # 그래프 초기화
    tf.reset_default_graph()

    # Eval Parameters
    tf.flags.DEFINE_string("checkpoint_dir", f"./runs/{stamp}/checkpoints", "Checkpoint directory from training run")

    # 플래그 오류
    # tf.flags.DEFINE_string("f", "", "kernel")

    FLAGS = tf.flags.FLAGS

    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)  # 가장 validation accuracy가 높은 시점 load
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            X = graph.get_operation_by_name("X").outputs[0]
            Y = graph.get_operation_by_name("Y").outputs[0]
            accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]

            test_accuracy = sess.run(accuracy, feed_dict={X: x_test, Y: y_test_one_hot})
            print(f'{setting}: Test Max Accuracy:', test_accuracy)
