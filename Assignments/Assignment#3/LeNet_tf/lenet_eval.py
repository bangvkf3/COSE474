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
    'Setting#4': "1601193083",
    'Setting#5': "1601189007",
    'Setting#6': "1601198557",
    'Setting#7': "1601205889",
    'Setting#8': "1601212595",
    'Setting#9': "1601217696",
    'Setting#10': "1601230730",
    'Setting#11': "1601243098",
    'Setting#12': "1601247123",
    'Setting#13': '1601251600',
    'Setting#14': '1601254813',
    'Setting#15': '1601265891',
    'Setting#16': '1601269946',
    'Setting#17': '1601271659',
    'Setting#18': '1601275310',
    'Setting#19': '1601278578',
    'Setting#20': '1601281581',
    'Setting#21': '1601284131',
    'Setting#22': '1601288338',
    'Setting#23': '1601291043',
    'Setting#24': '1601292982',
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
    #flag 초기화
    del_all_flags(tf.flags.FLAGS)

    # 그래프 초기화
    tf.reset_default_graph()

    # Eval Parameters
    tf.flags.DEFINE_string("checkpoint_dir", f"./runs/{stamp}/checkpoints", "Checkpoint directory from training run")

    FLAGS = tf.flags.FLAGS
    # ==================================================

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
            print(f'{setting}: Test Max Accuracy:', test_accuracy)
