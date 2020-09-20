import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data

# 에러 무시 코드
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TimeStamp = {
    'Setting#8': "1600611202",
    'Setting#12': "1600613458"
}

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, validation_size=5000)
# 모델이 저장된 checkpoint 경로

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

    tf.flags.DEFINE_string("checkpoint_dir", f"./runs/{stamp}/checkpoints", "Checkpoint directory from training run")

    FLAGS = tf.flags.FLAGS
    # ==================================================
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir) #가장 validation accuracy가 높은 시점 load
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file) # 저장했던 모델 load

            # Get the placeholders from the graph by name, name을 통해 operation 가져오기
            X = graph.get_operation_by_name("X").outputs[0]
            Y = graph.get_operation_by_name("Y").outputs[0]
            keep_prob = graph.get_operation_by_name("keep_prob").outputs[0]
            # keep_prob = tf.placeholder(tf.float32, name="keep_prob")

            hypothesis = graph.get_operation_by_name("hypothesis").outputs[0]

            correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            feed_dict = {X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0}
            test_accuracy = sess.run(accuracy, feed_dict=feed_dict)
            print(f'{setting}: Test Max Accuracy:', test_accuracy)


