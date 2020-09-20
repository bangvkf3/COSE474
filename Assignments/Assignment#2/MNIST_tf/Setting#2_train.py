import tensorflow as tf
import time
import os

from tensorflow.examples.tutorials.mnist import input_data

# 에러 무시 코드
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, validation_size=5000)

X = tf.placeholder(tf.float32, [None, 784], name="X")
Y = tf.placeholder(tf.float32, [None, 10], name="Y")

W1 = tf.get_variable("W1", shape=[784, 200])
b1 = tf.Variable(tf.random_normal([200]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.get_variable("W2", shape=[200, 200])
b2 = tf.Variable(tf.random_normal([200]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.get_variable("W3", shape=[200, 200])
b3 = tf.Variable(tf.random_normal([200]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)

W4 = tf.get_variable("W4", shape=[200, 10])
b4 = tf.Variable(tf.random_normal([10]))


hypothesis = tf.nn.xw_plus_b(L3, W4, b4, name="hypothesis")
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))


correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

summary_op = tf.summary.scalar("accuracy", accuracy)

learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

training_epochs = 100
batch_size = 200

# 저장 directory와 tensor board 시각화를 위한 코드
# ========================================================================
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
train_summary_dir = os.path.join(out_dir, "summaries", "train")
train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
val_summary_dir = os.path.join(out_dir, "summaries", "dev")
val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
# ========================================================================

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)

max = 0
early_stopped = 0
start_time = time.time()
for epoch in range(training_epochs):
    avg_cost = 0.0
    avg_acc = 0.0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _, a = sess.run([cost, optimizer, accuracy], feed_dict=feed_dict)
        avg_cost += c / total_batch
        avg_acc += a / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'training cost =', '{:.9f}'.format(avg_cost))

    # 시각화를 위한 accuracy 값 저장, validation accuracy 계산
    # ========================================================================
    train_summary = tf.Summary(value=[tf.Summary.Value(tag="train_accuracy", simple_value=avg_acc)])
    train_summary_writer.add_summary(train_summary, epoch)
    val_accuracy, summaries = sess.run([accuracy, summary_op],
                                       feed_dict={X: mnist.validation.images, Y: mnist.validation.labels})
    val_summary_writer.add_summary(summaries, epoch)
    # ========================================================================

    print('Validation Accuracy:', val_accuracy)
    if val_accuracy > max:
        max = val_accuracy
        early_stopped = epoch + 1
        saver.save(sess, checkpoint_prefix, global_step=early_stopped)

training_time = (time.time() - start_time) / 60
print('Learning Finished!')
print('Validation Max Accuracy:', max)
print('Early stopped time:', early_stopped)
print('training time: ', training_time)