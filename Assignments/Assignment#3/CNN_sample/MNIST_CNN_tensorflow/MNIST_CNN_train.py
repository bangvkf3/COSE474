import tensorflow as tf
import time
import os

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784], name="X") # [batch_size, num_pixel]
X_img = tf.reshape(X, [-1, 28, 28, 1])   #[batch_size, 가로, 세로, channel] img 28x28x1 (black/white)
Y = tf.placeholder(tf.float32, [None, 10], name="Y") #[batch_size, num_label]
keep_prob = tf.placeholder(tf.float32, name="keep_prob") #probalbility for dropout

# L1 ImgIn shape=(?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01)) #[filter_size, filter_size, channel, num_filter]
#    Conv     -> (?, 28, 28, 32)
#    Pool     -> (?, 14, 14, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME') # convolution
L1 = tf.nn.relu(L1) # activation
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # pooling
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

# L2 ImgIn shape=(?, 14, 14, 32)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
#    Conv      ->(?, 14, 14, 64)
#    Pool      ->(?, 7, 7, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob) #(?, 7, 7, 64)
L2_flat = tf.reshape(L2, [-1, 7 * 7 * 64]) # 7*7*64 = 448 * 7 = 3136

W3 = tf.get_variable("W3", shape=[7 * 7 * 64, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
hypothesis = tf.nn.xw_plus_b(L2_flat, W3, b, name="hypothesis") #hypothesis를 softmax를 통해 확률값으로 변형, Y와 비교해 cross_entropy error 계산

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))# Y = [0,0,0,1,0,0,0,0,0,0], argmax로 실제 정답 인덱스 표현 e.g., argmax([0,0,0,1,0,0,0,0,0,0]) -> 3
# equal 함수는 정답 예측 여부를 판단, e.g., tf.equal(3,3) = True, tf.equal (8,3) =False
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))#cast 함수는 형변환 함수, e.g., True -> 1.0 False -> 0.0, # 0.0 1.0 0.0 1.0 1.0 1.0...
# reduce_mean으로 batch 안에 있는 data들의 평균 정확도를 계산

costs = []
for var in tf.trainable_variables():
    costs.append(tf.nn.l2_loss(var))
l2_loss = tf.add_n(costs)

l2_reg_lambda = 0.000001
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y)) + l2_reg_lambda * l2_loss
summary_op = tf.summary.scalar("loss", cost)

learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) #back propagation

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer()) # 모든 parameter 값 초기화

training_epochs = 15
batch_size = 100

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
saver = tf.train.Saver(tf.global_variables(), max_to_keep=3) # 저장 객체 saver, max_to_keep은 저장하고자 하는 모델(시점) 개수

max = 0
early_stopped = 0
start_time = time.time()
for epoch in range(training_epochs):
    avg_cost = 0.0 # 전체 training data에 대한 평균 loss를 저장할 변수 초기화
    avg_acc = 0.0
    total_batch = int(mnist.train.num_examples / batch_size) #iteration 55000/ 100 = 550

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size) # batch size 단위로 input과 정답 리턴, e.g., (100, 784), (100, 10),
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.8} # placeholder에 실제 data를 먹여주기 위한 dictionary
        c, _, a = sess.run([cost, optimizer, accuracy], feed_dict=feed_dict) # sess.run을 통해 원하는 operation 실행 및 결과값 return
        avg_cost += c / total_batch
        avg_acc += a / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'training cost =', '{:.9f}'.format(avg_cost), 'training acc =', '{:.9f}'.format(avg_acc))
    # 시각화를 위한 accuracy 값 저장, validation accuracy 계산
    # ========================================================================
    train_summary = tf.Summary(value=[tf.Summary.Value(tag="train_accuracy", simple_value=avg_acc)])
    train_summary_writer.add_summary(train_summary, epoch)  ##
    val_accuracy, summaries = sess.run([accuracy, summary_op],feed_dict={X: mnist.validation.images, Y: mnist.validation.labels, keep_prob: 1.0})
    val_summary_writer.add_summary(summaries, epoch)
    # ========================================================================

    print('Validation Accuracy:', val_accuracy)
    if val_accuracy > max: # validation accuracy가 경신될 때
        max = val_accuracy
        early_stopped = epoch + 1 # early stopping 된 시점
        saver.save(sess, checkpoint_prefix, global_step=early_stopped) # best accuracy에 도달할 때만 모델을 저장함으로써 early stopping

training_time = (time.time() - start_time) / 60
print('Learning Finished!')
print('Validation Max Accuracy:', max)
print('Early stopped time:', early_stopped)
print('training time: ', training_time)