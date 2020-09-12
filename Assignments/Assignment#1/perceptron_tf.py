import tensorflow as tf

x_data = [[-1.1, 2.7, 4.3]]

X = tf.placeholder(tf.float32, shape=[None, 3])

W = tf.Variable(tf.random_normal([3, 1], name='weight'))
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis_step = tf.maximum(0.0, tf.sign(tf.matmul(X, W) + b))
hypothesis_sigmoid = tf.sigmoid(tf.matmul(X, W) + b)
hypothesis_ReLU = tf.nn.relu(tf.matmul(X, W) + b)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    prediction_step = sess.run(hypothesis_step, feed_dict={X: x_data})
    prediction_sigmoid = sess.run(hypothesis_sigmoid, feed_dict={X: x_data})
    prediction_ReLU = sess.run(hypothesis_ReLU, feed_dict={X: x_data})

    print(f'Step function: {prediction_step}')
    print(f'Sigmoid function: {prediction_sigmoid}')
    print(f'ReLU function: {prediction_ReLU}')