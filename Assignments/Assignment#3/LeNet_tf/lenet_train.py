import tensorflow as tf
import os
import time
import datetime
from tensorflow.keras.datasets.cifar10 import load_data
import data_helpers as dh
from lenet002 import LeNet
import math

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Hyperparameters
Settings = {
    'SET#1': {
        'BATCH_SIZE': 128,
        'LR': 0.001,
        'EPOCH': 200,
        'KEEP_PROB': 0.9,
        'L2_REG_LAMBDA': 0,
        'DATA_AUGMENTATION': False,
        'LR_DECAY': 1,
        'DECAY_STEPS': 1000
    },
    'SET#2': {
        'BATCH_SIZE': 128,
        'LR': 0.001,
        'EPOCH': 200,
        'KEEP_PROB': 0.9,
        'L2_REG_LAMBDA': 0.0001,
        'DATA_AUGMENTATION': False,
        'LR_DECAY': 1,
        'DECAY_STEPS': 1000
    },
    'SET#3': {
        'BATCH_SIZE': 128,
        'LR': 0.001,
        'EPOCH': 200,
        'KEEP_PROB': 0.9,
        'L2_REG_LAMBDA': 0.0001,
        'DATA_AUGMENTATION': True,
        'LR_DECAY': 1,
        'DECAY_STEPS': 1000
    },
    'SET#4': {
        'BATCH_SIZE': 128,
        'LR': 0.001,
        'EPOCH': 200,
        'KEEP_PROB': 0.9,
        'L2_REG_LAMBDA': 0.0001,
        'DATA_AUGMENTATION': True,
        'LR_DECAY': 0.99,
        'DECAY_STEPS': 1000
    },
    'SET#5': {
        'BATCH_SIZE': 128,
        'LR': 0.001,
        'EPOCH': 200,
        'KEEP_PROB': 1.0,
        'L2_REG_LAMBDA': 0.0001,
        'DATA_AUGMENTATION': True,
        'LR_DECAY': 1,
        'DECAY_STEPS': 1000
    },
    'SET#6': {
        'BATCH_SIZE': 128,
        'LR': 0.01,
        'EPOCH': 200,
        'KEEP_PROB': 0.95,
        'L2_REG_LAMBDA': 0.0001,
        'DATA_AUGMENTATION': True,
        'LR_DECAY': 0.99,
        'DECAY_STEPS': 1000
    },
    'SET#7': {  # 3x3x16 레이어 추가
        'BATCH_SIZE': 128,
        'LR': 0.001,
        'EPOCH': 200,
        'KEEP_PROB': 0.9,
        'L2_REG_LAMBDA': 0.0001,
        'DATA_AUGMENTATION': True,
        'LR_DECAY': 1,
        'DECAY_STEPS': 1000
    },
    'SET#8': {
        'BATCH_SIZE': 128,
        'LR': 0.01,  # 150에폭까지 0.01, 이후로는 0.05
        'EPOCH': 250,
        'KEEP_PROB': 0.9,
        'L2_REG_LAMBDA': 0.0001,
        'DATA_AUGMENTATION': True,
        'LR_DECAY': 1,
        'DECAY_STEPS': 1000
    },
    'SET#9': {
        'BATCH_SIZE': 128,
        'LR': 0.001,  # 100에폭까지 0.001, 150에폭 까지 0.0005 이후 0.00025
        'EPOCH': 200,
        'KEEP_PROB': 0.9,
        'L2_REG_LAMBDA': 0.0001,
        'DATA_AUGMENTATION': True,
        'LR_DECAY': 1,
        'DECAY_STEPS': 1000
    },
    'SET#10': {
        'BATCH_SIZE': 100,
        'LR': 0.001,  # 100에폭까지 0.001, 150에폭 까지 0.0005, 200에폭까지 0.00025, 250에폭까지 0.0001
        'EPOCH': 250,
        'KEEP_PROB': 0.9,
        'L2_REG_LAMBDA': 0.0001,
        'DATA_AUGMENTATION': True,
        'LR_DECAY': 1,
        'DECAY_STEPS': 1000
    },
    'SET#11': {
        'BATCH_SIZE': 128,
        'LR': 0.001,  # 100에폭까지 0.001, 200에폭 까지 0.0005 이후 0.0001
        'EPOCH': 400,
        'KEEP_PROB': 0.9,
        'L2_REG_LAMBDA': 0.0001,
        'DATA_AUGMENTATION': True,
        'LR_DECAY': 1,
        'DECAY_STEPS': 1000
    },
    'SET#12': {  # He
        'BATCH_SIZE': 128,
        'LR': 0.001,  # 250에폭까지 0.001, 350에폭 까지 0.0005 이후 0.0001
        'EPOCH': 400,
        'KEEP_PROB': 0.9,
        'L2_REG_LAMBDA': 0.0001,
        'DATA_AUGMENTATION': True,
        'LR_DECAY': 1,
        'DECAY_STEPS': 1000
    },
    'SET#13': {  # He, expo decay
        'BATCH_SIZE': 128,
        'LR': 0.001,
        'EPOCH': 300,
        'KEEP_PROB': 0.95,
        'L2_REG_LAMBDA': 0.0001,
        'DATA_AUGMENTATION': True,
        'LR_DECAY': 1,
        'DECAY_STEPS': 1000
    },
    'SET#14': {  # He, expo decay
        'BATCH_SIZE': 128,
        'LR': 0.001,
        'EPOCH': 400,
        'KEEP_PROB': 0.95,
        'L2_REG_LAMBDA': 0.0001,
        'DATA_AUGMENTATION': True,
        'LR_DECAY': 1,
        'DECAY_STEPS': 1000
    },
    'SET#15': {  # He, expo decay
        'BATCH_SIZE': 64,
        'LR': 0.001,
        'EPOCH': 300,
        'KEEP_PROB': 0.95,
        'L2_REG_LAMBDA': 0.0001,
        'DATA_AUGMENTATION': True,
        'LR_DECAY': 1,
        'DECAY_STEPS': 1000
    },
    'SET#16': {  # He, expo decay
        'BATCH_SIZE': 256,
        'LR': 0.001,
        'EPOCH': 300,
        'KEEP_PROB': 0.95,
        'L2_REG_LAMBDA': 0.0001,
        'DATA_AUGMENTATION': True,
        'LR_DECAY': 1,
        'DECAY_STEPS': 1000
    },
    'SET#17': {  # He, expo decay
        'BATCH_SIZE': 64,
        'LR': 0.001,
        'EPOCH': 300,
        'KEEP_PROB': 0.95,
        'L2_REG_LAMBDA': 0.0001,
        'DATA_AUGMENTATION': True,
        'LR_DECAY': 0.98,
        'DECAY_STEPS': 1000
    },
    'SET#18': {  # expo decay
        'BATCH_SIZE': 64,
        'LR': 0.001,
        'EPOCH': 300,
        'KEEP_PROB': 0.95,
        'L2_REG_LAMBDA': 0.0001,
        'DATA_AUGMENTATION': True,
        'LR_DECAY': 0.99,
        'DECAY_STEPS': 1000
    },
    'SET#19': {  # expo decay
        'BATCH_SIZE': 64,
        'LR': 0.001,
        'EPOCH': 400,
        'KEEP_PROB': 0.95,
        'L2_REG_LAMBDA': 0.0001,
        'DATA_AUGMENTATION': True,
        'LR_DECAY': 0.98,
        'DECAY_STEPS': 1000
    },
    'SET#20': {  # He, expo decay
        'BATCH_SIZE': 128,
        'LR': 0.001,
        'EPOCH': 400,
        'KEEP_PROB': 0.95,
        'L2_REG_LAMBDA': 0.0001,
        'DATA_AUGMENTATION': True,
        'LR_DECAY': 0.99,
        'DECAY_STEPS': 1000
    },
    'SET#21': {  # expo decay
        'BATCH_SIZE': 32,
        'LR': 0.001,
        'EPOCH': 400,
        'KEEP_PROB': 0.95,
        'L2_REG_LAMBDA': 0.0001,
        'DATA_AUGMENTATION': True,
        'LR_DECAY': 0.99,
        'DECAY_STEPS': 1000
    },
    'SET#22': {  # 002 init
        'BATCH_SIZE': 64,
        'LR': 0.001,
        'EPOCH': 300,
        'KEEP_PROB': 0.95,
        'L2_REG_LAMBDA': 0.0001,
        'DATA_AUGMENTATION': True,
        'LR_DECAY': 0.99,
        'DECAY_STEPS': 1000
    },
    'SET#23': {
        'BATCH_SIZE': 128,
        'LR': 0.001,
        'EPOCH': 300,
        'KEEP_PROB': 0.8,
        'L2_REG_LAMBDA': 0.0001,
        'DATA_AUGMENTATION': True,
        'LR_DECAY': 0.99,
        'DECAY_STEPS': 1000
    },
    'SET#24': {  # filter stddev 002
        'BATCH_SIZE': 64,
        'LR': 0.001,
        'EPOCH': 400,
        'KEEP_PROB': 0.95,
        'L2_REG_LAMBDA': 0.0001,
        'DATA_AUGMENTATION': True,
        'LR_DECAY': 0.99,
        'DECAY_STEPS': 1000
    },
}

# Choose a setting
setting = Settings['SET#24']

# clean flags, graph
def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)


del_all_flags(tf.flags.FLAGS)
tf.reset_default_graph()

# 플래그 오류
tf.flags.DEFINE_string("f", "", "kernel")

# Model Hyperparameters
tf.flags.DEFINE_float("lr", setting['LR'], "learning rate (default=0.1)")
tf.flags.DEFINE_float("lr_decay", setting['LR_DECAY'], "learning rate decay rate(default: 0.1)")
tf.flags.DEFINE_integer("decay_steps", setting['DECAY_STEPS'], "learning rate decay steps(default: 1000)")
tf.flags.DEFINE_float("l2_reg_lambda", setting['L2_REG_LAMBDA'], "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("keep_prob", setting['KEEP_PROB'], "keep probability for dropout (default: 1.0)")
tf.flags.DEFINE_integer("num_classes", 10, "The number of classes (default: 10)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", setting['BATCH_SIZE'], "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", setting['EPOCH'], "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 350, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 350, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 3, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_boolean("data_augmentation", setting['DATA_AUGMENTATION'], "data augmentation option")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.flags.FLAGS

(x_train_val, y_train_val), (x_test, y_test) = load_data() # training data: 50000, test data: 10000
x_train, y_train, x_test, y_test, x_val, y_val = dh.shuffle_data(x_train_val, y_train_val, x_test, y_test, FLAGS.num_classes)
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        lenet = LeNet(FLAGS) #LeNet 클래스의 인스턴스 생성 후 Hyperparameter가 정의돼 있는 FLAGS로 초기화

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False) # iteration 수




        # * hint learning rate decay를 위한 operation을 통해 감쇠된 learning rate를 optimizer에 적용)
        decayed_lr = tf.train.exponential_decay(FLAGS.lr, global_step, FLAGS.decay_steps, FLAGS.lr_decay, staircase=True)
        # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)

        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.lr)
        optimizer = tf.train.AdamOptimizer(learning_rate=decayed_lr)
        # optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)  # Optimizer (step decay)
        grads_and_vars = optimizer.compute_gradients(lenet.loss) # grad_ient 계산
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step) # back-propagation

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", lenet.loss)
        acc_summary = tf.summary.scalar("accuracy", lenet.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        sess.run(tf.global_variables_initializer()) # 모든 가중치 초기화

        def train_step(x_batch, y_batch):
            feed_dict = {
              lenet.X: x_batch,
              lenet.Y: y_batch,
              lenet.keep_prob: FLAGS.keep_prob,
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, lenet.loss, lenet.accuracy],
                feed_dict) # * hint learning rate decay operation 실행
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              lenet.X: x_batch,
              lenet.Y: y_batch,
              lenet.keep_prob: 1.0,
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, lenet.loss, lenet.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)
            return accuracy

        # Generate batches
        if FLAGS.data_augmentation: # data augmentation 적용시
            batches = dh.batch_iter_aug(x_train, y_train, FLAGS.batch_size, FLAGS.num_epochs)
        else:
            batches = dh.batch_iter(x_train, y_train, FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        max = 0
        start_time = time.time()

        for batch in batches: # len(batches) = (45000/batch size) * epoch 수
            x_batch, y_batch = zip(*batch) # batch size 단위로 input과 정답 리턴, e.g., (128, 32, 32, 3), (128, 10),
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            # if current_step == 352*200:
            #     FLAGS.lr *= 0.0005
            # if current_step == 352*350:
            #     FLAGS.lr = 0.0001
            # if current_step == 352*200:
            #     FLAGS.lr = 0.001
            if current_step % FLAGS.evaluate_every == 0: # 특정 iteration 마다
                print("\nEvaluation:")
                accuracy = dev_step(x_val, y_val, writer=dev_summary_writer) # validation accuracy 확인
                print("")
                if accuracy > max: # validation accuracy가 경신될 때
                    max = accuracy
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step) # best accuracy에 도달할 때만 모델을 저장함으로써 early stopping
                    print("Saved model checkpoint to {}\n".format(path))
        training_time = (time.time() - start_time) / 60
        print('Learning Finished!')
        print('Validation Max Accuracy:', max)
        print('Early stopped time:', math.ceil(current_step/704))
        print('training time: ', training_time)

