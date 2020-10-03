import tensorflow as tf
import os
import time
import datetime
from math import ceil
from tensorflow.keras.datasets.cifar10 import load_data
import data_helpers as dh
from resnet import ResNet
from settings import settings

# 에러 무시 코드
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 결과 시트
results = {}

# 플래그 초기 함수
def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)


# Choose start setting numbers
set_start = 1
set_end = 3

for i in range(set_start, set_end + 1):
    setting = settings[f'SET#{i}']

    # 플래그, 그래프 초기화
    del_all_flags(tf.flags.FLAGS)
    tf.reset_default_graph()

    # 플래그 오류
    tf.flags.DEFINE_string("f", "", "kernel")

    # Model Hyperparameters
    tf.flags.DEFINE_float("lr", setting['LR'], "learning rate (default=0.1)")
    tf.flags.DEFINE_float("lr_decay", setting['LR_DECAY'], "learning rate decay rate(default=0.1)")
    tf.flags.DEFINE_float("l2_reg_lambda", setting['L2_REG_LAMBDA'], "L2 regularization lambda (default: 0.0)")
    tf.flags.DEFINE_float("relu_leakiness", setting['RELU_LEAKINESS'], "relu leakiness (default: 0.1)")
    tf.flags.DEFINE_integer("num_residual_units", setting['NUM_RESIDUAL_UNITS'],
                            "The number of residual_units (default: 5)")
    tf.flags.DEFINE_integer("num_classes", 10, "The number of classes (default: 10)")
    tf.flags.DEFINE_string("weight_init", setting['WEIGHT_INIT'], "Weight initialization type (default: He)")

    # Training parameters
    tf.flags.DEFINE_integer("batch_size", setting['BATCH_SIZE'], "Batch Size (default: 64)")
    tf.flags.DEFINE_integer("num_epochs", setting['NUM_EPOCHS'], "Number of training epochs (default: 200)")
    tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
    tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
    tf.flags.DEFINE_integer("num_checkpoints", 3, "Number of checkpoints to store (default: 3)")
    tf.flags.DEFINE_boolean("data_augmentation", setting['DATA_AUGMENTATION'], "data augmentation option")

    # Optimizer
    tf.flags.DEFINE_string("optimizer", setting['OPTIMIZER'], "The optimizer (default: Momentum)")

    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    FLAGS = tf.flags.FLAGS

    (x_train_val, y_train_val), (x_test, y_test) = load_data()
    x_train, y_train, x_test, y_test, x_val, y_val = dh.shuffle_data(x_train_val, y_train_val, x_test, y_test,
                                                                     FLAGS.num_classes)

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            resnet = ResNet(FLAGS)  # ResNet 클래스의 인스턴스 생성 후 Hyperparameter가 정의돼 있는 FLAGS로 초기화

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            decayed_lr = tf.train.exponential_decay(FLAGS.lr, global_step, 24000, FLAGS.lr_decay, staircase=True)
            if FLAGS.optimizer == "ADAM":
                optimizer = tf.train.AdamOptimizer(learning_rate=decayed_lr)
            else:
                optimizer = tf.train.MomentumOptimizer(learning_rate=decayed_lr, momentum=0.9)
            grads_and_vars = optimizer.compute_gradients(resnet.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            train_ops = [train_op] + resnet.extra_train_ops
            train_ops = tf.group(*train_ops)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", resnet.loss)
            acc_summary = tf.summary.scalar("accuracy", resnet.accuracy)

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

            sess.run(tf.global_variables_initializer())


            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    resnet.X: x_batch,
                    resnet.Y: y_batch
                }
                _, step, lr, summaries, loss, accuracy = sess.run(
                    [train_ops, global_step, decayed_lr, train_summary_op, resnet.loss, resnet.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, lr {}, loss {:g}, acc {:g}".format(time_str, step, lr, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)


            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    resnet.X: x_batch,
                    resnet.Y: y_batch
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, resnet.loss, resnet.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)
                return accuracy


            # Generate batches
            if FLAGS.data_augmentation:  # data augmentation 적용시
                batches = dh.batch_iter_aug(x_train, y_train, FLAGS.batch_size, FLAGS.num_epochs)
            else:
                batches = dh.batch_iter(x_train, y_train, FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            max = 0
            start_time = time.time()
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    accuracy = dev_step(x_val, y_val, writer=dev_summary_writer)
                    print("")
                    if accuracy > max:
                        max = accuracy
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))

            result = {}
            result['training_time'] = training_time = (time.time() - start_time) / 60
            result['early_stopping_epoch'] = ceil(current_step / ceil(45000 / setting['BATCH_SIZE']))
            result['val_max'] = max
            results[f'Setting#{i}'] = result

for setting, result in results.items():
    print(f'< {setting} result >')
    print(' - Training time: ', result['training_time'])
    print(' - Early stopping epoch:', result['early_stopping_epoch'])
    print(' - Validation Max Accuracy:', result['val_max'])
    print()
