#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import smart_open
import pickle
from text_classification_master.text_cnn import TextCNN
import text_classification_master.data_helpers as dh

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("mr_test_file_pos", "./data/MR/rt-polarity_test.pos", "Data source for the ODP training")
tf.flags.DEFINE_string("mr_test_file_neg", "./data/MR/rt-polarity_test.neg", "Data source for the ODP training")
tf.flags.DEFINE_string("task", "MR", "Choose the classification task")

# Eval Parameters
tf.flags.DEFINE_string("dir", "./runs/1604643104", "Checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS


if FLAGS.task == "MR":
    x_raw, y_test = dh.load_mr_data(FLAGS.mr_test_file_pos, FLAGS.mr_test_file_neg)
elif FLAGS.task == "TREC":
    x_raw, y_test = dh.load_trec_data(FLAGS.x_test_file, FLAGS.t_test_file)
y_test = np.argmax(y_test, axis=1) # 0, 0, 0, 1 -> 3

# Map data into vocabulary
with smart_open.smart_open(os.path.join(FLAGS.dir, "vocab"), 'rb') as f:
    word_id_dict = pickle.load(f)
with smart_open.smart_open(os.path.join(FLAGS.dir, "config"), 'rb') as f:
    config = pickle.load(f)

    x_test = dh.text_to_index(x_raw, word_id_dict, max(list(map(int, config["filter_sizes"].split(",")))) - 1)
    x_test = dh.test_tensor(x_test, config["max_length"])

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(os.path.join(FLAGS.dir, "checkpoints"))
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(config)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, checkpoint_file)

        # Generate batches for one epoch
        batches = dh.batch_iter(list(x_test), config["batch_size"], 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(cnn.predictions, {cnn.input_x: x_test_batch, cnn.dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    print(len(all_predictions))
    print(len(y_test))
    print(sum(all_predictions == y_test))
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))