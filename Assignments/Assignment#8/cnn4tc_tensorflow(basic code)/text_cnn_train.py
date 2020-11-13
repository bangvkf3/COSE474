import tensorflow as tf
import numpy as np
import os
import time
import datetime
import re
import smart_open
import pickle
import text_classification_master.data_helpers as dh
from text_classification_master.text_cnn import TextCNN
from gensim.models.keyedvectors import KeyedVectors

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("trec_train_file", "./data/TREC/traindata.txt", "Data source for the training")
tf.flags.DEFINE_string("mr_train_file_pos", "./data/MR/rt-polarity.pos", "Data source for the MR training")
tf.flags.DEFINE_string("mr_train_file_neg", "./data/MR/rt-polarity.neg", "Data source for the MR training")

tf.flags.DEFINE_string("word2vec", "./data/GoogleNews-vectors-negative300.bin", "Word2vec file with pre-trained embeddings (default: None)")
tf.flags.DEFINE_string("task", "MR", "Choose the classification task")

# Model Hyperparameters
tf.flags.DEFINE_integer("vocab_size", 30000, "Vocabulary size (defualt: 0)")
tf.flags.DEFINE_integer("num_classes", 0, "The number of labels (defualt: 0)")
tf.flags.DEFINE_integer("max_length", 0, "max sequence length (defualt: 0)")
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0001, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("lr_decay", 0.99, "Learning rate decay rate (default: 0.98)")
tf.flags.DEFINE_float("lr", 1e-1, "Learning rate(default: 0.01)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 50, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 3, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

def preprocess():
    # Load data
    print("Loading data...")
    if FLAGS.task == "MR":
        x_text, y = dh.load_mr_data(FLAGS.mr_train_file_pos, FLAGS.mr_train_file_neg)
    elif FLAGS.task == "TREC":
        x_text, y = dh.load_trec_data(FLAGS.trec_train_file)
        pass # TREC data 전처리 구현

    # Build vocabulary
    word_id_dict, _ = dh.buildVocab(x_text, FLAGS.vocab_size) # training corpus를 토대로 단어사전 구축
    FLAGS.vocab_size = len(word_id_dict) + 4 #30000 + 4
    print("vocabulary size: ", FLAGS.vocab_size)

    for word in word_id_dict.keys():
        word_id_dict[word] += 4  # <pad>: 0, <unk>: 1, <s>: 2 (a: 0 -> 4)
    word_id_dict['<pad>'] = 0 # zero padding을 위한 토큰
    word_id_dict['<unk>'] = 1 # OOV word를 위한 토큰
    word_id_dict['<s>'] = 2 # 문장 시작을 알리는 start 토큰
    word_id_dict['</s>'] = 3 # 문장 마침을 알리는 end 토큰

    x = dh.text_to_index(x_text, word_id_dict, max(list(map(int, FLAGS.filter_sizes.split(",")))) - 1) # i am a boy, word_id_dict, max([3,4,5]) -> 5 - 1
    x, FLAGS.max_length = dh.train_tensor(x) # 문장 max length를 바탕으로 batch 구성

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/dev set
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    FLAGS.num_classes = y_train.shape[1] # 2 (eg., [0, 1]), class 개수를 y shape로 부터 획득

    del x, x_text, y, x_shuffled, y_shuffled
    print(x_train)
    print(y_train)

    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, word_id_dict, x_dev, y_dev

def train(x_train, y_train, word_id_dict, x_dev, y_dev):
    # Training
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(FLAGS.flag_values_dict())

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            # lr decay
            decayed_lr = tf.train.exponential_decay(FLAGS.lr, global_step, 1000, FLAGS.lr_decay, staircase=True)
            optimizer = tf.train.AdadeltaOptimizer(decayed_lr)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
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

            # Write vocabulary, configuration
            with smart_open.smart_open(os.path.join(out_dir, "vocab"), 'wb') as f:
                pickle.dump(word_id_dict, f)
            with smart_open.smart_open(os.path.join(out_dir, "config"), 'wb') as f:
                pickle.dump(FLAGS.flag_values_dict(), f)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            if FLAGS.word2vec: # word2vec 활용 시
                print("Loading W2V data...")
                pre_emb = KeyedVectors.load_word2vec_format(FLAGS.word2vec, binary=True) #pre-trained word2vec load
                pre_emb.init_sims(replace=True)
                num_keys = len(pre_emb.vocab)
                print("loaded word2vec len ", num_keys)

                # initial matrix with random uniform, pretrained word2vec으로 vocabulary 내 단어들을 초기화하기 위핸 weight matrix 초기화
                initW = np.random.uniform(-0.25, 0.25, (FLAGS.vocab_size, FLAGS.embedding_dim))
                # load any vectors from the word2vec
                print("init initW cnn.W in FLAG")
                for w in word_id_dict.keys():
                    arr = []
                    s = re.sub('[^0-9a-zA-Z]+', '', w)
                    if w in pre_emb: # 직접 구축한 vocab 내 단어가 google word2vec에 존재하면
                        arr = pre_emb[w] # word2vec vector를 가져옴
                    elif w.lower() in pre_emb: # 소문자로도 확인
                        arr = pre_emb[w.lower()]
                    elif s in pre_emb: # 전처리 후 확인
                        arr = pre_emb[s]
                    elif s.isdigit(): # 숫자이면
                        arr = pre_emb['1']
                    if len(arr) > 0: # 직접 구축한 vocab 내 단어가 google word2vec에 존재하면
                        idx = word_id_dict[w] # 단어 index
                        initW[idx] = np.asarray(arr).astype(np.float32) # 적절한 index에 word2vec word 할당
                print("assigning initW to cnn. len=" + str(len(initW)))
                sess.run(cnn.W.assign(initW)) # initW를 cnn.W에 할당

            def train_step(x_batch, y_batch):
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, lr, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, decayed_lr, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, lr{:g}, acc {:g}".format(time_str, step, loss, lr, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)
                return accuracy

            # Generate batches
            batches = dh.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            max = 0
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    accuracy = dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                    if accuracy > max:
                        max = accuracy
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))

def main(argv=None):
    x_train, y_train, word_id_dict, x_dev, y_dev = preprocess()
    train(x_train, y_train, word_id_dict, x_dev, y_dev)

if __name__ == '__main__':
    tf.app.run()