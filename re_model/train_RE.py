import logging
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

import tensorflow as tf
import pdb
import numpy as np
import network_RE
import utils_RE
import tqdm
from test_RE import main_test

from initial_RE import generate_train_npy_data, read_word_embedding
import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import config, utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--transfer', type=utils.str2bool)
parser.add_argument('--category', type=int)
parser.add_argument('--re_model_for_duplicate', type=utils.str2bool)
parser.add_argument('--train_re_gru_idx', type=int)
parser.add_argument('--test_re_gru_idx', type=int)

args = parser.parse_args()
transfer = args.transfer
category = 'memc'
if transfer:
    category = config.num_cat_dict[args.category]
    logging.info('category: ' + category)
model_for_duplicate = args.re_model_for_duplicate
train_re_gru_idx = args.train_re_gru_idx
test_re_gru_idx = args.test_re_gru_idx

logging.info('transfer: ' + str(transfer))
logging.info('re_model_for_duplicate: ' + str(model_for_duplicate))
logging.info('train_re_gru_idx: ' + str(train_re_gru_idx))
logging.info('test_re_gru_idx: ' + str(test_re_gru_idx))

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.gpu_options.visible_device_list = str(train_re_gru_idx)

# transfer = commons.re_model_transfer
# model_for_duplicate = commons.re_model_for_duplicate

save_path = config.re_model_path_before_transfer
if transfer:
    save_path = config.re_model_path_after_transfer
logging.info('model save path: ' + save_path)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('summary_dir', save_path, 'path to store summary')


def main(_):
    my_env = os.environ.copy()
    my_env["CUDA_VISIBLE_DEVICES"] = "1"
    # the path to save models

    logging.info('reading wordembedding')
    if not os.path.exists(config.vec_npy_path_and_name):
        read_word_embedding()
    wordembedding = np.load(config.vec_npy_path_and_name)
    
    logging.info('reading training data')
    train_y, train_char, train_word, train_pos1, train_pos2 = generate_train_npy_data(category)
    logging.info(str(train_y.shape) + ' ' + str(train_word.shape))
    # none_ind = re_utils.get_none_id(config.relation2id_file_path_and_name)
    # logging.info("None index: " + str(none_ind))
    settings = network_RE.Settings()
    settings.vocab_size = len(wordembedding)
    settings.num_classes = len(train_y[0])
    logging.info("vocab_size: " + str(settings.vocab_size))
    logging.info("num_classes: " + str(settings.num_classes))

    previous_step = 0
    previous_f1 = 0

    with tf.Graph().as_default():
        #config = tf.ConfigProto(device_count = {'GPU': 0})
        sess = tf.Session(config=tf_config)
        #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        with sess.as_default():

            initializer = tf.contrib.layers.xavier_initializer()
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                m = network_RE.GRU(is_training=True, word_embeddings=wordembedding, transfer=transfer, settings=settings)

            global_step = tf.Variable(0, name="global_step", trainable=False)
            # optimizer = tf.train.GradientDescentOptimizer(0.001)
            optimizer = tf.train.AdamOptimizer(0.001)

            # train_op=optimizer.minimize(m.total_loss,global_step=global_step)
            train_op = optimizer.minimize(m.final_loss, global_step=global_step)

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=None)
            

            # # # # # # # # transfer learning # # # # # # # #
            if transfer:
                saver = tf.train.Saver()
                saver.restore(sess, config.re_model_path_before_transfer + config.re_model_prefix  + "memc-" + utils_RE.get_model_list_from_re_model_dir(config.re_model_path_before_transfer, 'memc', True))
            # # # # # # # # transfer learning # # # # # # # #


            # merged_summary = tf.summary.merge_all()
            merged_summary = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/train_loss', sess.graph)

            def train_step(char_batch, word_batch, pos1_batch, pos2_batch, y_batch, big_num):
                
                # pdb.set_trace()
                # logging.info(str(len(char_batch)) + ' ' + str(len(word_batch)) + ' ' + str(len(pos1_batch)))
                feed_dict = {}
                total_shape = []
                total_num = 0

                total_char = []
                total_word = []
                total_pos1 = []
                total_pos2 = []
                for i_ in range(len(word_batch)):
                    total_shape.append(total_num)
                    total_num += len(word_batch[i_])
                    for char in char_batch[i_]:
                        # char = [np.array(j) for j in char]
                        total_char.append(char)
                    for word in word_batch[i_]:
                        total_word.append(word)
                    for pos1 in pos1_batch[i_]:
                        total_pos1.append(pos1)
                    for pos2 in pos2_batch[i_]:
                        total_pos2.append(pos2)

                total_shape.append(total_num)
                total_shape = np.array(total_shape)

                total_char = np.array(total_char)
                total_word = np.array(total_word)
                total_pos1 = np.array(total_pos1)
                total_pos2 = np.array(total_pos2)

                # logging.info('2: ' + str(total_shape) + str(total_char.shape) + str(total_word.shape) + str(total_pos1.shape) + str(total_pos2.shape))

                # logging.info(str(total_char.shape) + str(total_word.shape) + str(total_pos1.shape) + str(total_pos2.shape))
                # pdb.set_trace()

                # write_char(total_char)
                # write_word(total_word)

                feed_dict[m.total_shape] = total_shape
                feed_dict[m.input_char] = total_char
                feed_dict[m.input_word] = total_word
                feed_dict[m.input_pos1] = total_pos1
                feed_dict[m.input_pos2] = total_pos2
                feed_dict[m.input_y] = y_batch
                
                # pdb.set_trace()

                temp, step_, loss_, accuracy_, summary, l2_loss, final_loss = sess.run(
                    [train_op, global_step, m.total_loss, m.accuracy, merged_summary, m.l2_loss, m.final_loss],
                    feed_dict)
                accuracy_ = np.reshape(np.array(accuracy_), big_num)
                summary_writer.add_summary(summary, step_)
                return step_, loss_, accuracy_
            # training process
            for one_epoch in range(settings.num_epochs):
                logging.info("Starting Epoch: " + str( one_epoch))
                epoch_loss = 0
                temp_order = list(range(len(train_word)))
                np.random.shuffle(temp_order)

                all_true = []
                all_accuracy = []
                for i in tqdm.tqdm(range(int(len(temp_order) / float(settings.big_num)))):

                    temp_char = []
                    temp_word = []
                    temp_pos1 = []
                    temp_pos2 = []
                    temp_y = []

                    temp_input = temp_order[i * settings.big_num: (i + 1) * settings.big_num]
                    for k in temp_input:
                        # pdb.set_trace()
                        temp_char.append(train_char[k])
                        temp_word.append(train_word[k])
                        temp_pos1.append(train_pos1[k])
                        temp_pos2.append(train_pos2[k])
                        temp_y.append(train_y[k])
                    num = 0
                    for single_word in temp_word:
                        num += len(single_word)

                    if num > 1500:
                        logging.info('out of range')
                        continue

                    temp_char = np.array(temp_char)
                    temp_word = np.array(temp_word)
                    temp_pos1 = np.array(temp_pos1)
                    temp_pos2 = np.array(temp_pos2)
                    temp_y = np.array(temp_y)
                    
                    # logging.info('1: ' + str(temp_char.shape) + str(temp_word.shape) + str(temp_pos1.shape) + str(temp_pos2.shape))
                    step, loss, accuracy = train_step(temp_char, temp_word, temp_pos1, temp_pos2, temp_y, settings.big_num)
                    epoch_loss += loss
                    all_accuracy.append(accuracy)

                    all_true.append(temp_y)
                accu = np.mean(all_accuracy)
                logging.info("Epoch finished, loss:, " + str(epoch_loss) + "  accu: " + str(accu))

                if need_to_test(transfer, one_epoch):
                    logging.info('saving model')
                    current_step = tf.train.global_step(sess, global_step)
                    path = saver.save(sess, save_path + config.re_model_prefix + category, global_step=current_step)
                    logging.info(path)

                    current_f1 = test_current_step(current_step, save_path, model_for_duplicate=model_for_duplicate, test_re_gru_idx=test_re_gru_idx)
                    logging.info('current_f1 = ' + str(current_f1) + ', previous_f1 = ' + str(previous_f1))
                    if current_f1 >= previous_f1:
                        delete_re_model_from_disk(current_step, save_path)
                        previous_step = current_step
                        previous_f1 = current_f1
                    else:
                        logging.info('converge at step ' + str(previous_step))
                        delete_re_model_from_disk(previous_step, save_path)
                        return


def need_to_test(transfer, one_epoch):
    if one_epoch == 0:
         return False
    if transfer:
        if one_epoch % 5 == 0:
            return True
    return not transfer and one_epoch % 25 == 0


def test_current_step(current_step, model_path, model_for_duplicate=False, save_prediction=False, test_re_gru_idx=test_re_gru_idx):
    # test_cat = config.cat_list[0]
    test_cat = category
    f1, precision, recall, acc = main_test(current_step, category=test_cat, duplicate=model_for_duplicate, model_path=model_path, save_prediction=save_prediction, test_re_gru_idx=test_re_gru_idx, transfer=transfer)
    return f1


def delete_re_model_from_disk(save_model_id, model_path):
    file_list = os.listdir(model_path)
    for f in file_list:
        if not f.startswith(config.re_model_prefix + category):
            continue
        loc1 = f.find('-')
        loc2 = f.find('.')
        model_id = f[loc1+1:loc2]
        if int(model_id) != save_model_id:
            os.remove(model_path + f)
            logging.info(f + ' is removed from path: ' + model_path)


def write_char(total_char):
    np.save('char_file.npy', total_char)


def write_word(total_word):
    np.save('word_file.npy', total_word)


if __name__ == '__main__':
    tf.app.run()
