
import logging
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import config, utils

import argparse
parser = argparse.ArgumentParser()

from utils_NER import get_ner_model_from_dir, evaluate_each_class
from initial_NER import generate_line_dict_from_ner_data, read_word2embedding, \
    read_char_data, read_data, create_word_index, create_char_index
import network_NER

parser.add_argument('--cat_num', type=int)
parser.add_argument('--gru', type=int)

# fine tune

parser.add_argument('--batch_size', type=int)
parser.add_argument('--dropout_rate', type=float)
parser.add_argument('--char_hidden_size', type=int)
parser.add_argument('--hidden_size', type=int)
parser.add_argument('--double_layer', type=utils.str2bool)
parser.add_argument('--char_double_layer', type=utils.str2bool)
parser.add_argument('--char_rnn', type=utils.str2bool)

args = parser.parse_args()

cat_num = args.cat_num
gru = args.gru

batch_size = args.batch_size
dropout_rate = args.dropout_rate
char_hidden_size = args.char_hidden_size
hidden_size = args.hidden_size
double_layer = args.double_layer
char_double_layer = args.char_double_layer
char_rnn = args.char_rnn

os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda" + str(gru) + ",floatX=float32"


def train_ner():
    category = config.num_cat_dict[cat_num]

    TRAIN_DATA = config.labeled_ner_data_input_path + category + '_train' + config.ner_data_suffix
    # TRAIN_DATA = include_valid_set()
    TEST_DATA = config.labeled_ner_data_input_path + category + '_test' + config.ner_data_suffix

    transfer = not (category == 'memc')
    save_path = config.ner_model_path_before_transfer
    if transfer:
        save_path = config.ner_model_path_after_transfer

    logging.info('batch_size: ' + str(batch_size))
    logging.info('dropout_rate: ' + str(dropout_rate))
    logging.info('char_hidden_size: ' + str(char_hidden_size))
    logging.info('hidden_size: ' + str(hidden_size))
    logging.info('double_layer: ' + str(double_layer))
    logging.info('char_double_layer: ' + str(char_double_layer))
    logging.info('char_rnn: ' + str(char_rnn) + '\n\n\n')

    logging.info('train data: ' + TRAIN_DATA)
    logging.info('test data: ' + TEST_DATA)
    logging.info("read data...")

    word_index, word_cnt = create_word_index([config.hash_file])
    wx, y, m = read_data(TRAIN_DATA, word_index)
    twx, ty, tm = read_data(TEST_DATA, word_index)
    char_index, char_cnt = create_char_index([config.hash_file])

    x, cm = read_char_data(TRAIN_DATA, char_index)
    tx, tcm = read_char_data(TEST_DATA, char_index)
    gaze, tgaze = None, None

    model = None
    if None in [batch_size, dropout_rate, char_hidden_size, hidden_size, double_layer, char_double_layer, char_rnn]:
        model = network_NER.cnn_rnn(char_cnt, len(config.labels), word_cnt, save_path, category, use_crf=True,
                                    transfer=transfer
                                    )
    else:
        model = network_NER.cnn_rnn(char_cnt, len(config.labels), word_cnt, save_path, category, use_crf=True,
                                    batch_size=batch_size, dropout_rate=dropout_rate,
                                    char_hidden_size=char_hidden_size, hidden_size=hidden_size,
                                    double_layer=double_layer, char_double_layer=char_double_layer, char_rnn=char_rnn,
                                    transfer=transfer
                                    )

    test_data_dic = generate_line_dict_from_ner_data(TEST_DATA)

    logging.info("add data...")
    model.add_data(x, y, m, wx, cm, gaze, tx, ty, tm, twx, tcm, tgaze, test_data_dic)

    logging.info("build the model...")
    model.build()
    
    if transfer:
        model_name = get_ner_model_from_dir(config.ner_model_path_before_transfer)
        logging.info('teacher model is ' + model_name)
        model.load_params(config.ner_model_path_before_transfer + model_name)
    
    word2embedding = read_word2embedding()

    logging.info("set word embeddings...")
    model.set_embedding(word2embedding, word_index)

    model.train(evaluate_each_class)


if __name__ == '__main__':
    train_ner()
