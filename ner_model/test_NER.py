import logging
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import config, utils
from utils_NER import get_ner_model_from_dir, evaluate_each_class
from initial_NER import generate_line_dict_from_ner_data, read_word2embedding, \
    read_char_data, read_data, create_word_index, create_char_index

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--gru', type=int)
parser.add_argument('--duplicate', type=utils.str2bool)
parser.add_argument('--gazetteer', type=utils.str2bool)
parser.add_argument('--labeled', type=utils.str2bool)
parser.add_argument('--case_idx', type=str)
parser.add_argument('--cat_num', type=int)

args = parser.parse_args()

gru = args.gru
duplicate = args.duplicate
gazetteer = args.gazetteer
labeled = args.labeled
case_idx = args.case_idx
cat_num = args.cat_num

os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda" + str(gru) + ",floatX=float32"
import network_NER


def test_ner():
    category = config.num_cat_dict[cat_num]

    word_index, word_cnt = create_word_index([config.hash_file])
    TRAIN_DATA = config.labeled_ner_data_input_path + category + '_train' + config.data_suffix
    # TRAIN_DATA = include_valid_set()
    TEST_DATA = config.labeled_ner_data_input_path + category + '_test' + config.data_suffix

    wx, y, m = read_data(TRAIN_DATA, word_index)
    twx, ty, tm = read_data(TEST_DATA, word_index)
    char_index, char_cnt = create_char_index([config.hash_file])

    x, cm = read_char_data(TRAIN_DATA, char_index)
    tx, tcm = read_char_data(TEST_DATA, char_index)
    gaze, tgaze = None, None

    transfer = not (category == 'memc')

    model_name = get_ner_model_from_dir(config.ner_model_path, category)
    logging.info('test model is ' + model_name)

    test_cat_list = [category]
    # if transfer:
    #     test_cat_list = config.cat_list[1:]

    # save_path = config.ner_model_path_before_transfer
    # if transfer:
    #     save_path = config.ner_model_path_after_transfer

    test_flg = None
    ner_data_input_path = None
    output_path = None

    labeled = True
    if labeled:
        test_flg = '_test'
        ner_data_input_path = config.labeled_ner_data_input_path
        output_path = config.labeled_ner_data_output_after_transfer_path
        if not transfer:
            output_path = config.labeled_ner_data_output_before_transfer_path
        if duplicate:
            test_flg += '_duplicate'
    # elif not labeled:
    #     test_flg = '_full'
    #     if duplicate:
    #         test_flg += '_duplicate'
    #     ner_data_input_path = commons.unlabeled_ner_data_input_path
    #     output_path = commons.unlabeled_ner_data_output_path
    #     # test_cat_list = commons.cat_list[9:]
    #     if cat_num is None:
    #         test_cat_list = separate_unlabeled_ner_input_data(ner_data_input_path,
    #                                                           commons.cat_list[1:8] + commons.cat_list[9:], test_flg)
    #     else:
    #         test_cat_list = separate_unlabeled_ner_input_data(ner_data_input_path, [commons.cat_list[cat_num]],
    #                                                           test_flg)
    # else:
    #     print('ERROR')
    #     return

    # f_test_result_name = output_path + test_flg + '.txt'
    '''
    result_name = utils.get_f_result_name(case_idx, 'test', transfer, labeled, duplicate, gazetteer,
                                            neroutput=False, ner=True)
    f_test_result_name = config.sh_output_path + result_name

    logging.info('test result is saved in ' + f_test_result_name)
    with open(f_test_result_name, 'a') as f_test_result:
        f_test_result.write('\n\nmodel name: ' + model_name + '\n')
        # f_test_result.write('model notes: ' + str(model_notes) + '\n')
        f_test_result.write('model notes: ' + result_name + '\n')
        f_test_result.write(utils.format_str(
            ['category', 'acc', 'prec_version', 'recall_version', 'prec_software', 'recall_software']) + '\n')
    '''
    logging.info(test_cat_list)
    for category in test_cat_list:
        # for category in ['csrf', 'xss']:
        logging.info(category)

        file_idx = None
        if type(category) == list:
            category, file_idx = category

        SMALL_DATA_NAME = category + test_flg + config.data_suffix

        if file_idx is not None:
            SMALL_DATA_NAME += '_' + str(file_idx)

        SMALL_DATA = ner_data_input_path + SMALL_DATA_NAME
        test_data_dic = generate_line_dict_from_ner_data(SMALL_DATA)
        SMALL_DATA_PD_NOGAZE = output_path + SMALL_DATA_NAME
        SMALL_DATA_PD_GAZE = SMALL_DATA_PD_NOGAZE.replace(config.data_suffix, '_gaze' + config.data_suffix)
        SMALL_DATA_PD = SMALL_DATA_PD_GAZE if gazetteer else SMALL_DATA_PD_NOGAZE

        tx, tcm = read_char_data(SMALL_DATA, char_index)
        twx, ty, tm = read_data(SMALL_DATA, word_index)

        py = None

        if not gazetteer:
            logging.info('reading data from ' + SMALL_DATA)

            # tx, tcm = read_char_data(SMALL_DATA, char_index)
            # twx, ty, tm = read_data(SMALL_DATA, word_index)
            logging.info('building model ...')
            model = network_NER.cnn_rnn(char_cnt, len(config.labels), word_cnt, config.ner_model_path, category, use_crf=True)
            model.add_data(x, y, m, wx, cm, gaze, tx, ty, tm, twx, tcm, tgaze, test_data_dic)
            model.build()

            model.load_params(config.ner_model_path + model_name)
            # model_bkp = deepcopy(model)
            logging.info('finish loading, predicting starts...')

            py, py_score = model.predict(tx, tm, twx, tcm, tgaze=None, reload_model_path=config.ner_model_path + model_name)

            word2embedding = read_word2embedding()

            logging.info("set word embeddings...")
            model.set_embedding(word2embedding, word_index)

            # if 'crf' not in str(model_notes):
            #     ttt = 0.5
            #     py = fast(py_score, axis=1, threshold=float(ttt))
            #     logging.info('prediction is saved in ' + SMALL_DATA_PD)
        else:
            logging.info('applying gaze on no-gaze prediction result path :' + SMALL_DATA_PD)
            _, py, _ = read_data(SMALL_DATA_PD_NOGAZE, word_index)
        extracted_f = open(SMALL_DATA_PD, 'w')
        # pdb.set_trace()
        acc, f1, metric_result = evaluate_each_class(py, test_data_dic, ty, tm, apply_rule=gazetteer,
                                                     file_result=extracted_f, debug=False)
        extracted_f.close()
        logging.info(str(acc) + ' ' + str(f1) + ' ' + str(metric_result))

        str_write = utils.format_str(
            [category, float(acc), metric_result['V'][3], metric_result['V'][4],
             metric_result['N'][3], metric_result['N'][4]])
        logging.info(str_write)
        '''
        with open(f_test_result_name, 'a') as f_test_result:
            f_test_result.write(str_write + '\n')
        '''

if __name__ == '__main__':
    test_ner()