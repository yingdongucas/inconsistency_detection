import tensorflow as tf
import pdb
import numpy as np
import network_RE
import utils_RE

import logging
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

from initial_RE import generate_test_npy_data
import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import config, utils

# from dataset_preparation.test_new_category import complete_version_dict_from_dict, write_version_dict_to_py_file
from data_preparation import generate_prediction_and_gt_table
# from measurement.clean_version_and_measure import clean_version_dict, measure_by_year_os_software, measure_by_category

import argparse


def main_test(test_model_id, category, duplicate, model_path=config.re_model_path,
              save_prediction=False, test_gru='0', return_pd_and_gt_version_dict=False,
              return_pd_version_dict=False,
              output_path=config.labeled_re_data_write_path,
              f_test_result_name=None,
              generate_from_ner_output_path=None,
              labeled=True,
              category_separate_idx=None,
              gaze=False,
              transfer=False,
              ):
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.gpu_options.visible_device_list = str(test_gru)

    FLAGS = tf.app.flags.FLAGS

    test_model_id = str(test_model_id)
    # ATTENTION: change pathname before you load your model

    # pathname = model_path + "ATT_GRU_model-"
    pathname = model_path + config.re_model_prefix + category + "-"
    if not transfer:
        pathname = model_path + config.re_model_prefix + 'memc' + "-"
    # pdb.set_trace()
    logging.info(pathname)
    # none_ind = re_utils.get_none_id(config.relation2id_file_path_and_name)
    # logging.info("None index: " + str(none_ind))

    wordembedding = np.load(config.vec_npy_path_and_name)

    test_y, test_char, test_word, test_pos1, test_pos2, test_txt_file_path_and_name, test_data_name = generate_test_npy_data(
        category, duplicate=duplicate, generate_from_ner_output_path=generate_from_ner_output_path,
        category_separate_idx=category_separate_idx, gaze=gaze)

    logging.info('test_txt_file_path_and_name: ' + test_txt_file_path_and_name)
    test_settings = network_RE.Settings()
    test_settings.vocab_size = len(wordembedding)
    test_settings.big_num = len(test_y)
    if test_settings.big_num == 0:
        return None

    test_settings.num_classes = len(test_y[0])

    with tf.Graph().as_default():

        print('1 ****************************')
        sess = tf.Session(config=tf_config)
        with sess.as_default():

            def test_step(char_batch, word_batch, pos1_batch, pos2_batch, y_batch):
                feed_dict = {}
                total_shape = []
                total_num = 0

                total_char = []
                total_word = []
                total_pos1 = []
                total_pos2 = []

                for j in range(len(word_batch)):
                    total_shape.append(total_num)
                    total_num += len(word_batch[j])
                    for char in char_batch[j]:
                        # char = [np.array(j) for j in char]
                        total_char.append(char)
                    for word in word_batch[j]:
                        total_word.append(word)
                    for pos1 in pos1_batch[j]:
                        total_pos1.append(pos1)
                    for pos2 in pos2_batch[j]:
                        total_pos2.append(pos2)

                total_shape.append(total_num)
                total_shape = np.array(total_shape)

                total_char = np.array(total_char)
                total_word = np.array(total_word)
                total_pos1 = np.array(total_pos1)
                total_pos2 = np.array(total_pos2)

                feed_dict[mtest.total_shape] = total_shape
                feed_dict[mtest.input_char] = total_char
                feed_dict[mtest.input_word] = total_word
                feed_dict[mtest.input_pos1] = total_pos1
                feed_dict[mtest.input_pos2] = total_pos2
                feed_dict[mtest.input_y] = y_batch

                loss, accuracy_, predictions = sess.run(
                    [mtest.loss, mtest.accuracy, mtest.predictions], feed_dict)

                return predictions, accuracy_

            with tf.variable_scope("model"):
                mtest = network_RE.GRU(is_training=False, word_embeddings=wordembedding, transfer=False, settings=test_settings)

            saver = tf.train.Saver()

            # ATTENTION: change the list to the iters you want to test !!

            # testlist = range(9025,14000,25)

            testlist = [test_model_id]

            logging.info('00000 ****************************')
            for model_iter in testlist:

                logging.info('00 ****************************')
                saver.restore(sess, pathname + model_iter)

                all_pred = []
                all_true = []
                all_accuracy = []

                for i in range(int(len(test_word) / float(test_settings.big_num))):
                    logging.info('0 ****************************')
                    pred, accuracy = test_step(test_char[i * test_settings.big_num:(i + 1) * test_settings.big_num:],
                                               test_word[i * test_settings.big_num:(i + 1) * test_settings.big_num],
                                               test_pos1[i * test_settings.big_num:(i + 1) * test_settings.big_num],
                                               test_pos2[i * test_settings.big_num:(i + 1) * test_settings.big_num],
                                               test_y[i * test_settings.big_num:(i + 1) * test_settings.big_num])
                    pred = np.array(pred)
                    all_pred.append(pred)
                    all_true.append(test_y[i * test_settings.big_num:(i + 1) * test_settings.big_num])
                    all_accuracy.append(accuracy)
                all_pred = np.concatenate(all_pred, axis=0)
                all_true = np.concatenate(all_true, axis=0)
                all_true_inds = np.argmax(all_true, 1)

                pd_version_dict = generate_prediction_and_gt_table(test_txt_file_path_and_name, relations=all_pred)
                if labeled:
                    precision_gt, recall_gt, f1_gt, acc_gt, \
                    precision_all, recall_all, f1_all, acc_all, gt_version_dict  = \
                        0, 0, 0, 0, 0, 0, 0, 0, {}
                    gt_version_dict = generate_prediction_and_gt_table(config.labeled_re_data_input_path + test_data_name.replace('_gaze', ''))
                    '''
                    if generate_from_ner_output_path is None:
                        precision_gt, recall_gt, f1_gt, acc_gt = utils.evaluate_rm_neg(all_pred, all_true_inds, none_ind, with_acc=True)
                    else:
                        precision_gt, recall_gt, f1_gt, acc_gt, precision_all, recall_all, f1_all, acc_all, = compute_re_prediction_on_ner_output_with_gt(
                            test_txt_file_path_and_name, all_pred, commons.labeled_re_origin_data_input_path + test_data_name)
                    '''
                    precision_gt, recall_gt, f1_gt, acc_gt, precision_all, recall_all, f1_all, acc_all, = utils_RE.compute_re_prediction_on_ner_output_with_gt(test_txt_file_path_and_name, all_pred, config.labeled_re_data_input_path + test_data_name.replace('_gaze', ''))


                    logging.info('Accu = %.4f, F1 = %.4f, recall = %.4f, precision = %.4f)' % (acc_gt, f1_gt, recall_gt, precision_gt))

                    if return_pd_and_gt_version_dict:
                        return pd_version_dict, gt_version_dict, \
                               precision_gt, recall_gt, f1_gt, acc_gt, \
                               precision_all, recall_all, f1_all, acc_all
                    return f1_all, precision_all, recall_all, acc_all
                else:
                    if return_pd_version_dict:
                        return pd_version_dict


def test_re():
    parser = argparse.ArgumentParser()

    parser.add_argument('--transfer', type=utils.str2bool)
    parser.add_argument('--duplicate', type=utils.str2bool)
    parser.add_argument('--gru', type=int)
    parser.add_argument('--neroutput', type=utils.str2bool)
    # parser.add_argument('--model_notes', type=str)
    parser.add_argument('--labeled', type=utils.str2bool)

    parser.add_argument('--operation', type=str)
    parser.add_argument('--gazetteer', type=utils.str2bool)
    parser.add_argument('--re', type=utils.str2bool)
    parser.add_argument('--category', type=int)

    parser.add_argument('--case_idx', type=str)

    args = parser.parse_args()

    # transfer = args.transfer
    test_data_duplicate = args.duplicate
    test_gru = args.gru
    test_ner_output_not_gt = args.neroutput
    # model_notes = args.model_notes
    labeled = args.labeled
    operation = args.operation
    gazetteer = args.gazetteer
    re = args.re
    category_ = config.num_cat_dict[args.category]
    case_idx = args.case_idx

    transfer = not (category_ == 'memc')

    logging.info('transfer: ' + str(transfer))
    logging.info('test_data_duplicate: ' + str(test_data_duplicate))
    logging.info('test_gru: ' + str(test_gru))
    logging.info('test_ner_output_not_gt: ' + str(test_ner_output_not_gt))
    # logging.info('model_notes: ' + str(model_notes))
    logging.info('labeled: ' + str(labeled))
    logging.info('operation: ' + str(operation))
    logging.info('gazetteer: ' + str(gazetteer))
    logging.info('re: ' + str(re))
    logging.info('category: ' + str(category_))
    logging.info('case_idx: ' + str(case_idx))

    # configure ner output path
    generate_from_ner_output_path = None
    # if labeled:
    #     if test_ner_output_not_gt:
    #         generate_from_ner_output_path = config.labeled_ner_data_output_before_transfer_path
    #         if transfer:
    #             generate_from_ner_output_path = config.labeled_ner_data_output_after_transfer_path
    # else:
    #     generate_from_ner_output_path = config.unlabeled_ner_data_output_path

        # if os.path.exists(commons.unlabeled_version_dict_file_path_and_name):
        #     copy(commons.unlabeled_version_dict_file_path_and_name,
        #          commons.unlabeled_version_dict_file_path_and_name.replace('.py', '_bkp.py'))
        #     os.remove(commons.unlabeled_version_dict_file_path_and_name)

    if test_ner_output_not_gt:
        generate_from_ner_output_path = config.labeled_ner_data_output_before_transfer_path
        if transfer:
            generate_from_ner_output_path = config.labeled_ner_data_output_after_transfer_path

    # configure model path
    model_path = config.re_model_path
    # if transfer:
    #     model_path = config.re_model_path_after_transfer
    # else:
    #     model_path = config.re_model_path_before_transfer

    # configure test cats, output path
    test_cat_list = None
    output_path = None
    test_flg = None

    # if labeled:
    test_flg = '_test'
    test_cat_list = config.cat_list
    # test_cat_list = ['memc']
    output_path = config.labeled_re_data_output_before_transfer_path

    if transfer:
        test_cat_list = config.cat_list[1:]
        output_path = config.labeled_re_data_output_after_transfer_path

    # else:
    #     test_flg = '_full'
    #     output_path = commons.unlabeled_re_data_output_path

    if test_data_duplicate:
        test_flg += '_duplicate'

    if gazetteer:
        test_flg += '_gaze'

    # if not labeled:
    #     if cat_num is None:
    #         test_cat_list = get_separate_cat_file_list(generate_from_ner_output_path, commons.cat_list[1:8] + commons.cat_list[9:], test_flg)
    #     else:
    #         test_cat_list = get_separate_cat_file_list(generate_from_ner_output_path, [commons.cat_list[cat_num]], test_flg)

    if generate_from_ner_output_path is not None:
        test_flg += '_neroutput'

    # f_test_result_name = output_path + test_flg + '.txt'
    result_name = utils.get_f_result_name(case_idx, operation, transfer, labeled, test_data_duplicate, gazetteer, neroutput=test_ner_output_not_gt, re=re)
    f_test_result_name = config.sh_output_path + result_name
    logging.info('test result is saved in ' + f_test_result_name)


    title_list = ['category',
                  'gt',
                  'gt_acc', 'gt_precision', 'gt_recall',
                  'all',
                  'all_acc', 'all_precision', 'all_recall',
                  'rate',
                  'pd_rate', 'deviation', 'norm_deviation',
                  'complete_rate',
                  'pd_complete_rate', 'deviation_complete', 'norm_deviation_complete'
                  ]
    str_write = utils.format_str(title_list) + '\n'

    all_pd_dictionary_complete, merged_pd_dictionary_complete = dict(), dict()
    measurement_performance_str_all_cat = ''

    # for category in test_cat_list:
    for category in ['dirtra']:
        file_idx = None
        if type(category) == list:
            category, file_idx = category
        model_id = utils_RE.get_model_list_from_re_model_dir(model_path, category, transfer)
        logging.info('category: ' + category)
        logging.info('model id: ' + model_id)
        test_result = main_test(model_id, category=category, duplicate=test_data_duplicate, model_path=model_path,
                                save_prediction=False, test_gru=test_gru,
                                return_pd_and_gt_version_dict=labeled,
                                return_pd_version_dict=not labeled,
                                output_path=output_path,
                                generate_from_ner_output_path=generate_from_ner_output_path,
                                labeled=labeled,
                                category_separate_idx=file_idx,
                                gaze=gazetteer,
                                transfer=transfer,
                                )
        if test_result is None:
            logging.info(category + ' re data is none!')
            continue
        pd_dictionary = None
        if labeled:
            pd_dictionary, gt_dictionary, \
            prec_gt_, rec_gt_, f1_gt_, acc_gt_, \
            prec_all_, rec_all_, f1_all_, acc_all_ = test_result
            # todo: prepare a sample data
            '''
            pd_dictionary, pd_dictionary_complete = complete_version_dict_from_dict(category, pd_dictionary)
            gt_dictionary, gt_dictionary_complete = complete_version_dict_from_dict(category, gt_dictionary)

            re_performance_str, measurement_performance_str = get_format_result(acc_gt_, prec_gt_, rec_gt_, acc_all_, prec_all_, rec_all_, pd_dictionary, pd_dictionary_complete, gt_dictionary, gt_dictionary_complete, category)
            str_write += re_performance_str + '\n'
            measurement_performance_str_all_cat += measurement_performance_str + '\n'
            print(5, measurement_performance_str)
            '''

        # else:
        #     pd_dictionary = test_result
        #     _, pd_dictionary_complete = complete_version_dict_from_dict(category, pd_dictionary)
        #     write_version_dict_to_py_file(category, pd_dictionary_complete, labeled_data=False, category_separate_idx=file_idx, gaze=gazetteer)

        # if file_idx is not None:
        #     all_pd_dictionary_complete[category + '_' + str(file_idx)] = pd_dictionary_complete
        # else:
        #     all_pd_dictionary_complete[category] = pd_dictionary_complete

        # all_pd_dictionary_complete[category] = pd_dictionary_complete
        # merged_pd_dictionary_complete = utils.merge_dict_to_write(merged_pd_dictionary_complete, pd_dictionary_complete)

    # merged_pd_dictionary, merged_pd_dictionary_complete = complete_version_dict_from_dict(None, merged_pd_dictionary, all_data_name='all')
    # measurement_performance_str_all_cat += '\n' + measure_by_year_os_software(all_pd_dictionary_complete, merged_pd_dictionary_complete, 50, debug_mode=False, by_year=True, by_os=True, by_software=True) + '\n'

    with open(f_test_result_name, 'a') as f_test_result:
        f_test_result.write('\n\nmodel name: ' + model_id + '\n')
        f_test_result.write('model notes: ' + result_name + '\n')

        # f_test_result.write('model notes: ' + model_notes + '\n')

        if generate_from_ner_output_path is not None:
            f_test_result.write('ner output path: ' + generate_from_ner_output_path + '\n')
        logging.info('\n\nmodel name: ' + model_id + '\nmodel notes: ' + result_name + '\n')

        # if labeled:
        #         f_test_result.write(str_write + '\n' + measurement_performance_str_all_cat + '\n')


if __name__ == "__main__":
    test_re()



