import numpy as np
import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import config, utils
# from gazetteer.apply_gaze import apply_rules
from ner_model.initial_NER import extract_ent


def get_ner_model_from_dir(save_path, category):
    f_list = os.listdir(save_path)
    for f in f_list:
        if f.find(config.ner_model_prefix + category + '_') != -1:
            return f


def evaluate_each_class(py, test_data_dic, y_, m_, file_result=None, debug=False, apply_rule=True,
                        debug_gazetteer=False):
    CAT = ['N','V']
    if debug_gazetteer:
        with open('evaluate_each_class_params.py', 'w') as f_write:
            f_write.write('py = ' + str(py) + '\n\n')
            f_write.write('test_data_dic = ' + str(test_data_dic) + '\n\n')
            f_write.write('y_ = ' + str(y_) + '\n\n')
            f_write.write('m_ = ' + str(m_) + '\n\n')
            f_write.write('file_result = ' + str(file_result) + '\n\n')
    if not apply_rule:
        if len(py.shape) > 1:
            py = np.argmax(py, axis=1)
    else:
        py = py.flatten()

    y, m = y_.flatten(), m_.flatten()
    y_size = y.shape[0]
    py_size = py.shape[0]
    sentence_list = []
    # format: [[[py_idx1, word1, label1], [py_idx2, word2, label2], [py_idx3, word3, label3], ...], [], ...]
    # all sentences --> one sentence --> one word with one label
    one_sentence = []
    for td in test_data_dic:
        if td - 1 < y_size:
            if td - 1 < py_size:
                predicted_label = py[td - 1]

                if predicted_label != 0:
                    predicted_label = config.labels[predicted_label]
                else:
                    predicted_label = 'O'

                test_data_dic[td][1] = predicted_label
                # test_data_dic[td][2] = 'O'

                one_sentence.append([td - 1, test_data_dic[td][0], predicted_label])
                if len(test_data_dic[td]) in [3, 7]:
                    sentence_list.append(one_sentence)
                    one_sentence = []

    if apply_rule:
        # todo: double check labeled=True
        py = apply_rules(py, sentence_list, True)

    print_and_write_test_results(test_data_dic, y_size, py_size, y, py, file_result=file_result, debug=debug)
    acc = 1.0 * (np.array(y == py, dtype=np.int32) * m).sum() / m.sum()
    tp, fp, fn = 0, 0, 0
    p_ent = extract_ent(py, m)
    y_ent = extract_ent(y, m)
    metric_result = {}
    for c in CAT:
        metric_result[c] = [0] * 6
    for ent in p_ent:
        if ent in y_ent:
            metric_result[ent[2]][0] += 1
        else:
            metric_result[ent[2]][1] += 1
    for ent in y_ent:
        if ent not in p_ent:
            metric_result[ent[2]][2] += 1
    for c in CAT:
        tp = metric_result[c][0]
        fp = metric_result[c][1]
        fn = metric_result[c][2]

        prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0

        metric_result[c][3] = prec
        metric_result[c][4] = recall
        metric_result[c][5] = f1

    tp, fp, fn = 0, 0, 0
    for ent in p_ent:
        if ent in y_ent:
            tp += 1
        else:
            fp += 1
    for ent in y_ent:
        if ent not in p_ent:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0

    return acc, f1, metric_result


def print_and_write_test_results(test_data_dic, y_size, py_size, y, py, file_result=None, debug=True):
    # todo: y_size == py_size?
    for td in test_data_dic:
        if td - 1 >= y_size:
            continue
        if td - 1 >= py_size:
            continue
        ground_truth_label = y[td - 1]
        predicted_label = py[td - 1]

        if ground_truth_label != 0:
            ground_truth_label = config.labels[ground_truth_label]
        else:
            ground_truth_label = 'O'

        if predicted_label != 0:
            predicted_label = config.labels[predicted_label]
        else:
            predicted_label = 'O'

        test_data_dic[td][1] = predicted_label
        # test_data_dic[td][2] = 'O'
        print_line = utils.transform_list_to_str(test_data_dic[td][:6])

        if file_result is not None:
            file_result.write(print_line + '\n')
            if len(test_data_dic[td]) in [3, 7]:
                file_result.write('\n')
        if debug:
            if ground_truth_label != predicted_label:
                print_line += ' ------------ ' + ground_truth_label
            print(print_line)
            if len(test_data_dic[td]) in [3, 7]:
                print()