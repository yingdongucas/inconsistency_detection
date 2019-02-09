from numpy import np
import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import config


def get_model_list_from_re_model_dir(model_dir, category, transfer):
    if not transfer:
        category = 'memc'
    file_list = os.listdir(model_dir)
    model_id_set = set()
    print(config.re_model_prefix + category, transfer)
    for f in file_list:
        # if not f.startswith('ATT_GRU_model-'):
        if not f.startswith(config.re_model_prefix + category):
            continue
        loc1 = f.find('-')
        loc2 = f.find('.')
        model_id_set.add(f[loc1 + 1:loc2])

    model_id_list = list(model_id_set)
    if len(model_id_list) > 1:
        print('ERROR ')
        exit()
    model_id = model_id_list[0]
    return model_id


def compute_re_prediction_on_ner_output_with_gt(pd_txt_file, pd_array, gt_txt_file):
    pd_list = pd_array
    if type(pd_list) == np.ndarray:
        pd_list = pd_array.tolist()

    with open(pd_txt_file) as f_pd_txt:
        pd_txt_lines = f_pd_txt.readlines()

    if len(pd_txt_lines) != len(pd_list):
        logging.info('ERROR in pd txt file!')
        return

    with open(gt_txt_file) as f_gt_txt:
        gt_txt_lines = f_gt_txt.readlines()

    pd_data_label_dict = convert_txt_line_to_data_label_dict(pd_txt_lines, pd_list)
    gt_data_label_dict = convert_txt_line_to_data_label_dict(gt_txt_lines)

    pre, rec, f1, acc = performance_on_gt_pairs(gt_data_label_dict, pd_data_label_dict)
    pre1, rec1, f11, acc1 = performance_on_all_pairs(gt_data_label_dict, pd_data_label_dict)

    return pre, rec, f1, acc, pre1, rec1, f11, acc1


def performance_on_gt_pairs(gt_data_label_dict, pd_data_label_dict):
    tp, fp, fn = 0, 0, 0
    correct_cnt = 0
    gt_in_pd_cnt = 0
    for data in gt_data_label_dict:
        if data in pd_data_label_dict:
            gt_in_pd_cnt += 1
            if gt_data_label_dict[data] == pd_data_label_dict[data]:
                correct_cnt += 1
            if gt_data_label_dict[data] == pd_data_label_dict[data] == 1:
                tp += 1
            if gt_data_label_dict[data] == 1 and pd_data_label_dict[data] == 0:
                fn += 1
            if gt_data_label_dict[data] == 0 and pd_data_label_dict[data] == 1:
                fp += 1
    pre = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * pre * rec / (pre + rec) if pre + rec > 0 else 0.0
    acc = (correct_cnt + 0.0) / gt_in_pd_cnt

    logging.info('\ngt performance')
    logging.info('tp = ' + str(tp))
    logging.info('fp = ' + str(fp))
    logging.info('fn = ' + str(fn))
    logging.info('correct_cnt = ' + str(correct_cnt))
    logging.info('gt_in_pd_cnt = ' + str(gt_in_pd_cnt))
    logging.info('len(pd_data_label_dict) = ' + str(len(pd_data_label_dict)))
    logging.info('len(gt_data_label_dict) = ' + str(len(gt_data_label_dict)))
    logging.info('pre = ' + str(pre))
    logging.info('rec = ' + str(rec))
    logging.info('acc = ' + str(acc))

    return pre, rec, f1, acc


def performance_on_all_pairs(gt_data_label_dict, pd_data_label_dict):
    tp, fn = 0, 0
    tn1, tn2 = 0, 0
    fp1, fp2 = 0, 0
    for data in gt_data_label_dict:
        if data in pd_data_label_dict:
            if gt_data_label_dict[data] == pd_data_label_dict[data] == 0:
                tn1 += 1
            elif gt_data_label_dict[data] == pd_data_label_dict[data] == 1:
                tp += 1
            elif gt_data_label_dict[data] == 1 and pd_data_label_dict[data] == 0:
                fn += 1
            elif gt_data_label_dict[data] == 0 and pd_data_label_dict[data] == 1:
                fp1 += 1
            else:
                print('ERROR')

    for data in pd_data_label_dict:
        if pd_data_label_dict[data] == 1:
            if data not in gt_data_label_dict:
                fp2 += 1
        elif pd_data_label_dict[data] == 0:
            if data not in gt_data_label_dict:
                tn2 += 1
        else:
            print('ERROR')

    fp = fp1 + fp2
    tn = tn1 + tn2
    pre = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * pre * rec / (pre + rec) if pre + rec > 0 else 0.0
    acc = (tp + tn + 0.0) / len(pd_data_label_dict)

    logging.info('\nall performance')
    logging.info('tp = ' + str(tp))
    logging.info('fn = ' + str(fn))
    logging.info('tn1 = ' + str(tn1))
    logging.info('tn2 = ' + str(tn2))
    logging.info('fp1 = ' + str(fp1))
    logging.info('fp2 = ' + str(fp2))
    logging.info('len(pd_data_label_dict) = ' + str(len(pd_data_label_dict)))
    logging.info('len(gt_data_label_dict) = ' + str(len(gt_data_label_dict)))
    logging.info('pre = ' + str(pre))
    logging.info('rec = ' + str(rec))
    logging.info('acc = correct_cnt / len(pd_data_label_dict) = ' + str(acc))

    return pre, rec, f1, acc


def convert_txt_line_to_data_label_dict(content_lines, relations=None):
    data_label_dict = dict()
    idx = 0
    for content in content_lines:
        content = content.strip().split()
        relation = content[2]
        relation = utils.relation2id[relation]
        if relations is not None:
            relation = relations[idx]
        en1pos = int(content[0])
        en2pos = int(content[1])
        sentence = content[3:-1]
        data = utils.transform_list_to_str([en1pos] + [en2pos] + sentence)
        if data not in data_label_dict:
            data_label_dict[data] = relation
        idx += 1
    return data_label_dict


def get_format_result_loose(pd_dict, pd_dict_complete, gt_dict, gt_dict_complete, cc, loose=True):
    pd_match_rate, _ = clean_version_dict(pd_dict, cc, loose=loose)
    print(1, _)
    gt_match_rate, _ = clean_version_dict(gt_dict, cc, loose=loose)

    print(2, _)
    pd_match_rate_complete, db_match_rate_str = clean_version_dict(pd_dict_complete, cc, loose=loose)
    print(3, db_match_rate_str)
    gt_match_rate_complete, _ = clean_version_dict(gt_dict_complete, cc, loose=loose)

    print(4, _)
    deviation = '-'
    if type(pd_match_rate) != str and type(gt_match_rate) != str:
        deviation = pd_match_rate - gt_match_rate
    normalized_deviation = '-'
    if gt_match_rate != 0 and deviation != '-':
        normalized_deviation = abs(deviation / gt_match_rate)

    deviation_complete = '-'
    if type(pd_match_rate_complete) != str and type(gt_match_rate_complete) != str:
        deviation_complete = pd_match_rate_complete - gt_match_rate_complete
    normalized_deviation_complete = '-'
    if gt_match_rate_complete != 0 and deviation_complete != '-':
        normalized_deviation_complete = abs(deviation_complete / gt_match_rate_complete)

    str_write_one_cat = utils.format_str(['loose' if loose else 'strict',
                                          'rate',
                                          pd_match_rate,
                                          deviation,
                                          normalized_deviation,
                                          'complete_rate',
                                          pd_match_rate_complete,
                                          deviation_complete,
                                          normalized_deviation_complete
                                          ])
    return str_write_one_cat, db_match_rate_str


def get_format_result(accuracy_gt, precision_gt, recall_gt, accuracy_all, precision_all, recall_all, pd_dict,
                      pd_dict_complete, gt_dict, gt_dict_complete, cc):
    str_write_one_cat = utils.format_str([cc,
                                          'gt',
                                          float(accuracy_gt), float(precision_gt), float(recall_gt),
                                          'all',
                                          float(accuracy_all), float(precision_all), float(recall_all)
                                          ])
    str_write_one_cat_loose, db_match_rate_str_loose = get_format_result_loose(pd_dict, pd_dict_complete, gt_dict,
                                                                               gt_dict_complete, cc, loose=True)
    str_write_one_cat_strict, db_match_rate_str_strict = get_format_result_loose(pd_dict, pd_dict_complete, gt_dict,
                                                                                 gt_dict_complete, cc, loose=False)
    return str_write_one_cat + '\t' + str_write_one_cat_loose + '\t' + str_write_one_cat_strict, \
           db_match_rate_str_loose + '\t' + db_match_rate_str_strict


def get_category_thr_from_re_dir(new_category_data_model_dir):
    category_thr_list = set()
    for f in new_category_data_model_dir:
        if f.endswith('_pos1.npy'):
            for thr in ['_0.0', '_0.1']:
                if thr in f:
                    continue
            category_thr_list.add(f[:f.find('_q&a.txt')])
    return category_thr_list


def get_separate_cat_file_list(ner_output_path, category_list, test_flag):
    file_list = os.listdir(ner_output_path)
    separate_cat_file_list = []
    for category_ in category_list:
        for f in file_list:
            prefix = category_ + test_flag + config.ner_data_suffix + '_'
            if f.startswith(prefix):
                if filter_cat_idx(category_, f.split('_')[-1]):
                    separate_cat_file_list.append([category_, f.split('_')[-1]])
    return separate_cat_file_list


def filter_cat_idx(cc, idd):
    return True
    # if cc not in ['sqli', 'bypass']:
    #     return True
    # if cc == 'sqli' and int(idd) >= 45:
    #     return True
    # if cc == 'bypass' and int(idd) >= 20:
    #     return True


def get_none_id(type_filename):
    with open(type_filename, encoding='utf-8') as type_file:
        for line in type_file:
            ls = line.strip().split()
            if ls[0] == "None":
                return int(ls[1])


def evaluate_rm_neg(prediction, ground_truth, none_label_index, with_acc=False):
    """
    Evaluation matrix.
    :param prediction: a dictionary of labels. e.g {0:[1,0],1:[2],2:[3,4],3:[5,6,7]}
    :param ground_truth: a dictionary of labels
    :return:
    """
    pos_pred = 0.0
    pos_gt = 0.0
    true_pos = 0.0
    fp = 0.0
    fn = 0.0
    correct_cnt = 0.0
    for i in range(len(ground_truth)):
        if ground_truth[i] != none_label_index:
            pos_gt += 1.0

    for i in range(len(prediction)):
        if prediction[i] != none_label_index:
            # classified as pos example (Is-A-Relation)
            pos_pred += 1.0
            if prediction[i] == 1 and ground_truth[i] == 1:
                true_pos += 1.0
                correct_cnt += 1.0
            elif prediction[i] == 1 and ground_truth[i] == 0:
                fp += 1.0
            elif prediction[i] == 0 and ground_truth[i] == 1:
                fn += 1.0
            else:
                correct_cnt += 1.0

    '''
    precision = true_pos / (pos_pred + 1e-8)
    recall = true_pos / (pos_gt + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    '''
    precision = true_pos / (true_pos + fp)
    recall = true_pos / (true_pos + fn)
    f1 = 2 * precision * recall / (precision + recall)
    acc = correct_cnt/len(prediction) 
    if with_acc:
        return precision, recall, f1, acc
    else:
        return precision, recall, f1





