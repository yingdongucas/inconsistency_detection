import numpy as np
import nltk
import os, sys, inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import config, utils

from measurement.clean_version_and_measure import get_all_cat_version_dict, get_software_freq_from_version_dict
from corpus_and_embedding.clean_cvedetails_software_list import clean_software_name_list

excluded_software_token_list = []

def get_software_from_gazetteer(get_gazetteer_from_version_dict=False, get_gazetteer_from_report_content_freq=False,
                                get_gazetteer_from_cvedetails=False):
    if [get_gazetteer_from_version_dict, get_gazetteer_from_report_content_freq, get_gazetteer_from_cvedetails].count(
            True) != 1:
        print('ERROR')
        return
    software_name_list = None

    if get_gazetteer_from_version_dict:
        all_cat_dict = get_all_cat_version_dict(config.version_dict_file_path, config.labeled_version_dict_file_name)
        software_name_with_freq_dic = get_software_freq_from_version_dict(get_from_version_dict=True,
                                                                          get_from_report_content_freq=False,
                                                                          all_cat_version_dict=all_cat_dict)
        software_name_list = list(software_name_with_freq_dic.keys())

    elif get_gazetteer_from_cvedetails:
        # with commons.add_path('../corpus_and_embedding/'):
        #     # category_module = __import__(commons.clean_cvedetails_software_file_name)
        #     category_module = __import__('clean_cvedetails_software_list')
        #     software_name_list = category_module.software_name_list
        software_name_list = clean_software_name_list

    elif get_gazetteer_from_report_content_freq:
        with utils.add_path('../dataset_preparation/'):
            category_module = __import__('software_name_collected')
            software_name_with_freq_dic = category_module.software_name_with_freq
            software_name_list = list(software_name_with_freq_dic.keys())

    return software_name_list


def apply_rules(py, sentence_list, labeled, debug_gazetteer=False):
    if debug_gazetteer:
        with open('apply_rules_params.py', 'w') as f_write:
            f_write.write('sentence_list = ' + str(sentence_list) + '\n\n')
        np.save('py.npy', np.array(py))

    if type(py) == np.ndarray:
        if len(py.shape) > 1:
            print('ERROR')
            return
        py = list(py)

    py = [config.labels[i] for i in py]

    if labeled:
        get_gazetteer_from_version_dict = True
        get_gazetteer_from_report_content_freq = False
        get_gazetteer_from_cvedetails = False
    else:
        get_gazetteer_from_version_dict = False
        get_gazetteer_from_report_content_freq = False
        get_gazetteer_from_cvedetails = True

    software_name_list = get_software_from_gazetteer(get_gazetteer_from_version_dict=get_gazetteer_from_version_dict,
                                                     get_gazetteer_from_report_content_freq=get_gazetteer_from_report_content_freq,
                                                     get_gazetteer_from_cvedetails=get_gazetteer_from_cvedetails)
    software_tokenized_list = []
    for software in software_name_list:
        software_tokenized = nltk.word_tokenize(software)
        software_tokenized_list.append(software_tokenized)

    # sentence_list format:
    # [[[py_idx1, word1, label1], [py_idx2, word2, label2], [py_idx3, word3, label3], ...], [], ...]
    # all sentences --> one sentence --> one word idx with word and label

    for sentence in sentence_list:
        label_list = [i[2] for i in sentence]
        word_list = [i[1].lower() for i in sentence]
        if config.version_label in label_list:
            if config.software_label not in label_list:
                py = match_software_close_to_version(py, sentence, word_list, software_tokenized_list)
            else:
                py = deal_with_half_detected_software(py, sentence, word_list, software_tokenized_list)
    corrected_py = [config.labels.index(i) for i in py]
    print(corrected_py == py)
    corrected_py = np.array(corrected_py)
    return corrected_py


def deal_with_half_detected_software(py, sentence, word_list, software_tokenized_list):
    loc_software_dict = get_software_with_loc(sentence)
    for pd_loc_range in loc_software_dict:
        pd_software_token_list = loc_software_dict[pd_loc_range]
        if pd_software_token_list not in software_tokenized_list:
            # for software_token in pd_software_token_list:
            matched_loc, matched_software, wrong_software_loc_list = match_longest_software_in_gazetteer(
                pd_software_token_list, pd_loc_range, sentence, word_list, software_tokenized_list)
            if matched_loc is not None and matched_software is not None:
                matched_range = utils.convert_loc_range_to_index(matched_loc)
                py = set_py_label(py, matched_range, config.software_label)
                print_str = 'case 2 -- match half/over software for version | ' + utils.format_str(
                    pd_software_token_list, sep=' ') + ' | ' + utils.format_str(matched_software, sep=' ')
                print(print_str)

                # todo: maybe versions, not non-entities
                if wrong_software_loc_list is not None:
                    py = set_py_label(py, wrong_software_loc_list, 'O')
                    if wrong_software_loc_list != []:
                        base = sentence[0][0]
                        print_str = 'case 2 -- exclude rest software word after matching | '
                        for i in wrong_software_loc_list:
                            if sentence[i - base][1] == '':
                                print('ERROR')
                            print_str += ' ' + sentence[i - base][1]
                        print(print_str)

            else:
                py = set_py_label(py, pd_loc_range, 'O')
                print_str = 'case 3 -- exclude software not in gazetteer | ' + utils.format_str(
                    pd_software_token_list, sep=' ')
                print(print_str)
        elif pd_software_token_list in excluded_software_token_list:
            py = set_py_label(py, pd_loc_range, 'O')
            print_str = 'case 4 -- exclude illegal software | ' + utils.format_str(pd_software_token_list,
                                                                                     sep=' ')
            print(print_str)
    return py


def match_software_close_to_version(py, sentence, word_list, software_tokenized_list):
    version_start_loc_list = get_version_start_loc(sentence)
    # format: [start_loc_1, start_loc_2, ...]

    software_end_loc_dict = get_software_end_loc_of_gazatteer_software_in_sentence(sentence, word_list,
                                                                                   software_tokenized_list)
    # format: {end_loc_1: start_1, end_loc_2: start_2, ...}

    for version_start_loc in version_start_loc_list:
        closet_dist = config.ner_max_len
        closet_end_loc = None
        closet_start_loc = None
        for software_end_loc in software_end_loc_dict:
            dist = software_end_loc - version_start_loc
            if dist < 0 and abs(dist) < abs(closet_dist):
                closet_end_loc = software_end_loc
                closet_start_loc = software_end_loc_dict[software_end_loc]
        if closet_end_loc is not None and closet_start_loc is not None:
            closet_software_loc_range = utils.convert_loc_range_to_index((closet_start_loc, closet_end_loc))
            base = sentence[0][0]
            closet_software_token = []
            for i in closet_software_loc_range:
                token = sentence[i - base][1]
                closet_software_token.append(token)

            if closet_software_token not in commons.excluded_software_token_list:
                py = set_py_label(py, closet_software_loc_range, config.software_label)

                print_str = 'case 1 -- match software for version | ' + sentence[version_start_loc - base][1] + ' | '
                print_str += utils.format_str(closet_software_token, sep=' ')
                print(print_str)
            else:
                print(closet_software_token)
    return py


def set_py_label(py, target_range, target_label):
    for i in target_range:
        py[i] = target_label
    return py


def get_software_end_loc_of_gazatteer_software_in_sentence(sentence, word_list, software_tokenized_list):
    software_end_loc_dict = dict()
    # format: {end_loc_1: software_1, end_loc_2: software_2, ...}

    for software_tokenized in software_tokenized_list:
        start_end_list = utils.find_multiple_sub_list(software_tokenized, word_list)
        # if len(start_end_list) > 1:
        #     print(start_end_list, software_tokenized, word_list)
        if start_end_list != []:
            for start_end in start_end_list:
                start, end = start_end
                end += sentence[0][0]
                start += sentence[0][0]
                software_end_loc_dict[end] = start
    return software_end_loc_dict


def get_version_start_loc(sentence):
    # sentence format:
    # [[py_idx1, word1, label1], [py_idx2, word2, label2], [py_idx3, word3, label3], ...]
    version_start_loc = []
    previous_label_is_version = False
    for word_triple_list in sentence:
        py_idx, word, label = word_triple_list
        if label == config.version_label:
            if not previous_label_is_version:
                version_start_loc.append(py_idx)
            previous_label_is_version = True
        else:
            previous_label_is_version = False
    return version_start_loc


def get_software_with_loc(sentence):
    # sentence format:
    # [[py_idx1, word1, label1], [py_idx2, word2, label2], [py_idx3, word3, label3], ...]
    loc_software_dict = dict()
    # loc_software_dict format:
    # {(idx1, idx2, idx3): [word1, word2, word3], ...}
    previous_label_is_software = False
    one_software_loc = []
    one_software_word = []
    for word_triple_list in sentence:
        py_idx, word, label = word_triple_list
        if label == config.software_label:
            one_software_loc.append(py_idx)
            one_software_word.append(word.lower())
            previous_label_is_software = True
        elif previous_label_is_software:
            loc_software_dict[tuple(one_software_loc)] = one_software_word
            one_software_loc = []
            one_software_word = []
            previous_label_is_software = False
    return loc_software_dict


def match_longest_software_in_gazetteer(pd_software_token_list, pd_loc_range, sentence, word_list,
                                        software_tokenized_list):
    pd_software_start = pd_loc_range[0]
    pd_software_end = pd_loc_range[-1]
    # if pd_software_token_list == ['debian', 'unstable']:
    #     print(111111)
    matched_loc, matched_software, wrong_software_loc_list = None, None, None
    previous_gazetteer_software_len = -1
    for gazetter_software_token in software_tokenized_list:
        for pd_software_token in pd_software_token_list:
            if pd_software_token in gazetter_software_token:
                start_end_list = utils.find_multiple_sub_list(gazetter_software_token, word_list)
                # if len(start_end_list) > 1:
                #     print(start_end_list, gazetter_software_token, word_list)
                if start_end_list != []:
                    for start_end in start_end_list:
                        gazetteer_start, gazetteer_end = start_end
                        gazetteer_end += sentence[0][0]
                        gazetteer_start += sentence[0][0]
                        intersect_and_wrong_software_loc = pd_gazetteer_software_intersect(pd_software_start,
                                                                                           pd_software_end,
                                                                                           gazetteer_start,
                                                                                           gazetteer_end)
                        if intersect_and_wrong_software_loc is not None:
                            intersect, wrong_software_loc = intersect_and_wrong_software_loc
                            current_gazetteer_software_len = gazetteer_end - gazetteer_start
                            if current_gazetteer_software_len > previous_gazetteer_software_len and intersect:
                                matched_loc = (gazetteer_start, gazetteer_end)
                                previous_matched_software = matched_software
                                matched_software = gazetter_software_token
                                wrong_software_loc_list = wrong_software_loc
                                previous_gazetteer_software_len = current_gazetteer_software_len

                                if previous_gazetteer_software_len != -1 and previous_matched_software is not None:
                                    print_str = 'case 2 -- match half/over software for version | replaced by | ' + utils.format_str(
                                        previous_matched_software, sep=' ') + ' | ' + utils.format_str(
                                        matched_software, sep=' ')
                                    print(print_str)

    return matched_loc, matched_software, wrong_software_loc_list


def pd_gazetteer_software_intersect(pd_software_start, pd_software_end, gazetteer_start, gazetteer_end):
    pd_range = utils.convert_loc_range_to_index((pd_software_start, pd_software_end))
    gazetteer_range = utils.convert_loc_range_to_index((gazetteer_start, gazetteer_end))
    wrong_software_loc = []
    for i in pd_range:
        if i not in gazetteer_range:
            wrong_software_loc.append(i)
    if set(pd_range).intersection(set(gazetteer_range)) != set():
        return True, wrong_software_loc


def debug_apply_rules():
    py = np.load('py.npy')
    from apply_rules_params import sentence_list
    apply_rules(py, sentence_list, labeled=True)


if __name__ == '__main__':
    debug_apply_rules()

