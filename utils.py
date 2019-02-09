import sys
from config import *


class add_path():
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        sys.path.remove(self.path)


def contain_letter(s):
    return any(i.isalpha() for i in s)


def contain_number(s):
    return any(i.isdigit() for i in s)


def only_contain_dots_and_number(s):
    return all(i.isdigit() or i == '.' for i in s)


def only_contain_number(s):
    return all(i.isdigit() for i in s)


def contain_dots(s):
    return any(i == '.' for i in s)


def is_ascii(ss):
    return all(ord(c) < 128 for c in ss)


def transform_list_to_str(l):
    s = ''
    for i in l:
        s += str(i) + ' '
    return s


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_word2id():
    with add_path(labeled_re_data_write_path):
        word2id_module = __import__(word2id_file_name)
        return word2id_module.word2id


def cut_str(i, cut_len=6):
    i = str(i)
    if len(i) >= cut_len:
        return i[:cut_len]
    return i


def format_str(l, sep='\t'):
    s = ''
    for i in l:
        if type(i) == float:
            s += cut_str(i) + sep
        else:
            s += str(i) + sep
    return s.strip(sep)


def merge_dict_to_write(dict1, dict2):
    merged_dict = dict1
    for cveid in dict2:
        if cveid not in merged_dict:
            merged_dict[cveid] = dict2[cveid]
    return merged_dict


def find_multiple_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))
    return results


def convert_loc_range_to_index(start_end):
    start, end = start_end
    loc = []
    temp = start
    while temp <= end:
        loc.append(temp)
        temp += 1
    return loc


def apply_choice(f_result_name, transfer, tip):
    if transfer:
        f_result_name += '_' + tip
    else:
        f_result_name += '_not' + tip
    return f_result_name


def get_f_result_name(case_idx, operation, transfer, labeled, duplicate, gazetteer, neroutput=False, ner=False, re=False):
    f_result_name = 'case_' + case_idx + '_' + operation
    f_result_name = apply_choice(f_result_name, transfer, 'transfer')
    f_result_name = apply_choice(f_result_name, labeled, 'labeled')
    f_result_name = apply_choice(f_result_name, duplicate, 'duplicate')
    f_result_name = apply_choice(f_result_name, gazetteer, 'gazetteer')
    if neroutput:
        f_result_name = apply_choice(f_result_name, neroutput, 'neroutput')
    elif ner:
        f_result_name = apply_choice(f_result_name, ner, 'ner')
    elif re:
        f_result_name = apply_choice(f_result_name, re, 're')
    else:
        print('ERROR')
        return
    return f_result_name


def check_dir():
    if not os.path.exists(labeled_ner_data_input_path) or not os.path.exists(labeled_re_data_input_path):
        print('Please check data input.')
        return
    if not os.path.exists(data_write_path):
        print('Please set a data output path.')
        return
    for dir_name in [labeled_re_data_write_path,
                     labeled_ner_data_output_after_transfer_path,
                     labeled_ner_data_output_before_transfer_path,
                     labeled_re_data_output_after_transfer_path,
                     labeled_re_data_output_before_transfer_path,
                     ner_model_path_before_transfer,
                     ner_model_path_after_transfer,
                     re_model_path_before_transfer,
                     re_model_path_after_transfer,
                     sh_output_path,
                     ]:
        if os.path.exists(dir_name):
            continue
        os.makedirs(dir_name)


check_dir()

