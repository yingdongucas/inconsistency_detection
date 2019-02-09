import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import utils


def compare_single_version(version_str_1, version_str_2, compare_single_set=False):
    if compare_single_set:
        if version_str_1 == '0' or version_str_2 == '0':
            return '='
    # make sure version_str1 and 2 only contain letters and dots
    if not utils.only_contain_dots_and_number(version_str_1) or not utils.only_contain_dots_and_number(version_str_2):
        return False

    version_1_split, version_2_split = make_split_equal_len(version_str_1, version_str_2)

    if version_str_1 == version_str_2:
        return '='

    version_1_contain_dots = utils.contain_dots(version_str_1)
    version_2_contain_dots = utils.contain_dots(version_str_2)

    # test 1.1.0, 1.1
    for i in list(range(len(version_1_split))):
        num_version_1 = version_1_split[i]
        num_version_2 = version_2_split[i]

        # {'9.0.6', '6060'},  ['desktop 9.0.6', 'build 6060']
        if compare_single_set:
            if abs(num_version_1 - num_version_2) > 10 and (version_1_contain_dots is not version_2_contain_dots):
                # print(version_str_1, version_str_2)
                return '='

        if num_version_1 > num_version_2:
            return '>'
        elif num_version_1 < num_version_2:
            return '<'
    return '='


def make_split_equal_len(version_str_1, version_str_2):

    version_1_split = convert_str_into_int(version_str_1.split('.'))
    version_2_split = convert_str_into_int(version_str_2.split('.'))

    version_1_len = len(version_1_split)
    version_2_len = len(version_2_split)

    max_len = max(version_1_len, version_2_len)

    make_up_zero_cnt = max_len - version_1_len
    if make_up_zero_cnt != 0:
        for i in list(range(make_up_zero_cnt)):
            version_1_split.append(0)

    make_up_zero_cnt = max_len - version_2_len
    if make_up_zero_cnt != 0:
        for i in list(range(make_up_zero_cnt)):
            version_2_split.append(0)

    return version_1_split, version_2_split


def convert_str_into_int(str_list):
    int_list = []
    for s in str_list:
        if s != '':
            int_list.append(int(s))
    return int_list