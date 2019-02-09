import re
import copy
import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import config, utils
from patterns import opposite_symbol_dict, version_regex
from version_comparision import compare_single_version


def clean_version_list(db, before_clean_list):
    # [v1, v2, ...]
    # {software1: [v1, v2...], software2:[v1, v2...]}
    after_clean_list = []

    # remove comma and fenhao
    replace_str = '!@#$%'
    for i in before_clean_list:
        # if i == '6 and 7':
        #     print(i)
        for s in [',', ';', ]:
            i = i.replace(s, replace_str)
        after_clean_list += strip_and(i.split(replace_str))
    # if db == standard:
    #     return after_clean_list

    # remove word in other db versions that are not in cve versions, except for words about range
    after_clean_list = clean_redundant_words_and_reserve_range(after_clean_list)
    return after_clean_list


def clean_redundant_words_and_reserve_range(before_clean_list):
    # preserve words that are
    # (1) in the cve word set or
    # (2) contain numbers or
    # (3) do not contain number and letter or
    # (4) in range word set
    # todo: enrich range word set
    range_word_set = {'before', 'older', 'prior', 'up', 'to', 'through', 'and', 'earlier',
                      'upper', 'higher', 'lower', 'including', 'since', 'onwards'}

    after_clean_set = set()
    for version_str in before_clean_list:
        clean_version_str = ''
        version_str_word_list = version_str.split()
        for word in version_str_word_list:
            if utils.contain_number(word) or (not utils.contain_number(word) and not utils.contain_letter(
                    word)) or word in range_word_set:
                clean_version_str += word + ' '
        after_clean_set.add(clean_version_str.strip())
    after_clean_set -= {''}
    # if '' in after_clean_set:
    #     after_clean_set.remove('')
    return list(after_clean_set)


def remove_letters_from_version_list(version_list):
    # input format: [set(), set(), set()]
    # only preserve 1.2.x format words

    single_set, one_set, two_set = version_list
    new_single_set, new_one_set, new_two_set = set(), set(), set()

    for version_str in single_set:
        version_str = remove_space_in_focus_version(version_str)
        version_str = get_dot_word_or_int_word(version_str)
        if version_str != '':
            new_single_set.add(version_str)

    for one_version in one_set:
        symbol, number = one_version
        number = get_dot_word_or_int_word(number)
        if number != '':
            new_one_set.add((symbol, number))

    for two_version in two_set:
        number_1, symbol_1, _, symbol_2, number_2 = two_version
        number_1 = get_dot_word_or_int_word(number_1)
        number_2 = get_dot_word_or_int_word(number_2)
        if number_1 != '' and number_2 != '':
            new_two_set.add((number_1, symbol_1, _, symbol_2, number_2))
        elif number_1 != '' and number_2 == '':
            new_one_set.add((opposite_symbol_dict[symbol_1], number_1))
        elif number_1 == '' and number_2 != '':
            new_one_set.add((symbol_2, number_2))

    new_version_list = [new_single_set, new_one_set, new_two_set]
    return new_version_list


def remove_space_in_focus_version(version_str):
    version_split = version_str.split()
    if len(version_split) == 2:
        if version_split[1][0] == '.' and utils.contain_number(version_split[1][1:]):
            version_str = version_str.replace(' ', '')
    return version_str


def clean_software_name_in_version_dict(version_dict_for_a_cve_id):
    # remove company name in software
    clean_software_version_dict = {}
    for db in version_dict_for_a_cve_id:
        clean_software_version_dict[db] = {}
        if db == 'vul_type':
            continue
        for link in version_dict_for_a_cve_id[db]:
            clean_software_version_dict[db][link] = {}
            for t_or_c in version_dict_for_a_cve_id[db][link]:
                clean_software_version_dict[db][link][t_or_c] = {}
                # print(version_dict[cveid][db][link][t_or_c])
                if type(version_dict_for_a_cve_id[db][link][t_or_c]) != dict:
                    continue
                for software in version_dict_for_a_cve_id[db][link][t_or_c]:
                    if version_dict_for_a_cve_id[db][link][t_or_c][software] == []:
                        continue
                    clean_software = normalize_software_name(software)
                    if clean_software in clean_software_version_dict[db][link][t_or_c]:
                        already_versions = clean_software_version_dict[db][link][t_or_c][clean_software]
                        new_versions = version_dict_for_a_cve_id[db][link][t_or_c][software]
                        clean_software_version_dict[db][link][t_or_c][clean_software] = \
                            list(set(already_versions + new_versions))
                    else:
                        clean_software_version_dict[db][link][t_or_c][clean_software] = \
                            version_dict_for_a_cve_id[db][link][t_or_c][software]
    return clean_software_version_dict


def normalize_software_name(software):
    # todo: enrich company name list
    # some software names are also company names, e.g., huawei. hehehhe
    property_name_list = ['module', 'system', 'software', 'component', 'function', 'plugin']
    clean_software = ''
    software_word_list = software.replace('_', ' ').split()
    if len(software_word_list) == 1:
        return software

    idx = 0
    for word in software_word_list:
        # if word in company_name_list:
        #     continue
        if idx == 0 or software_word_list[idx-1] != word:
            clean_software += word + ' '
        idx += 1

    return clean_software.strip()


def get_dot_word_or_int_word(version_str):
    # return dot word, if not exists, return int word, if not exists, return null
    version_word_split = version_str.split()
    # if find dot word, return dot word, else return the first num word
    dot_word = get_dot_word(version_word_split)
    if dot_word != '':
        return dot_word

    int_word = get_int_word(version_word_split)
    return int_word


def remove_symbol_and_letter_from_dot_word(word):
    # [{'5.0-2.7', '6.0-2.7', '5.0-1.10', '5.0-2.0-beta2', '5.0-1.9', '6.0-2.0-beta3', '6.0-2.0-beta2', '5.0-2.0-beta3'}, set(), set()]
    # replace '-' with '.'
    word = word.replace('.x', '.0').replace('-', '.')

    # remove from the first non-digit-dot
    new_word = ''
    for i in word:
        if i.isdigit() or i == '.':
            new_word += i
        else:
            return new_word
    if new_word == '':
        print('ERROR 3', word)
    return new_word


def get_dot_word(version_word_split):
    for word in version_word_split:
        mat1 = re.match(version_regex, word)
        if mat1:
            return remove_symbol_and_letter_from_dot_word(word)
    return ''


def get_int_word(version_word_split):
    for word in version_word_split:
        if utils.only_contain_number(word):
            return word
    return ''


def strip_and(l):
    new_l = []
    for i in l:
        for strip_str in ['and']:
            if i.startswith(strip_str):
                i = i[len(strip_str):]
            elif i.endswith(strip_str):
                i = i[:-len(strip_str)]
            new_l.append(i.strip())
    new_l = list(set(new_l))
    return new_l


def remove_noise(version_list):
    # todo: remove two same words, i.e., 'and and 7'
    # format: [[single], [double], [triple]]

    new_single_set = set(remove_dirty_words_single(version_list[0]))
    new_single_set -= {''}
    new_single_set -= {'0'}
    new_one_set = set(remove_dirty_words_one(version_list[1]))
    new_two_set = set(remove_dirty_words_two(version_list[2]))

    new_single_set, new_one_set, new_two_set = remove_redundant_versions(new_single_set, new_one_set, new_two_set)

    version_list[0] = new_single_set
    version_list[1] = new_one_set
    version_list[2] = new_two_set

    return version_list


def remove_dirty_words_single(version_list):

    new_single_list = []
    for i in version_list:
        # remove ' 0'
        strip_str = ' 0'
        if i.startswith(strip_str):
            i = i[len(strip_str):]
        elif i.endswith(strip_str):
            i = i[:-len(strip_str)]
        if i != '0':
            new_single_list.append(i.strip())

    new_version_list = []

    for version_str in version_list:
        version_str = replace_dirty_str(version_str)

        new_version_list.append(version_str.strip())
    return new_version_list


def remove_dirty_words_one(version_list):
    new_version_list = []

    for version_tuple_2_elem in version_list:
        version_str = replace_dirty_str(version_tuple_2_elem[1])

        new_version_list.append((version_tuple_2_elem[0], version_str))
    return new_version_list


def remove_dirty_words_two(version_list):
    new_version_list = []

    for version_tuple_5_elem in version_list:
        version_str_0 = replace_dirty_str(version_tuple_5_elem[0])
        version_str_4 = replace_dirty_str(version_tuple_5_elem[4])

        new_version_list.append((version_str_0, version_tuple_5_elem[1], version_tuple_5_elem[2], version_tuple_5_elem[3], version_str_4))
    return new_version_list


def replace_dirty_str(version_str):
    words_to_be_removed = ['x64', 'x86_64', 'version', 'versions', '32-bit', '64-bit', 'other']
    if version_str[0] == 'v' and version_str[1].isdigit():
        version_str = version_str[1:]
    if version_str[:2] in ['v.', 'v-', 'v '] and version_str[2].isdigit():
        version_str = version_str[2:]
    # str_to_be_replaced_dict = {'< =': '<=', '> =': '>='}
    # for dirty_s in str_to_be_replaced_dict:
    #     version_str = version_str.replace(dirty_s, str_to_be_replaced_dict[dirty_s])
    new_version_str = ''
    word_list = version_str.split()
    word_idx = 0
    for word in word_list:
        exist = False
        exist_word = ''
        for w in words_to_be_removed:
            if w in word:
                exist = True
                exist_word = w
                break
        if exist or word == 'and' and word_idx in [0, len(word_list) - 1]:
            # print(1, exist_word, 2, word)
            pass
        else:
            new_version_str += word + ' '
        word_idx += 1

    return version_str.strip()


def remove_redundant_versions(single_set, one_set, two_set):
    # remove duplicate versions
    new_single_set, new_one_set, new_two_set = copy.deepcopy(single_set), copy.deepcopy(one_set), copy.deepcopy(two_set)

    # only remove from single set
    # todo: remove some two_set from one set
    for i in single_set:
        single_elem_redundant = False

        for j in one_set:
            single_in_one = is_single_elem_in_one_elem(i, j)
            if single_in_one:
                single_elem_redundant = single_in_one
                break

        if single_elem_redundant:
            continue

        for k in two_set:
            single_in_two = is_single_elem_in_two_elem(i, k)
            if single_in_two:
                single_elem_redundant = single_in_two
                break

        if single_elem_redundant:
            new_single_set -= {i}

    # if (single_set, one_set, two_set) != (new_single_set, new_one_set, new_two_set):
    #     a = (single_set, one_set, two_set)
    #     b = (new_single_set, new_one_set, new_two_set)
    #     print(a)
    #     print(b)

    return new_single_set, new_one_set, new_two_set


def is_single_elem_in_one_elem(i, j):
    # i: single_set_elem,
    # j: one_set_elem
    symbol, number = j
    compare_result = compare_single_version(i, number)

    # case 1:
    # {'1.2.4.2'}, {('<=', '1.2.4.2')}, set()
    # set(), {('<=', '1.2.4.2')}, set()

    if compare_result == '=':
        if symbol in ['<=', '>=']:
            return True

    # case 2:
    # {'7.8'}, {('<', '7.9')}, set(
    # set(), {('<=', '1.2.4.2')}, set()

    elif compare_result == '<':
        if symbol in ['<', '<=']:
            return True

    elif compare_result == '>':
        if symbol in ['>', '>=']:
            return True

    return False


def is_single_elem_in_two_elem(i, k):
    # i: single_set_elem,
    # k: two_set_elem
    number_1, symbol_1, _, symbol_2, number_2 = k
    compare_result_1 = compare_single_version(i, number_1)
    if compare_result_1 == '=':
        if symbol_1 in ['<=', '>=']:
            return True

    elif compare_result_1 == '>':
        if symbol_1 in ['<', '<=']:
            return True

    elif compare_result_1 == '<':
        if symbol_1 in ['>', '>=']:
            return True

    compare_result_2 = compare_single_version(i, number_1)
    if compare_result_2 == '=':
        if symbol_2 in ['<=', '>=']:
            return True

    elif compare_result_2 == '<':
        if symbol_1 in ['<', '<=']:
            return True

    elif compare_result_2 == '>':
        if symbol_1 in ['>', '>=']:
            return True

    return False


def is_single_in_one_two_range(single_set_1, one_set_2, two_set_2):
    # all single in the range set is ok
    for version_number in single_set_1:
        for i in one_set_2:
            symbol_2, number_2 = i
            if not in_version_range(version_number, symbol_2, number_2):
                return False
        for i in two_set_2:
            number_2_left, symbol_2_left, _, symbol_2_right, number_2_right = i
            if not in_version_range(version_number, symbol_2_left, number_2_left):
                return False
            if not in_version_range(version_number, opposite_symbol_dict[symbol_2_right], number_2_right):
                return False
    return True


def in_version_range(version_number, range_symbol, range_number):
    compare_symbol = compare_single_version(version_number, range_number)
    if compare_symbol == '=':
        return True
    elif compare_symbol == False:
        print('ERROR 4')
    elif range_symbol == opposite_symbol_dict[compare_symbol]:
        return False
    else:
        return True



