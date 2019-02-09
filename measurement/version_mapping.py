from cpe_name_dic import cpe_software_version_dict
from version_comparision import compare_single_version
from version_data_cleaning import get_dot_word_or_int_word


def map_range_to_points(converted_range_version_list, software):
    cpe_version_list = get_cpe_version_list(software)
    version_point_set = converted_range_version_list[0]
    one_version_range_point_set, two_version_range_point_set = set(), set()

    if converted_range_version_list[1] != set():
        symbol, range_boundary_version = parse_one_range_version_format(converted_range_version_list[1])
        if cpe_version_list != []:
            one_version_range_point_set = map_one_version_range(cpe_version_list, symbol, range_boundary_version)
        else:
            one_version_range_point_set.add(range_boundary_version)

    if converted_range_version_list[2] != set():
        version_bound_1, symbol_1, symbol_2, version_bound_2 = parse_two_range_version_format(converted_range_version_list[2])
        if cpe_version_list != []:
            two_version_range_point_set = map_two_version_range(cpe_version_list, version_bound_1, symbol_1, symbol_2,
                                                                version_bound_2)
            # if two_version_range_point_set != set():
            #     print(two_version_range_point_set)
        else:
            two_version_range_point_set.add(symbol_1)
            two_version_range_point_set.add(symbol_2)

    version_point_set = version_point_set.union(one_version_range_point_set).union(two_version_range_point_set)
    version_point_set = normalize_point_set(version_point_set)
    return version_point_set


def get_cpe_version_list(software):
    cpe_version_list = []
    if software in cpe_software_version_dict:
        cpe_version_list = cpe_software_version_dict[software]
        # print('found', software)
    else:
        matched_software = get_the_matched_software_name_from_cpe(software)
        if matched_software != '':
            cpe_version_list = cpe_software_version_dict[matched_software]
        # else:
        #     print(2, software, ' not in cpe dic !!')
    return cpe_version_list


def get_the_matched_software_name_from_cpe(software):
    matched_cpe_software = ''
    matched_cpe_software_list = []
    software_split = software.split()
    for key in cpe_software_version_dict:
        # if key == 'apple mac os x':
        #     print(key)
        if sublist(software_split, key.split()):
            matched_cpe_software_list.append(key)

    if matched_cpe_software_list != []:
        matched_cpe_software = get_the_shortest_one(matched_cpe_software_list)
    return matched_cpe_software


def sublist(ls1, ls2):
    return all(i in ls2 for i in ls1)


def get_the_shortest_one(matched_cpe_software_list):
    matched_software = ''
    for software in matched_cpe_software_list:
        if matched_software == '':
            matched_software = software
        elif len(software) < len(matched_software):
            matched_software = software
    return matched_software


def normalize_point_set(version_point_set):
    # CVE-2000-0699 securityfocus_official not loosely matched
    # {'10.20', '11.00'}
    # {'10.20', '11.0'}

    # CVE-2006-5304 securityfocus_official not loosely matched
    # {'1.0.0'}
    # {'1.0'}

    # CVE-2006-5302 edb not loosely matched
    # {'1.0000'}
    # {'1.0'}
    new_set = set()
    for version_point in version_point_set:
        change_flg = True
        while change_flg:
            # print(version_point, '  |  ', version_point_set)
            version_point, change_flg = normalize_point(version_point)
            # print(change_flg)
        new_set.add(remove_zero(get_dot_word_or_int_word(version_point)))
        # print(new_set)
    for elem in ['0', '']:
        if elem in new_set:
            new_set.remove(elem)

    return new_set


def normalize_point(version_point):
    change_flg = False
    for ii in range(5):
        replace_str = '.' + '0' * ii
        if version_point.endswith(replace_str):
            new = version_point[:-ii-1]
            change_flg = True
            return new, change_flg
        else:
            change_flg = False
    return version_point, change_flg


def parse_one_range_version_format(converted_range_version_set):
    range_tuple = tuple()
    for i in converted_range_version_set:
        range_tuple = i
    # {('<=', '11.3')}
    if converted_range_version_set == set():
        return set()
    symbol, range_boundary_version = range_tuple
    return symbol, range_boundary_version


def map_one_version_range(cpe_version_list, symbol, range_boundary_version):
    one_version_range_point_set = set()
    cpe_versions_matched = []

    if range_boundary_version in cpe_version_list:
        version_range_idx = cpe_version_list.index(range_boundary_version)
        # print('found version at index: ', version_range_idx)

        if symbol == '<=':
            cpe_versions_matched = cpe_version_list[:version_range_idx + 1]
        elif symbol == '<':
            cpe_versions_matched = cpe_version_list[:version_range_idx]
        elif symbol == '>=':
            cpe_versions_matched = cpe_version_list[version_range_idx:]
        elif symbol == '>':
            cpe_versions_matched = cpe_version_list[version_range_idx + 1:]
        else:
            print('ERROR!')
    else:
        cpe_versions_matched = get_matched_versions_for_boundary_not_in_cpe(cpe_version_list, symbol, range_boundary_version)
    one_version_range_point_set = one_version_range_point_set.union(set(cpe_versions_matched))

    return one_version_range_point_set


def remove_zero(version_point):
    # '008' --> '8'
    version_point_split = version_point.split('.')
    new_version_point = ''
    for v in version_point_split:
        if set(list(v)) != {'0'}:
            v = v.lstrip('0')
            new_version_point += '.' + v
        else:
            new_version_point += '.0'
    return new_version_point.lstrip('.')


def get_matched_versions_for_boundary_not_in_cpe(cpe_version_list, symbol, range_boundary_version):
    # for range boundaries that are not in cpe version list
    cpe_versions_matched = []
    matched_symbol_dict = {'<': ['<=', '<'], '>': ['>=', '>'], '=': ['=']}
    for cpe_version in cpe_version_list:
        compare_result = compare_single_version(cpe_version, range_boundary_version)
        if type(compare_result) == str:
            if symbol in matched_symbol_dict[compare_result]:
                cpe_versions_matched.append(cpe_version)
    return cpe_versions_matched


def parse_two_range_version_format(converted_range_version_set):
    range_tuple = tuple()
    for i in converted_range_version_set:
        range_tuple = i
    # {('<=', '11.3')}
    if converted_range_version_set == set():
        return set()
    # {('11.3', '<=', 'X', '<=', '12.2')}
    version_bound_1, symbol_1, _, symbol_2, version_bound_2 = range_tuple
    return version_bound_1, symbol_1, symbol_2, version_bound_2


def map_two_version_range(cpe_version_list, version_bound_1, symbol_1, symbol_2, version_bound_2):
    two_version_range_point_set = set()
    if symbol_1 == symbol_2 == '<=':
        pass
    else:
        print('ERROR 1')
    if version_bound_1 in cpe_version_list and version_bound_2 in cpe_version_list:
        version_range_idx_1 = cpe_version_list.index(version_bound_1)
        version_range_idx_2 = cpe_version_list.index(version_bound_2)
        cpe_versions_matched = cpe_version_list[version_range_idx_1: version_range_idx_2 + 1]
        two_version_range_point_set = two_version_range_point_set.union(set(cpe_versions_matched))
    return two_version_range_point_set


