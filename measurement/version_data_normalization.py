import copy
from patterns import one_version_range_keyword_list, two_version_range_keyword_list
from version_data_cleaning import remove_noise, get_dot_word_or_int_word


def format_version(cveid, db, software_1, software_2, version_list):
    # todo: debug separate_version function, which produces different result at each run
    version_list1 = separate_version(version_list)
    # version_list = new_separate_version(version_list)
    # if version_list1 == ['< =1.3.39 < = 2.0.61 < = 2.2.6']:
    # if version_list1 == ['> =8.0.0 < =8.1.1']:
    #     print('333', cveid, db, software_1, software_2)
    # for v in version_list1:
    #     if '8.0.0' in v and '8.1.1' in v:
    #         print(version_list1)
    version_list2 = convert_range(version_list1)
    if version_list2 is not None:
        version_list3 = convert_numeric_begin(version_list2)
        version_list4 = remove_noise(version_list3)
        return version_list4
    return [set(), set(), set()]


def separate_version(version_l):
    separate_version_list = []
    for v in version_l:
        split_version = v.split()
        indices = [i for i, x in enumerate(split_version) if x in ['and', 'or']]

        for i in indices:
            idx_after_and = i + 1
            if idx_after_and == len(split_version):
                split_version[i] = ','
            elif split_version[idx_after_and][0].isdigit():
                split_version[i] = ','
        version_str = ''
        for i in split_version:
            version_str += i + ' '
        separate_version_list.append(version_str.strip())
    final_version_list = []
    for v in separate_version_list:
        final_version_list += v.split(',')
    if '' in final_version_list:
        final_version_list.remove('')
    final_version_list = [v.strip() for v in final_version_list]
    return final_version_list


def convert_range(version_l, debug_mode=True):
    two_range_keyword_list = one_version_range_keyword_list

    if len(version_l) == 2:
        merged_two_versions = only_one_find_keyword(version_l, two_range_keyword_list)
        if merged_two_versions is not None:
            version_l = merged_two_versions

    one_version_range_list = []
    two_version_range_list = []
    version_l_copy = copy.deepcopy(version_l)
    # version_l_copy = version_l
    for v in version_l:
        find_keyword = False
        for pair in two_range_keyword_list:
            # tpdo: if match two keywords?
            if not find_keyword:
                keyword, mark = pair
                if v.find(keyword) != -1:
                    find_keyword = True
                    version_l_copy.remove(v)
                    # rest = v.replace(keyword, '').strip()
                    # one_version_range_list.append((mark, rest))
                    v_split = v.split(keyword)
                    if len(v_split) == 2:
                        rest_left, rest_right = v_split
                        if keyword[0] == ' ':
                            one_version_range_list.append((mark, rest_left))
                            if rest_right.strip() != '':
                                version_l_copy.append(rest_right.strip())
                        elif keyword[-1] == ' ':
                            one_version_range_list.append((mark, rest_right))
                            if rest_left.strip() != '':
                                version_l_copy.append(rest_left.strip())
                        elif rest_left == '':
                            one_version_range_list.append((mark, rest_right))
                        elif rest_right == '':
                            one_version_range_list.append((mark, rest_left))
                        elif keyword == '< =':
                            keyword_right = v[v.find(keyword):].split()
                            if len(keyword_right) > 0:
                                keyword_right = keyword_right[0]
                                keyword_right = get_dot_word_or_int_word(keyword_right)
                                one_version_range_list.append((mark, keyword_right))
                            elif debug_mode:
                                print('ERROR keyword right: ', keyword, '|', v)
                        elif keyword == '> =':
                            keyword_left = v[:v.find(keyword)].split()
                            if len(keyword_left) > 0:
                                keyword_left = keyword_left[-1]
                                keyword_left = get_dot_word_or_int_word(keyword_left)
                                one_version_range_list.append((mark, keyword_left))
                            elif debug_mode:
                                print('ERROR keyword right: ', keyword, '|', v)

                        elif debug_mode:
                            print('ERROR match two: ', keyword, '|', v)
                            # return
                    elif debug_mode:
                        print('ERROR two range: ', v)
                        # return
                    continue
                # # deal with ['1.0', 'and earlier']
                # elif v.find(keyword.strip()) != -1 and len(version_l) == 2:
                #     print(version_l, v)

        for pair in two_version_range_keyword_list:
            if not find_keyword:
                keyword, mark1, mark2 = pair
                if v.find(keyword) != -1:
                    v_split = v.split(keyword)
                    if len(v_split) == 2:
                        if len(v_split[0].split()) > 0:
                            left_v = v_split[0].split()[-1]
                            right_v = v_split[1].split()[0]
                            left_number = any(i.isdigit() for i in left_v)
                            right_number = any(i.isdigit() for i in right_v)
                            if left_number and right_number:
                                # print('number: ', '|', left_v, '|', right_v, '|', v)
                                find_keyword = True
                                two_version_range_list.append((left_v, mark1, 'X', mark2, right_v))
                                version_l_copy.remove(v)
                                continue
                            else:
                                one_version_range_list.append((mark2, right_v))
                                if debug_mode:
                                    print('convert two to one: ', '|', left_v, '|', right_v, '|', keyword, '|', v)
                        elif debug_mode:
                            print('ERROR v_split', v)
                            return
                    elif debug_mode:
                        print('one range: ', v)
                        return

    # before might be two, and three
    # three: up to, before, through

    single_version_list = []
    for v in version_l_copy:
        contains_number = any(i.isdigit() for i in v)
        if contains_number:
            single_version_list.append(v)

    return [single_version_list] + [one_version_range_list] + [two_version_range_list]


def only_one_find_keyword(version_l, two_range_keyword_list):
    found_list = [False, False]

    idx = 0
    for v in version_l:
        for pair in two_range_keyword_list:
            keyword, mark = pair
            if keyword.strip() == v:
                found_list[idx] = True
                if keyword[0] == ' ':
                    # print(1, version_l, [version_l[1 if idx == 0 else 0] + ' ' + v])
                    return [version_l[1 if idx == 0 else 0] + ' ' + v]
                else:
                    # print(2, version_l, [v + ' ' + version_l[1 if idx == 0 else 0]])
                    return [v + ' ' + version_l[1 if idx == 0 else 0]]
        idx += 1
    return None


def convert_numeric_begin(version_l):
    single_version_list, one_version_range_list, two_version_range_list = version_l
    one_version_range_list_copy = []
    for pair in one_version_range_list:
        mark, vv = pair
        vv_split = vv.split()
        version_xxx = contain_xxx(vv_split)
        if version_xxx is not None:
            one_version_range_list_copy.append((mark, version_xxx))
    # Todo: do the same for single_version_list and two_version_range_list
    return [single_version_list] + [one_version_range_list_copy] + [two_version_range_list]


def contain_xxx(spl):
    # input: version_s = ['firmware', '2.2.1']
    # output: idx = 1
    # judge contain versions like 1.2.3
    idx = 0
    for i in spl:
        tmp = i.replace('.', '')
        if any(j.isdigit() for j in tmp):
            return i
        idx += 1
